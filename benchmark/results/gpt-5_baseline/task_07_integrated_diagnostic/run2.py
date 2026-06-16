import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from datetime import datetime
from scipy.stats import wasserstein_distance
from urllib.request import urlretrieve

# ----------------------------
# Configuration
# ----------------------------
CASE_NAME = "sample.v3.LR.historical"
DATA_DIR = "./data/sample"
ELM_DIR = os.path.join(DATA_DIR, "e3sm", "lnd")
MOSART_DIR = os.path.join(DATA_DIR, "e3sm", "rof")
GAUGE_META_CSV = os.path.join(DATA_DIR, "obs", "gauge_metadata.csv")
OBS_STREAMFLOW_DIR = os.path.join(DATA_DIR, "obs", "streamflow")
BASIN_GEOJSON = os.path.join(DATA_DIR, "obs", "basin_polygons.geojson")

START_DATE = "1985-01-01"
END_DATE = "1989-12-31"

# Output directory (use absolute path specified by user)
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_07_integrated_diagnostic/run2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ILAMB_BASE = "https://www.ilamb.org/ILAMB-Data/DATA"
ILAMB_FILES = {
    "pr": {
        "dataset": "GPCCv2018",
        "filename": "pr.nc",
        "varname": "pr"
    },
    "evspsbl": {
        "dataset": "MODIS",
        "filename": "et_0.5x0.5.nc",
        "varname": "et"  # inside file variable name
    },
    "mrro": {
        "dataset": "LORA",
        "filename": "LORA.nc",
        "varname": "mrro"
    }
}
ILAMB_CACHE_DIR = os.path.join(OUTPUT_DIR, "ilamb_cache")
os.makedirs(ILAMB_CACHE_DIR, exist_ok=True)

# Basin selection: gauge_id to name mapping
BASINS = {
    3629000: "Amazon",
    4121801: "Missouri",
    4115200: "Columbia",
    6742900: "Danube",
    2969100: "Mekong",
    1159100: "Orange",
}

# ----------------------------
# Utilities
# ----------------------------
def print_status(msg):
    print(f"[INFO] {msg}")

def unit_to_mmd(data_array):
    """Convert flux units to mm/day if possible."""
    da = data_array
    units = (da.attrs.get("units", "") or "").lower().strip()
    if units in ["mm/day", "mm d-1", "mm/day ", "mm d^-1"]:
        return da
    if "kg" in units and "m-2" in units and "s-1" in units:
        # kg m-2 s-1 -> mm/s -> convert to mm/day
        return da * 86400.0
    if ("mm" in units and "/s" in units) or units == "mm s-1":
        return da * 86400.0
    if ("m/s" in units) or units == "m s-1":
        # m s-1 water equivalent -> convert to mm/day
        return da * 86400.0 * 1000.0
    # If units unknown but values look small, assume per second and convert to per day
    # Heuristic: if median < 0.01 and > 0, likely per second in mm/s
    try:
        med = float(da.median().values)
        if 0 < med < 0.01:
            return da * 86400.0
    except Exception:
        pass
    return da  # leave unchanged if uncertain

def ensure_lon_neg180_180(ds, lon_name="lon"):
    """Shift lon from [0,360) to (-180,180] and sort."""
    if lon_name not in ds.coords:
        return ds
    lons = ds[lon_name].values
    if np.nanmax(lons) > 180.0:
        new_lons = ((lons + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords({lon_name: new_lons})
        try:
            ds = ds.sortby(lon_name)
        except Exception:
            # xarray can fail if lon is not monotonic due to duplication; attempt argsort
            order = np.argsort(ds[lon_name].values)
            ds = ds.isel({lon_name: order})
    return ds

def load_geojson_polygons(geojson_path):
    """Load polygons from GeoJSON, keyed by GRDC gauge number (grdc_no)."""
    polygons_by_id = {}
    try:
        with open(geojson_path, 'r') as f:
            gj = json.load(f)
        features = gj.get("features", [])
        for feat in features:
            props = feat.get("properties", {})
            gid = props.get("grdc_no")
            if gid is None:
                continue
            try:
                gid = int(str(gid))
            except Exception:
                continue
            geom = feat.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            poly_list = []
            if gtype == "Polygon":
                # coords: [ [ [x,y], ... ] outer ring, [holes...] ]
                if coords:
                    outer = coords[0]
                    poly_list.append(np.array(outer))
            elif gtype == "MultiPolygon":
                for poly in coords:
                    if poly and poly[0]:
                        outer = poly[0]
                        poly_list.append(np.array(outer))
            if poly_list:
                polygons_by_id[gid] = poly_list
    except Exception as e:
        print_status(f"Error reading GeoJSON: {e}")
    return polygons_by_id

def grid_mask_for_polygon(lon2d, lat2d, polygon_rings):
    """Return boolean mask of points within any polygon ring (union).
    lon2d, lat2d: 2D arrays
    polygon_rings: list of arrays [[lon, lat], ...]
    """
    ny, nx = lon2d.shape
    points = np.vstack([lon2d.ravel(), lat2d.ravel()]).T
    mask_flat = np.zeros(points.shape[0], dtype=bool)
    for ring in polygon_rings:
        # Handle dateline crossing if necessary by shifting longitudes for PIP test
        path = Path(ring)
        mask_flat |= path.contains_points(points)
    return mask_flat.reshape(lon2d.shape)

def area_weighted_mean(field2d, lat2d, mask2d):
    """Compute area-weighted mean with cos(lat) weights."""
    # Ensure arrays
    arr = np.array(field2d)
    lat = np.array(lat2d)
    msk = np.array(mask2d).astype(bool)
    valid = np.isfinite(arr) & msk
    if not np.any(valid):
        return np.nan
    weights = np.cos(np.deg2rad(lat))
    weights = np.where(valid, weights, 0.0)
    num = np.nansum(arr * weights)
    den = np.nansum(weights)
    return float(num / den) if den > 0 else np.nan

def find_nearest_grid_point(lat2d, lon2d, target_lat, target_lon):
    """Find nearest grid index to a target lat/lon."""
    lat = np.array(lat2d)
    lon = np.array(lon2d)
    # Normalize longitude ranges for distance calc
    lon0 = target_lon
    # Simple Euclidean on lat/lon with latitude scaling
    dlat = lat - target_lat
    dlon = (lon - lon0) * np.cos(np.deg2rad(target_lat))
    dist2 = dlat**2 + dlon**2
    idx = np.unravel_index(np.nanargmin(dist2), lat.shape)
    return idx

def open_elm_monthly_files(varnames, start_date, end_date):
    """Open ELM monthly files for given period and variables."""
    # Build file list for period
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    months = pd.period_range(start=start, end=end, freq="M")
    files = []
    for per in months:
        y = per.year
        m = per.month
        fn = f"{CASE_NAME}.elm.h0.{y:04d}-{m:02d}.nc"
        path = os.path.join(ELM_DIR, fn)
        if os.path.exists(path):
            files.append(path)
    if not files:
        raise FileNotFoundError("No ELM monthly files found for the given period.")
    print_status(f"Found {len(files)} ELM files")
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=False, decode_times=True)
    # Ensure time bounds
    ds = ds.sel(time=slice(start_date, end_date))
    # Select only needed variables to reduce memory
    keep = [vn for vn in varnames if vn in ds.variables]
    missing = [vn for vn in varnames if vn not in ds.variables]
    if missing:
        print_status(f"Missing ELM variables: {missing}")
    return ds

def download_ilamb_file(var_key):
    info = ILAMB_FILES[var_key]
    url = f"{ILAMB_BASE}/{var_key}/{info['dataset']}/{info['filename']}"
    out_path = os.path.join(ILAMB_CACHE_DIR, f"{var_key}_{info['dataset']}_{info['filename']}")
    if not os.path.exists(out_path):
        try:
            print_status(f"Downloading {url}")
            urlretrieve(url, out_path)
        except Exception as e:
            print_status(f"Failed to download {url}: {e}")
            return None
    return out_path

def open_ilamb_dataset(var_key):
    fpath = download_ilamb_file(var_key)
    if fpath is None or not os.path.exists(fpath):
        return None
    try:
        ds = xr.open_dataset(fpath, decode_times=True)
        # Normalize lon to -180..180
        if 'lon' in ds.coords:
            ds = ensure_lon_neg180_180(ds, 'lon')
        elif 'longitude' in ds.coords:
            ds = ds.rename({'longitude': 'lon'})
            ds = ensure_lon_neg180_180(ds, 'lon')
        if 'lat' not in ds.coords and 'latitude' in ds.coords:
            ds = ds.rename({'latitude': 'lat'})
        return ds
    except Exception as e:
        print_status(f"Error opening ILAMB dataset {var_key}: {e}")
        return None

def select_time_mean_to_period(da, start_date, end_date):
    """Select the time slice and compute time-mean for specified period if time dimension exists."""
    if 'time' in da.dims:
        try:
            da_sel = da.sel(time=slice(start_date, end_date))
            if da_sel.time.size == 0:
                print_status("Requested period not in dataset; using full time range.")
                da_sel = da
            tm = da_sel.mean(dim='time', skipna=True)
        except Exception:
            tm = da.mean(dim='time', skipna=True)
    else:
        tm = da
    return tm

def get_lat_lon_from_dataarray(da):
    """Return 2D arrays lat2d, lon2d corresponding to dataarray spatial grid."""
    lat_name = None
    lon_name = None
    for name in ['lat', 'latitude', 'yc', 'y']:
        if name in da.coords:
            lat_name = name
            break
    for name in ['lon', 'longitude', 'xc', 'x']:
        if name in da.coords:
            lon_name = name
            break
    if lat_name is None or lon_name is None:
        # Try from dataset variables
        for nm in da.dims:
            if 'lat' in nm.lower():
                lat_name = nm
            if 'lon' in nm.lower():
                lon_name = nm
    lat = da[lat_name]
    lon = da[lon_name]
    # If they are 1D, meshgrid to 2D
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d = lat.values
        lon2d = lon.values
    else:
        # Fallback to attempt meshgrid
        try:
            lon2d, lat2d = np.meshgrid(lon.values, lat.values)
        except Exception:
            raise ValueError("Unable to derive 2D lat/lon")
    return lat2d, lon2d, lat_name, lon_name

# ----------------------------
# Part 1: Extract ELM climatological fields (P, ET, Q)
# ----------------------------
print_status("Part 1 — Extracting ELM climatological fields (P, ET, Q)")
elm_vars_needed = ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]

try:
    ds_elm = open_elm_monthly_files(elm_vars_needed, START_DATE, END_DATE)
except Exception as e:
    print_status(f"Failed to open ELM files: {e}")
    ds_elm = None

P_elm_mean = None
ET_elm_mean = None
Q_elm_mean = None
lat2d_elm = lon2d_elm = None
if ds_elm is not None:
    try:
        # Compute components
        # Some datasets may use different var names (lowercase). Handle gracefully.
        def get_var(ds, names):
            for nm in names:
                if nm in ds:
                    return ds[nm]
                if nm.lower() in ds:
                    return ds[nm.lower()]
            return None
        RAIN = get_var(ds_elm, ["RAIN"])
        SNOW = get_var(ds_elm, ["SNOW"])
        QVEGE = get_var(ds_elm, ["QVEGE"])
        QVEGT = get_var(ds_elm, ["QVEGT"])
        QSOIL = get_var(ds_elm, ["QSOIL"])
        QRUNOFF = get_var(ds_elm, ["QRUNOFF"])

        P_flux = None
        if RAIN is not None and SNOW is not None:
            P_flux = RAIN + SNOW
        elif RAIN is not None:
            P_flux = RAIN
        elif SNOW is not None:
            P_flux = SNOW

        ET_flux = None
        et_list = [Q for Q in [QVEGE, QVEGT, QSOIL] if Q is not None]
        if et_list:
            ET_flux = sum(et_list)

        # Units conversion for fluxes to mm/day
        if P_flux is not None:
            P_flux = unit_to_mmd(P_flux)
            P_elm_mean = select_time_mean_to_period(P_flux, START_DATE, END_DATE)
        if ET_flux is not None:
            ET_flux = unit_to_mmd(ET_flux)
            ET_elm_mean = select_time_mean_to_period(ET_flux, START_DATE, END_DATE)
        if QRUNOFF is not None:
            Q_flux = unit_to_mmd(QRUNOFF)
            Q_elm_mean = select_time_mean_to_period(Q_flux, START_DATE, END_DATE)

        # Extract lat/lon
        ref_da = None
        for da in [P_elm_mean, ET_elm_mean, Q_elm_mean]:
            if da is not None:
                ref_da = da
                break
        if ref_da is not None:
            lat2d_elm, lon2d_elm, elm_lat_name, elm_lon_name = get_lat_lon_from_dataarray(ref_da)
            # Normalize longitudes to -180..180
            lon2d_elm = ((lon2d_elm + 180.0) % 360.0) - 180.0
    except Exception as e:
        print_status(f"Error computing ELM climatologies: {e}")

# ----------------------------
# Part 2: Fetch and extract observation fields from ILAMB
# ----------------------------
print_status("Part 2 — Fetching ILAMB observation fields and computing climatologies")
obs_fields = {}  # dict: key -> DataArray time mean
for key in ["pr", "evspsbl", "mrro"]:
    ds = open_ilamb_dataset(key)
    if ds is None:
        continue
    varname = ILAMB_FILES[key]["varname"]
    if varname not in ds.variables:
        print_status(f"Variable {varname} not found in ILAMB {key}")
        continue
    da = ds[varname]
    # Normalize units
    da = unit_to_mmd(da)
    # Select period mean
    da_mean = select_time_mean_to_period(da, START_DATE, END_DATE)
    obs_fields[key] = da_mean

# ----------------------------
# Part 3: Clip to basin means for model and observation fields
# ----------------------------
print_status("Part 3 — Computing basin-averaged means for model and observations")
polygons_by_id = load_geojson_polygons(BASIN_GEOJSON)

basin_results = []  # list of dicts per basin with mean P,ET,Q for model and obs
for gid, name in BASINS.items():
    record = {
        "gauge_id": gid,
        "basin_name": name,
        "P_model_mmday": np.nan,
        "ET_model_mmday": np.nan,
        "Q_model_mmday": np.nan,
        "P_obs_mmday": np.nan,
        "ET_obs_mmday": np.nan,
        "Q_obs_mmday": np.nan,
        "WB_residual_model_mmday": np.nan
    }
    poly_rings = polygons_by_id.get(gid)
    if poly_rings is None:
        print_status(f"No polygon found for basin {name} ({gid})")
        basin_results.append(record)
        continue

    # Model (ELM) means
    try:
        if P_elm_mean is not None:
            lat2d, lon2d, _, _ = get_lat_lon_from_dataarray(P_elm_mean)
            lon2d = ((lon2d + 180.0) % 360.0) - 180.0
            mask = grid_mask_for_polygon(lon2d, lat2d, poly_rings)
            record["P_model_mmday"] = area_weighted_mean(P_elm_mean.values, lat2d, mask)
        if ET_elm_mean is not None:
            lat2d, lon2d, _, _ = get_lat_lon_from_dataarray(ET_elm_mean)
            lon2d = ((lon2d + 180.0) % 360.0) - 180.0
            mask = grid_mask_for_polygon(lon2d, lat2d, poly_rings)
            record["ET_model_mmday"] = area_weighted_mean(ET_elm_mean.values, lat2d, mask)
        if Q_elm_mean is not None:
            lat2d, lon2d, _, _ = get_lat_lon_from_dataarray(Q_elm_mean)
            lon2d = ((lon2d + 180.0) % 360.0) - 180.0
            mask = grid_mask_for_polygon(lon2d, lat2d, poly_rings)
            record["Q_model_mmday"] = area_weighted_mean(Q_elm_mean.values, lat2d, mask)
        # Water balance residual
        if np.isfinite(record["P_model_mmday"]) and np.isfinite(record["ET_model_mmday"]) and np.isfinite(record["Q_model_mmday"]):
            record["WB_residual_model_mmday"] = record["P_model_mmday"] - record["ET_model_mmday"] - record["Q_model_mmday"]
    except Exception as e:
        print_status(f"Error computing model basin mean for {name}: {e}")

    # Observations
    try:
        # Precipitation
        if "pr" in obs_fields:
            da = obs_fields["pr"]
            lat2d, lon2d, _, _ = get_lat_lon_from_dataarray(da)
            lon2d = ((lon2d + 180.0) % 360.0) - 180.0
            mask = grid_mask_for_polygon(lon2d, lat2d, poly_rings)
            record["P_obs_mmday"] = area_weighted_mean(da.values, lat2d, mask)
        # ET (MODIS et)
        if "evspsbl" in obs_fields:
            da = obs_fields["evspsbl"]
            lat2d, lon2d, _, _ = get_lat_lon_from_dataarray(da)
            lon2d = ((lon2d + 180.0) % 360.0) - 180.0
            mask = grid_mask_for_polygon(lon2d, lat2d, poly_rings)
            record["ET_obs_mmday"] = area_weighted_mean(da.values, lat2d, mask)
        # Runoff (LORA mrro)
        if "mrro" in obs_fields:
            da = obs_fields["mrro"]
            lat2d, lon2d, _, _ = get_lat_lon_from_dataarray(da)
            lon2d = ((lon2d + 180.0) % 360.0) - 180.0
            mask = grid_mask_for_polygon(lon2d, lat2d, poly_rings)
            record["Q_obs_mmday"] = area_weighted_mean(da.values, lat2d, mask)
    except Exception as e:
        print_status(f"Error computing obs basin mean for {name}: {e}")
    basin_results.append(record)

# Save basin means table
basin_means_df = pd.DataFrame(basin_results)
try:
    out_csv = os.path.join(OUTPUT_DIR, "basin_means_model_obs.csv")
    basin_means_df.to_csv(out_csv, index=False)
    print_status(f"Saved basin means to {out_csv}")
except Exception as e:
    print_status(f"Failed to save basin means CSV: {e}")

# ----------------------------
# Part 4: Streamflow FDC metrics
# ----------------------------
print_status("Part 4 — Computing streamflow FDC metrics")

# Load MOSART grid from one monthly file to get lat/lon and also later for fallback
mosart_sample_file = None
try:
    sample_month_file = os.path.join(MOSART_DIR, f"{CASE_NAME}.mosart.h0.1985-01.nc")
    if os.path.exists(sample_month_file):
        mosart_sample_file = sample_month_file
    else:
        # Try any month in the period
        cand = glob.glob(os.path.join(MOSART_DIR, f"{CASE_NAME}.mosart.h0.*.nc"))
        if cand:
            mosart_sample_file = cand[0]
except Exception:
    pass

mosart_lat2d = mosart_lon2d = None
mosart_grid_ds = None
if mosart_sample_file:
    try:
        mosart_grid_ds = xr.open_dataset(mosart_sample_file)
        # Determine lat/lon variables
        lat_var = None
        lon_var = None
        for v in ['lat', 'latitude']:
            if v in mosart_grid_ds:
                lat_var = v
                break
            if v in mosart_grid_ds.coords:
                lat_var = v
                break
        for v in ['lon', 'longitude']:
            if v in mosart_grid_ds:
                lon_var = v
                break
            if v in mosart_grid_ds.coords:
                lon_var = v
                break
        if lat_var is None or lon_var is None:
            # some datasets store lat/lon as variables
            if 'LAT' in mosart_grid_ds: lat_var = 'LAT'
            if 'LON' in mosart_grid_ds: lon_var = 'LON'
        lat_da = mosart_grid_ds[lat_var]
        lon_da = mosart_grid_ds[lon_var]
        if lat_da.ndim == 1 and lon_da.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_da.values, lat_da.values)
        else:
            lat2d = lat_da.values
            lon2d = lon_da.values
        mosart_lat2d = lat2d
        mosart_lon2d = ((lon2d + 180.0) % 360.0) - 180.0
    except Exception as e:
        print_status(f"Failed to read MOSART grid: {e}")

# Load gauge metadata
try:
    gauge_meta = pd.read_csv(GAUGE_META_CSV)
except Exception as e:
    print_status(f"Failed to read gauge metadata: {e}")
    gauge_meta = pd.DataFrame(columns=['gauge_id', 'lat', 'lon', 'area_km2', 'river_name'])

# Helper to load model discharge time series near a gauge location
def load_mosart_discharge_timeseries(lat, lon, start_date, end_date):
    """Return pandas Series (time, discharge m3/s) for nearest MOSART grid point."""
    var_candidates = ["RIVER_DISCHARGE_OVER_LAND_LIQ", "RIVER_DISCHARGE_OVER_LAND"]
    if mosart_lat2d is None or mosart_lon2d is None:
        print_status("MOSART grid not available")
        return None
    try:
        iy, ix = find_nearest_grid_point(mosart_lat2d, mosart_lon2d, lat, lon)
    except Exception as e:
        print_status(f"Failed to find nearest grid cell: {e}")
        return None

    # Try daily h1 files
    daily_files = sorted(glob.glob(os.path.join(MOSART_DIR, f"{CASE_NAME}.mosart.h1.*-00000.nc")))
    # Filter to date range
    daily_files = [f for f in daily_files if (f[-20:-11] >= START_DATE[:10] and f[-20:-11] <= END_DATE[:10]) or True]  # Keep all then slice later
    ser = None
    if daily_files:
        try:
            ds = xr.open_mfdataset(daily_files, combine='by_coords', decode_times=True)
            varname = None
            for v in var_candidates:
                if v in ds.variables:
                    varname = v
                    break
            if varname is None:
                # Try case-insensitive
                for v in ds.variables:
                    if v.lower() == "river_discharge_over_land_liq":
                        varname = v
                        break
            if varname is None:
                print_status("Discharge variable not found in daily files; will try monthly.")
            else:
                da = ds[varname]
                # ensure lon/lat dims order to index iy,ix
                # da dims likely time, y, x or time, nlat, nlon
                if 'time' not in da.dims:
                    print_status("Daily files without time dimension; skip.")
                else:
                    # Select nearest grid cell
                    # Assume dims order time, y, x
                    other_dims = list(da.dims)
                    time_dim = 'time'
                    ydim = None
                    xdim = None
                    for d in other_dims:
                        if d != time_dim:
                            if ydim is None: ydim = d
                            elif xdim is None: xdim = d
                    if ydim is None or xdim is None:
                        # Try using indices explicitly
                        cell_ts = da.isel({da.dims[1]: iy, da.dims[2]: ix})
                    else:
                        cell_ts = da.isel({ydim: iy, xdim: ix})
                    # Convert to pandas series
                    ts = cell_ts.to_series()
                    ts = ts.sort_index()
                    # Slice to desired period
                    ts = ts.loc[(ts.index >= pd.to_datetime(start_date)) & (ts.index <= pd.to_datetime(end_date))]
                    ser = ts
        except Exception as e:
            print_status(f"Failed reading daily MOSART files: {e}")

    # Fallback to monthly h0 files
    if ser is None:
        try:
            months = pd.period_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="M")
            files = []
            for per in months:
                y = per.year
                m = per.month
                fn = f"{CASE_NAME}.mosart.h0.{y:04d}-{m:02d}.nc"
                path = os.path.join(MOSART_DIR, fn)
                if os.path.exists(path):
                    files.append(path)
            if not files and mosart_sample_file:
                files = [mosart_sample_file]
            if files:
                ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True)
                varname = None
                for v in var_candidates:
                    if v in ds.variables:
                        varname = v
                        break
                if varname is None:
                    for v in ds.variables:
                        if "discharge" in v.lower():
                            varname = v
                            break
                da = ds[varname]
                # Identify y/x dims
                other_dims = list(da.dims)
                time_dim = 'time' if 'time' in da.dims else da.dims[0]
                ydim = None
                xdim = None
                for d in other_dims:
                    if d != time_dim:
                        if ydim is None: ydim = d
                        elif xdim is None: xdim = d
                if ydim is None or xdim is None:
                    cell_ts = da.isel({da.dims[1]: iy, da.dims[2]: ix})
                else:
                    cell_ts = da.isel({ydim: iy, xdim: ix})
                ts = cell_ts.to_series()
                ts = ts.sort_index()
                ts = ts.loc[(ts.index >= pd.to_datetime(start_date)) & (ts.index <= pd.to_datetime(end_date))]
                # If monthly, upsample to daily by repeating values for each day of month
                if ts.index.inferred_freq in [None, 'M', 'MS', 'ME'] or (len(ts) < 50):
                    ts_daily = ts.resample("D").ffill()
                    ser = ts_daily
                else:
                    ser = ts
        except Exception as e:
            print_status(f"Failed reading monthly MOSART files: {e}")
    return ser

# Compute FDC metrics per basin gauge
fdc_records = []
for gid, name in BASINS.items():
    rec = {
        "gauge_id": gid,
        "basin_name": name,
        "streamflow_model_sum_m3s_days": np.nan,
        "streamflow_obs_sum_m3s_days": np.nan,
        "streamflow_volume_bias_frac": np.nan,
        "wasserstein_distance_m3s": np.nan,
        "n_days": 0
    }
    # Gauge metadata
    gmeta = gauge_meta[gauge_meta['gauge_id'] == gid]
    if gmeta.empty:
        # Try type conversion
        try:
            gmeta = gauge_meta[gauge_meta['gauge_id'].astype(int) == gid]
        except Exception:
            pass
    if gmeta.empty:
        print_status(f"No gauge metadata found for {gid}")
        fdc_records.append(rec)
        continue
    lat = float(gmeta.iloc[0]['lat'])
    lon = float(gmeta.iloc[0]['lon'])

    # Observed discharge time series
    obs_csv = os.path.join(OBS_STREAMFLOW_DIR, f"{gid}.csv")
    try:
        obs_df = pd.read_csv(obs_csv, parse_dates=['date'])
        obs_df = obs_df.set_index('date').sort_index()
        obs_df = obs_df.loc[(obs_df.index >= pd.to_datetime(START_DATE)) & (obs_df.index <= pd.to_datetime(END_DATE))]
        obs_ts = obs_df['discharge_m3s'].astype(float)
    except Exception as e:
        print_status(f"Failed to read observed streamflow for {gid}: {e}")
        obs_ts = None

    # Model discharge time series
    mod_ts = load_mosart_discharge_timeseries(lat, lon, START_DATE, END_DATE)

    # Align and compute metrics
    if obs_ts is not None and mod_ts is not None and len(obs_ts) > 0 and len(mod_ts) > 0:
        # Align on overlapping dates
        df = pd.DataFrame({'obs': obs_ts}).join(pd.DataFrame({'mod': mod_ts}), how='inner')
        df = df.dropna()
        n_days = len(df)
        if n_days > 0:
            sum_mod = df['mod'].sum()
            sum_obs = df['obs'].sum()
            vol_bias = (sum_mod - sum_obs) / sum_obs if sum_obs != 0 else np.nan
            try:
                wd = wasserstein_distance(df['mod'].values, df['obs'].values)
            except Exception:
                wd = np.nan
            rec.update({
                "streamflow_model_sum_m3s_days": float(sum_mod),
                "streamflow_obs_sum_m3s_days": float(sum_obs),
                "streamflow_volume_bias_frac": float(vol_bias),
                "wasserstein_distance_m3s": float(wd),
                "n_days": int(n_days)
            })
    else:
        print_status(f"Missing streamflow series for basin {name} ({gid})")
    fdc_records.append(rec)

# Save streamflow metrics
fdc_df = pd.DataFrame(fdc_records)
try:
    out_csv = os.path.join(OUTPUT_DIR, "streamflow_fdc_metrics.csv")
    fdc_df.to_csv(out_csv, index=False)
    print_status(f"Saved streamflow FDC metrics to {out_csv}")
except Exception as e:
    print_status(f"Failed to save streamflow metrics CSV: {e}")

# ----------------------------
# Part 5: Combine and visualize
# ----------------------------
print_status("Part 5 — Combining metrics and generating visualizations")

# Merge basin means and FDC metrics
summary_df = basin_means_df.merge(fdc_df[['gauge_id', 'streamflow_volume_bias_frac', 'wasserstein_distance_m3s']], on='gauge_id', how='left')

# Compute biases (model - obs) for P, ET, Q
summary_df['P_bias_mmday'] = summary_df['P_model_mmday'] - summary_df['P_obs_mmday']
summary_df['ET_bias_mmday'] = summary_df['ET_model_mmday'] - summary_df['ET_obs_mmday']
summary_df['Q_bias_mmday'] = summary_df['Q_model_mmday'] - summary_df['Q_obs_mmday']

# Save summary table
try:
    out_csv = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    summary_df.to_csv(out_csv, index=False)
    print_status(f"Saved summary metrics to {out_csv}")
except Exception as e:
    print_status(f"Failed to save summary metrics CSV: {e}")

# Bar chart comparing P, ET, Q for each basin (model vs obs)
try:
    vars_to_plot = ['P', 'ET', 'Q']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()
    for i, (gid, name) in enumerate(BASINS.items()):
        ax = axes[i]
        sub = summary_df[summary_df['gauge_id'] == gid]
        if sub.empty:
            ax.set_title(f"{name}\n(no data)")
            ax.axis('off')
            continue
        model_vals = [sub[f"{v}_model_mmday"].iloc[0] for v in vars_to_plot]
        obs_vals = [sub[f"{v}_obs_mmday"].iloc[0] for v in vars_to_plot]
        x = np.arange(len(vars_to_plot))
        width = 0.35
        ax.bar(x - width/2, obs_vals, width, label='Obs')
        ax.bar(x + width/2, model_vals, width, label='Model')
        ax.set_xticks(x)
        ax.set_xticklabels(vars_to_plot)
        ax.set_ylabel("mm/day")
        ax.set_title(f"{name}")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend()
    # Hide any extra axes if any
    for j in range(len(BASINS), len(axes)):
        axes[j].axis('off')
    out_fig = os.path.join(OUTPUT_DIR, "bars_P_ET_Q_by_basin.png")
    fig.suptitle("Model vs Observation Basin Means (1985-1989)")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print_status(f"Saved bar chart to {out_fig}")
except Exception as e:
    print_status(f"Failed to create bar chart: {e}")

# Radar chart per basin for diagnostics
try:
    # Metrics: absolute P_bias, ET_bias, Q_bias, WB_residual_model, |streamflow_volume_bias|, Wasserstein
    metrics_keys = ['P_bias_mmday', 'ET_bias_mmday', 'Q_bias_mmday', 'WB_residual_model_mmday', 'streamflow_volume_bias_frac', 'wasserstein_distance_m3s']
    metric_labels = ['|P bias| (mm/d)', '|ET bias| (mm/d)', '|Q bias| (mm/d)', '|WB resid| (mm/d)', '|Q vol bias| (frac)', 'Wasserstein (m3/s)']

    # Prepare normalized values (0-1) across basins for consistent axes
    df_abs = summary_df.copy()
    df_abs['P_bias_mmday'] = df_abs['P_bias_mmday'].abs()
    df_abs['ET_bias_mmday'] = df_abs['ET_bias_mmday'].abs()
    df_abs['Q_bias_mmday'] = df_abs['Q_bias_mmday'].abs()
    df_abs['WB_residual_model_mmday'] = df_abs['WB_residual_model_mmday'].abs()
    df_abs['streamflow_volume_bias_frac'] = df_abs['streamflow_volume_bias_frac'].abs()
    # Replace inf with nan then fill
    for mk in metrics_keys:
        df_abs[mk] = df_abs[mk].replace([np.inf, -np.inf], np.nan)

    max_vals = []
    for mk in metrics_keys:
        max_val = df_abs[mk].max(skipna=True)
        if not np.isfinite(max_val) or max_val == 0:
            max_val = 1.0
        max_vals.append(max_val)

    def normalize(values, max_values):
        arr = np.array(values, dtype=float)
        out = []
        for v, m in zip(arr, max_values):
            if not np.isfinite(v):
                out.append(0.0)
            else:
                out.append(float(v) / float(m))
        return out

    # Create radar chart subplots
    N = len(metrics_keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(18, 10))
    subplot_positions = [(2,3,i+1) for i in range(6)]
    for i, (gid, name) in enumerate(BASINS.items()):
        ax = plt.subplot(2, 3, i+1, polar=True)
        sub = df_abs[df_abs['gauge_id'] == gid]
        if sub.empty:
            ax.set_title(f"{name}\n(no data)")
            continue
        values = [sub[mk].iloc[0] for mk in metrics_keys]
        values_norm = normalize(values, max_vals)
        values_norm += values_norm[:1]
        ax.plot(angles, values_norm, linewidth=2, label=name)
        ax.fill(angles, values_norm, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_yticklabels([])
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle("Basin Diagnostics Radar (normalized; lower is better)", fontsize=14)
    out_fig = os.path.join(OUTPUT_DIR, "radar_diagnostic_by_basin.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print_status(f"Saved radar chart to {out_fig}")
except Exception as e:
    print_status(f"Failed to create radar chart: {e}")

print_status("Analysis complete.")