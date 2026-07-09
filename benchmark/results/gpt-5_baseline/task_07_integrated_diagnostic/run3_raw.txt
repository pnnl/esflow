import os
import glob
import json
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from datetime import datetime
from urllib.request import urlretrieve

# Attempt to import shapely for polygon operations; fallback to matplotlib.path if not available
try:
    from shapely.geometry import shape, Point, Polygon, MultiPolygon
    from shapely.prepared import prep as shapely_prep
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False
    from matplotlib.path import Path as MplPath

# -------------------------
# Configuration and Paths
# -------------------------
CASE_NAME = "sample.v3.LR.historical"

# Data input locations
E3SM_BASE = "./data/sample/e3sm"
ELM_DIR = os.path.join(E3SM_BASE, "lnd")
MOSART_DIR = os.path.join(E3SM_BASE, "rof")

OBS_BASE = "./data/sample/obs"
GAUGE_META_PATH = os.path.join(OBS_BASE, "gauge_metadata.csv")
STREAMFLOW_DIR = os.path.join(OBS_BASE, "streamflow")
BASIN_GEOJSON = os.path.join(OBS_BASE, "basin_polygons.geojson")

ILAMB_BASE_URL = "https://www.ilamb.org/ILAMB-Data/DATA"
ILAMB_LOCAL_BASE = "./data/ilamb"

# Output directories
OUTDIR_REL = "./output/task07_integrated_diagnostic"
OUTDIR_ABS = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_07_integrated_diagnostic/run3_output"

os.makedirs(OUTDIR_REL, exist_ok=True)
os.makedirs(OUTDIR_ABS, exist_ok=True)

def save_to_outdirs(local_path_rel):
    try:
        base_name = os.path.basename(local_path_rel)
        dst_rel = os.path.join(OUTDIR_REL, base_name)
        dst_abs = os.path.join(OUTDIR_ABS, base_name)
        if os.path.isdir(local_path_rel):
            # Not used, but included for completeness
            if os.path.exists(dst_rel): shutil.rmtree(dst_rel)
            if os.path.exists(dst_abs): shutil.rmtree(dst_abs)
            shutil.copytree(local_path_rel, dst_rel)
            shutil.copytree(local_path_rel, dst_abs)
        else:
            shutil.copy2(local_path_rel, dst_rel)
            shutil.copy2(local_path_rel, dst_abs)
        print(f"Saved to: {dst_rel} and {dst_abs}")
    except Exception as e:
        print(f"Warning: Failed to save {local_path_rel} to output directories: {e}")

# -------------------------
# Basins and Period
# -------------------------
BASINS = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}
START_DATE = "1985-01-01"
END_DATE = "1989-12-31"

# -------------------------
# Utilities
# -------------------------

def generate_monthly_file_list(base_dir, component_dir, case, start_year=1985, end_year=1989, stream="h0"):
    files = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if component_dir == "lnd":
                pattern = os.path.join(base_dir, component_dir, f"{case}.elm.{stream}.{y:04d}-{m:02d}.nc")
            elif component_dir == "rof":
                pattern = os.path.join(base_dir, component_dir, f"{case}.mosart.{stream}.{y:04d}-{m:02d}.nc")
            else:
                continue
            if os.path.exists(pattern):
                files.append(pattern)
    return sorted(files)

def generate_daily_file_list(base_dir, case, start_year=1985, end_year=1989):
    files = []
    for y in range(start_year, end_year + 1):
        pattern = os.path.join(base_dir, f"{case}.mosart.h1.{y:04d}-*.nc")
        files.extend(sorted(glob.glob(pattern)))
    return files

def open_elm_monthly(case=CASE_NAME, start_year=1985, end_year=1989):
    files = generate_monthly_file_list(E3SM_BASE, "lnd", case, start_year, end_year, stream="h0")
    if not files:
        raise FileNotFoundError("No ELM monthly files found for the specified period.")
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)
    ds = ds.sel(time=slice(START_DATE, END_DATE))
    return ds

def convert_units_to_mm_per_day(da):
    # Attempt to infer units and convert to mm/day
    units = da.attrs.get("units", "").lower()
    # If mm/s or kg m-2 s-1, multiply by 86400
    if "s-1" in units or "/s" in units:
        return da * 86400.0
    # If mm/day already or unspecified, assume mm/day
    if "mm/day" in units or "mm d-1" in units or "mm/day" in units:
        return da
    # If mm/month, divide by days in month (approx use clim time or mean days)
    if "mm/month" in units or "mm mon-1" in units:
        # Attempt to get time dimension to compute per-day
        if "time" in da.dims:
            time = da["time"].to_index()
            days_in_month = time.days_in_month.values
            expanded = xr.DataArray(days_in_month, dims=["time"], coords={"time": da["time"]})
            return da / expanded
        else:
            # Fallback assume 30 days
            return da / 30.0
    # Default: return as is, assume mm/day
    return da

def ensure_lon_neg180_180(ds):
    lon_name = None
    for name in ["lon", "longitude", "Lon", "LON"]:
        if name in ds.coords:
            lon_name = name
            break
    if lon_name is None:
        # Try variable named lon
        for v in ds.variables:
            if v.lower() in ["lon", "longitude"]:
                lon_name = v
                break
    if lon_name is None:
        return ds

    lon = ds[lon_name]
    lon_vals = lon.values
    if np.nanmin(lon_vals) >= 0 and np.nanmax(lon_vals) > 180:
        lon_new = ((lon_vals + 180) % 360) - 180
        # Sort along lon dimension if 1D
        if lon.ndim == 1:
            order = np.argsort(lon_new)
            ds = ds.assign_coords({lon_name: lon_new}).isel({lon_name: order})
            for v in ds.data_vars:
                if lon_name in ds[v].dims and ds[v].dims.index(lon_name) == ds[v].dims.index(lon_name):
                    ds[v] = ds[v].isel({lon_name: order})
        else:
            ds = ds.assign_coords({lon_name: (lon.dims, lon_new)})
    return ds

def get_lat_lon_arrays(ds):
    lat_name = None
    lon_name = None
    for name in ["lat", "latitude", "LAT", "Lat"]:
        if name in ds.coords:
            lat_name = name
            break
    for name in ["lon", "longitude", "LON", "Lon"]:
        if name in ds.coords:
            lon_name = name
            break
    if lat_name is None or lon_name is None:
        # Try as variables
        for v in ds.variables:
            if v.lower() == "lat":
                lat_name = v
            if v.lower() == "lon":
                lon_name = v
    if lat_name is None or lon_name is None:
        raise KeyError("Latitude/Longitude coordinate names not found in dataset.")
    lat = ds[lat_name]
    lon = ds[lon_name]
    # Broadcast to 2D if 1D
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    else:
        lat2d = lat.values
        lon2d = lon.values
    return lat2d, lon2d, lat_name, lon_name

def download_ilamb(variable, dataset, filename, local_base=ILAMB_LOCAL_BASE):
    url = f"{ILAMB_BASE_URL}/{variable}/{dataset}/{filename}"
    local_dir = os.path.join(local_base, variable, dataset)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        try:
            print(f"Downloading {url} -> {local_path}")
            urlretrieve(url, local_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"Found existing ILAMB file: {local_path}")
    return local_path

def points_in_polygon_mask(lats2d, lons2d, polygon):
    # Ensure longitudes in -180 to 180 for consistency
    lons_adj = np.where(lons2d > 180, lons2d - 360, lons2d)
    mask = np.zeros(lats2d.shape, dtype=bool)
    if HAS_SHAPELY:
        prep_poly = shapely_prep(polygon)
        # Flatten and test
        flat_mask = np.array([prep_poly.contains(Point(x, y)) for x, y in zip(lons_adj.ravel(), lats2d.ravel())])
        mask = flat_mask.reshape(lats2d.shape)
    else:
        # Fallback using matplotlib.path
        if isinstance(polygon, Polygon):
            coords = np.array(polygon.exterior.coords)
        elif isinstance(polygon, MultiPolygon):
            # Use union of masks from all parts
            mask_total = np.zeros(lats2d.shape, dtype=bool)
            for geom in polygon.geoms:
                coords = np.array(geom.exterior.coords)
                path = MplPath(coords)
                pts = np.vstack([lons_adj.ravel(), lats2d.ravel()]).T
                m = path.contains_points(pts).reshape(lats2d.shape)
                mask_total |= m
            return mask_total
        else:
            raise ValueError("Unsupported polygon type for masking.")
        path = MplPath(coords)
        pts = np.vstack([lons_adj.ravel(), lats2d.ravel()]).T
        mask = path.contains_points(pts).reshape(lats2d.shape)
    return mask

def area_weighted_mean(field2d, lats2d, mask):
    # field2d: numpy array
    # lats2d: numpy array
    # mask: boolean mask
    arr = np.array(field2d)
    latr = np.array(lats2d)
    m = np.array(mask)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for field.")
    w = np.cos(np.deg2rad(latr)) * m
    valid = np.isfinite(arr) & m
    if np.nansum(w[valid]) == 0:
        return np.nan
    return np.nansum(arr[valid] * w[valid]) / np.nansum(w[valid])

def safe_plot_save(fig, filename_base):
    for outdir in [OUTDIR_REL, OUTDIR_ABS]:
        try:
            os.makedirs(outdir, exist_ok=True)
            figpath = os.path.join(outdir, filename_base)
            fig.savefig(figpath, dpi=150, bbox_inches="tight")
            print(f"Saved figure: {figpath}")
        except Exception as e:
            print(f"Warning: Failed to save figure {filename_base} to {outdir}: {e}")

# -------------------------
# Part 1: ELM model fields
# -------------------------
def compute_elm_climatology():
    ds = open_elm_monthly(CASE_NAME, 1985, 1989)
    # Variable names provided
    for var in ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]:
        if var not in ds:
            print(f"Warning: Variable {var} not found in ELM dataset.")
    P = (ds.get("RAIN", 0) + ds.get("SNOW", 0)) * 86400.0  # mm/day
    ET = (ds.get("QVEGE", 0) + ds.get("QVEGT", 0) + ds.get("QSOIL", 0)) * 86400.0  # mm/day
    Q = ds.get("QRUNOFF", xr.zeros_like(P)) * 86400.0  # mm/day
    P_mean = P.mean("time", skipna=True)
    ET_mean = ET.mean("time", skipna=True)
    Q_mean = Q.mean("time", skipna=True)
    P_mean.attrs["units"] = "mm/day"
    ET_mean.attrs["units"] = "mm/day"
    Q_mean.attrs["units"] = "mm/day"
    elm_clim = xr.Dataset({"P": P_mean, "ET": ET_mean, "Q": Q_mean})
    # Save NetCDF
    tmp_path = os.path.join(OUTDIR_REL, "elm_climatology_1985_1989.nc")
    try:
        elm_clim.to_netcdf(tmp_path)
        save_to_outdirs(tmp_path)
    except Exception as e:
        print(f"Warning: Failed to save ELM climatology NetCDF: {e}")
    return elm_clim

# -------------------------
# Part 2: Observations (ILAMB)
# -------------------------
def compute_obs_climatology():
    # Download files
    pr_path = download_ilamb("pr", "GPCCv2018", "pr.nc")
    et_path = download_ilamb("evspsbl", "MODIS", "et_0.5x0.5.nc")
    mrro_path = download_ilamb("mrro", "LORA", "LORA.nc")

    # Precipitation
    ds_pr = xr.open_dataset(pr_path, decode_times=True)
    ds_pr = ensure_lon_neg180_180(ds_pr)
    pr = ds_pr.get("pr")
    if pr is None:
        # Try other naming
        for v in ds_pr.data_vars:
            if v.lower() == "pr":
                pr = ds_pr[v]
                break
    pr = pr.sel(time=slice(START_DATE, END_DATE))
    pr = convert_units_to_mm_per_day(pr)
    pr_mean = pr.mean("time", skipna=True)
    pr_mean.attrs["units"] = "mm/day"

    # ET (MODIS)
    ds_et = xr.open_dataset(et_path, decode_times=True)
    ds_et = ensure_lon_neg180_180(ds_et)
    et = ds_et.get("et")
    if et is None:
        # fallback if named differently
        for v in ds_et.data_vars:
            if v.lower() in ["et", "evspsbl"]:
                et = ds_et[v]
                break
    et = et.sel(time=slice(START_DATE, END_DATE))
    et = convert_units_to_mm_per_day(et)
    et_mean = et.mean("time", skipna=True)
    et_mean.attrs["units"] = "mm/day"

    # Runoff (LORA)
    ds_q = xr.open_dataset(mrro_path, decode_times=True)
    ds_q = ensure_lon_neg180_180(ds_q)
    mrro = ds_q.get("mrro")
    if mrro is None:
        for v in ds_q.data_vars:
            if v.lower() == "mrro":
                mrro = ds_q[v]
                break
    mrro = mrro.sel(time=slice(START_DATE, END_DATE))
    mrro = convert_units_to_mm_per_day(mrro)
    mrro_mean = mrro.mean("time", skipna=True)
    mrro_mean.attrs["units"] = "mm/day"

    obs_clim = xr.Dataset({"pr": pr_mean, "et": et_mean, "mrro": mrro_mean})
    tmp_path = os.path.join(OUTDIR_REL, "obs_climatology_1985_1989.nc")
    try:
        obs_clim.to_netcdf(tmp_path)
        save_to_outdirs(tmp_path)
    except Exception as e:
        print(f"Warning: Failed to save observation climatology NetCDF: {e}")
    return obs_clim

# -------------------------
# Part 3: Basin means
# -------------------------
def load_basin_polygons(geojson_path):
    try:
        with open(geojson_path, "r") as f:
            gj = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load basin polygons: {e}")
    basin_polys = {}
    for feature in gj.get("features", []):
        properties = feature.get("properties", {})
        grdc_no = str(properties.get("grdc_no"))
        geom = feature.get("geometry")
        if geom is None:
            continue
        if HAS_SHAPELY:
            poly = shape(geom)
        else:
            # Minimal conversion for Polygon/MultiPolygon to shapely-like via matplotlib or back to shapely if available
            if geom["type"] == "Polygon":
                poly = Polygon(geom["coordinates"][0])
            elif geom["type"] == "MultiPolygon":
                geoms = [Polygon(coords[0]) for coords in geom["coordinates"]]
                poly = MultiPolygon(geoms)
            else:
                continue
        basin_polys[grdc_no] = poly
    return basin_polys

def compute_basin_means(elm_clim, obs_clim, basin_polys, basin_map):
    # Extract lat/lon for ELM and OBS grids
    elm_lats2d, elm_lons2d, elm_lat_name, elm_lon_name = get_lat_lon_arrays(elm_clim)
    obs_lats2d, obs_lons2d, obs_lat_name, obs_lon_name = get_lat_lon_arrays(obs_clim)

    results = []
    for basin_name, gauge_id in basin_map.items():
        poly = basin_polys.get(str(gauge_id))
        if poly is None:
            print(f"Warning: Polygon for basin {basin_name} (gauge {gauge_id}) not found.")
            p_m = et_m = q_m = p_o = et_o = q_o = np.nan
        else:
            # ELM Means
            mask_elm = points_in_polygon_mask(elm_lats2d, elm_lons2d, poly)
            p_m = area_weighted_mean(elm_clim["P"].values, elm_lats2d, mask_elm)
            et_m = area_weighted_mean(elm_clim["ET"].values, elm_lats2d, mask_elm)
            q_m = area_weighted_mean(elm_clim["Q"].values, elm_lats2d, mask_elm)

            # Obs Means
            mask_obs = points_in_polygon_mask(obs_lats2d, obs_lons2d, poly)
            p_o = area_weighted_mean(obs_clim["pr"].values, obs_lats2d, mask_obs)
            et_o = area_weighted_mean(obs_clim["et"].values, obs_lats2d, mask_obs)
            q_o = area_weighted_mean(obs_clim["mrro"].values, obs_lats2d, mask_obs)

        results.append({
            "basin": basin_name,
            "gauge_id": str(gauge_id),
            "P_model_mmday": p_m,
            "ET_model_mmday": et_m,
            "Q_model_mmday": q_m,
            "P_obs_mmday": p_o,
            "ET_obs_mmday": et_o,
            "Q_obs_mmday": q_o
        })
    return pd.DataFrame(results)

# -------------------------
# Part 4: Streamflow Metrics
# -------------------------
def find_nearest_grid_cell(lat2d, lon2d, target_lat, target_lon):
    # Adjust lon to grid convention
    lons = lon2d.copy()
    if np.nanmin(lons) >= 0 and np.nanmax(lons) > 180:
        lon0 = target_lon if target_lon >= 0 else target_lon + 360.0
    else:
        lon0 = target_lon
    # Compute simple distance metric
    dlat = lat2d - target_lat
    dlon = (lons - lon0) * np.cos(np.deg2rad(target_lat))
    dist2 = dlat ** 2 + dlon ** 2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return idx  # (iy, ix)

def get_mosart_lat_lon():
    # Use a monthly file to extract grid
    files = generate_monthly_file_list(E3SM_BASE, "rof", CASE_NAME, 1985, 1985, stream="h0")
    if not files:
        # Try other years
        files = glob.glob(os.path.join(MOSART_DIR, f"{CASE_NAME}.mosart.h0.*.nc"))
    if not files:
        raise FileNotFoundError("No MOSART monthly files found to extract grid.")
    ds = xr.open_dataset(files[0], decode_times=False)
    lat2d, lon2d, lat_name, lon_name = get_lat_lon_arrays(ds)
    ds.close()
    return lat2d, lon2d, (lat_name, lon_name)

def extract_mosart_timeseries_for_cell(iy, ix, start_date=START_DATE, end_date=END_DATE):
    varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    daily_files = generate_daily_file_list(MOSART_DIR, CASE_NAME, 1985, 1989)
    ts = None
    freq = None
    if daily_files:
        try:
            ds = xr.open_mfdataset(daily_files, combine="by_coords", decode_times=True)
            if varname not in ds:
                # Try to list variables
                candidates = [v for v in ds.data_vars if "DISCHARGE" in v.upper()]
                if candidates:
                    varname = candidates[0]
            var = ds[varname]
            # Identify spatial dims
            dims = var.dims
            # Find indices for lat/lon-like dims
            lat_dim = None
            lon_dim = None
            for d in dims:
                if d.lower() in ["lat", "nlat", "y", "jy", "nj", "ydim"]:
                    if lat_dim is None: lat_dim = d
                if d.lower() in ["lon", "nlon", "x", "ix", "ni", "xdim"]:
                    if lon_dim is None: lon_dim = d
            if lat_dim is None or lon_dim is None:
                # Assume 2 spatial dims after time
                non_time_dims = [d for d in dims if d != "time"]
                if len(non_time_dims) >= 2:
                    lat_dim, lon_dim = non_time_dims[:2]
            var_cell = var.isel({lat_dim: iy, lon_dim: ix})
            var_cell = var_cell.sel(time=slice(start_date, end_date))
            ts = var_cell.to_series()
            freq = "D"
            ds.close()
        except Exception as e:
            print(f"Warning: Failed to read daily MOSART files: {e}")
    if ts is None:
        # Fallback to monthly
        monthly_files = generate_monthly_file_list(E3SM_BASE, "rof", CASE_NAME, 1985, 1989, stream="h0")
        if not monthly_files:
            print("Warning: No MOSART monthly files available for streamflow extraction.")
            return None, None
        try:
            ds = xr.open_mfdataset(monthly_files, combine="by_coords", decode_times=True)
            var = ds.get(varname)
            if var is None:
                # Try matching pattern
                for v in ds.data_vars:
                    if "DISCHARGE" in v.upper():
                        var = ds[v]
                        break
            dims = var.dims
            lat_dim = None
            lon_dim = None
            for d in dims:
                if d.lower() in ["lat", "nlat", "y", "jy", "nj", "ydim"]:
                    if lat_dim is None: lat_dim = d
                if d.lower() in ["lon", "nlon", "x", "ix", "ni", "xdim"]:
                    if lon_dim is None: lon_dim = d
            if lat_dim is None or lon_dim is None:
                non_time_dims = [d for d in dims if d != "time"]
                if len(non_time_dims) >= 2:
                    lat_dim, lon_dim = non_time_dims[:2]
            var_cell = var.isel({lat_dim: iy, lon_dim: ix})
            var_cell = var_cell.sel(time=slice(start_date, end_date))
            ts = var_cell.to_series()
            freq = "M"
            ds.close()
        except Exception as e:
            print(f"Warning: Failed to read monthly MOSART files: {e}")
            return None, None
    ts.name = "Q_mosart_m3s"
    return ts, freq

def read_observed_gauge_timeseries(gauge_id, start_date=START_DATE, end_date=END_DATE):
    path = os.path.join(STREAMFLOW_DIR, f"{gauge_id}.csv")
    if not os.path.exists(path):
        print(f"Warning: Observed streamflow CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as e:
        print(f"Warning: Failed to read observed streamflow CSV {path}: {e}")
        return None
    df = df.set_index("date").sort_index()
    df = df.loc[start_date:end_date]
    # Ensure numeric
    ser = pd.to_numeric(df["discharge_m3s"], errors="coerce")
    ser.name = "Q_obs_m3s"
    return ser

def compute_streamflow_metrics(basin_map):
    # Load MOSART grid
    lat2d, lon2d, _ = get_mosart_lat_lon()
    # Load gauge metadata
    try:
        gmeta = pd.read_csv(GAUGE_META_PATH)
    except Exception as e:
        print(f"Warning: Failed to read gauge metadata: {e}")
        return pd.DataFrame(columns=["basin", "gauge_id", "streamflow_bias_rel", "wasserstein_distance"])
    rows = []
    for basin_name, gauge_id in basin_map.items():
        # Find gauge lat/lon
        gm = gmeta[gmeta["gauge_id"].astype(str) == str(gauge_id)]
        if gm.empty:
            print(f"Warning: Gauge metadata for {gauge_id} not found.")
            rows.append({"basin": basin_name, "gauge_id": str(gauge_id),
                         "streamflow_bias_rel": np.nan, "wasserstein_distance": np.nan})
            continue
        lat0 = float(gm.iloc[0]["lat"])
        lon0 = float(gm.iloc[0]["lon"])
        iy, ix = find_nearest_grid_cell(lat2d, lon2d, lat0, lon0)
        # Extract MOSART series
        q_mod, freq = extract_mosart_timeseries_for_cell(iy, ix, START_DATE, END_DATE)
        if q_mod is None:
            print(f"Warning: No model streamflow for basin {basin_name} ({gauge_id}).")
            rows.append({"basin": basin_name, "gauge_id": str(gauge_id),
                         "streamflow_bias_rel": np.nan, "wasserstein_distance": np.nan})
            continue
        q_obs = read_observed_gauge_timeseries(gauge_id, START_DATE, END_DATE)
        if q_obs is None:
            print(f"Warning: No observed streamflow for basin {basin_name} ({gauge_id}).")
            rows.append({"basin": basin_name, "gauge_id": str(gauge_id),
                         "streamflow_bias_rel": np.nan, "wasserstein_distance": np.nan})
            continue
        # Align frequencies
        if freq == "D":
            q_obs_aligned = q_obs
            q_mod_aligned = q_mod
        else:
            # Monthly mean
            q_obs_aligned = q_obs.resample("M").mean()
            q_mod_aligned = q_mod
        # Align in time index
        df = pd.concat([q_obs_aligned, q_mod_aligned], axis=1).dropna()
        if df.empty:
            print(f"Warning: No overlapping streamflow period for {basin_name} ({gauge_id}).")
            rows.append({"basin": basin_name, "gauge_id": str(gauge_id),
                         "streamflow_bias_rel": np.nan, "wasserstein_distance": np.nan})
            continue
        q_obs_vals = df["Q_obs_m3s"].values
        q_mod_vals = df["Q_mosart_m3s"].values
        # Volume bias (relative mean bias)
        if np.nanmean(q_obs_vals) == 0:
            streamflow_bias_rel = np.nan
        else:
            streamflow_bias_rel = (np.nanmean(q_mod_vals) - np.nanmean(q_obs_vals)) / np.nanmean(q_obs_vals)
        # Wasserstein distance
        try:
            wd = wasserstein_distance(q_obs_vals[np.isfinite(q_obs_vals)], q_mod_vals[np.isfinite(q_mod_vals)])
        except Exception:
            wd = np.nan
        rows.append({"basin": basin_name, "gauge_id": str(gauge_id),
                     "streamflow_bias_rel": streamflow_bias_rel, "wasserstein_distance": wd})
    return pd.DataFrame(rows)

# -------------------------
# Part 5: Combine and Visualize
# -------------------------
def make_bar_chart(df_metrics):
    # Plot model vs observation for P, ET, Q per basin
    basins = df_metrics["basin"].tolist()
    x = np.arange(len(basins))
    width = 0.25

    Pm = df_metrics["P_model_mmday"].values
    Po = df_metrics["P_obs_mmday"].values
    ETm = df_metrics["ET_model_mmday"].values
    ETo = df_metrics["ET_obs_mmday"].values
    Qm = df_metrics["Q_model_mmday"].values
    Qo = df_metrics["Q_obs_mmday"].values

    fig, ax = plt.subplots(figsize=(12, 6))
    # Offsets for 3 variables
    ax.bar(x - width, Po, width, label="P obs", color="#a6cee3")
    ax.bar(x - width, Pm, width, label="P model", color="#1f78b4", alpha=0.8, edgecolor="k")

    ax.bar(x, ETo, width, label="ET obs", color="#b2df8a")
    ax.bar(x, ETm, width, label="ET model", color="#33a02c", alpha=0.8, edgecolor="k")

    ax.bar(x + width, Qo, width, label="Q obs", color="#fb9a99")
    ax.bar(x + width, Qm, width, label="Q model", color="#e31a1c", alpha=0.8, edgecolor="k")

    ax.set_xticks(x)
    ax.set_xticklabels(basins, rotation=45, ha="right")
    ax.set_ylabel("mm/day")
    ax.set_title("Model vs Observation: Basin-mean P, ET, Q (1985-1989)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    safe_plot_save(fig, "bar_comparison_P_ET_Q.png")
    plt.close(fig)

def make_radar_charts(df_metrics):
    # Metrics: absolute percent biases for P, ET, Q; streamflow bias percent; |WB residual| normalized by P_obs; Wasserstein distance normalized by mean Q_obs
    categories = ["|P bias|%", "|ET bias|%", "|Q bias|%", "|Streamflow bias|%", "|WB residual|%", "WD (rel)"]
    num_vars = len(categories)

    # Compute metrics per basin
    rows = []
    for _, r in df_metrics.iterrows():
        # Relative biases (%)
        p_rel = np.nan if r["P_obs_mmday"] == 0 or np.isnan(r["P_obs_mmday"]) else 100.0 * (r["P_model_mmday"] - r["P_obs_mmday"]) / r["P_obs_mmday"]
        et_rel = np.nan if r["ET_obs_mmday"] == 0 or np.isnan(r["ET_obs_mmday"]) else 100.0 * (r["ET_model_mmday"] - r["ET_obs_mmday"]) / r["ET_obs_mmday"]
        q_rel = np.nan if r["Q_obs_mmday"] == 0 or np.isnan(r["Q_obs_mmday"]) else 100.0 * (r["Q_model_mmday"] - r["Q_obs_mmday"]) / r["Q_obs_mmday"]
        sf_rel = np.nan
        if "streamflow_bias_rel" in r and pd.notnull(r["streamflow_bias_rel"]):
            sf_rel = 100.0 * r["streamflow_bias_rel"]
        # Water balance residual normalized by P_obs
        wb_res = (r["P_model_mmday"] - r["ET_model_mmday"] - r["Q_model_mmday"]) - (r["P_obs_mmday"] - r["ET_obs_mmday"] - r["Q_obs_mmday"])
        wb_rel = np.nan if r["P_obs_mmday"] == 0 or np.isnan(r["P_obs_mmday"]) else 100.0 * np.abs(wb_res) / r["P_obs_mmday"]
        # Wasserstein distance (normalize by mean Q_obs)
        wd_rel = np.nan
        if "wasserstein_distance" in r and pd.notnull(r["wasserstein_distance"]):
            # normalize by mean observed flow
            # Need mean observed flow from time series; not available here, so approximate via Q_obs_mmday converted to m3/s is not trivial
            # Instead, normalize WD by itself max across basins after computing; placeholder here - store raw
            wd_rel = r["wasserstein_distance"]
        rows.append({"basin": r["basin"], "metrics": [np.abs(p_rel), np.abs(et_rel), np.abs(q_rel), np.abs(sf_rel), wb_rel, wd_rel]})
    # Normalize WD across basins to percent-like scale
    wd_vals = [row["metrics"][-1] for row in rows if pd.notnull(row["metrics"][-1])]
    wd_max = np.nanmax(wd_vals) if len(wd_vals) > 0 else np.nan
    for row in rows:
        if pd.isnull(row["metrics"][-1]) or pd.isnull(wd_max) or wd_max == 0:
            row["metrics"][-1] = np.nan
        else:
            row["metrics"][-1] = 100.0 * row["metrics"][-1] / wd_max

    # Determine max for normalization of other metrics for plotting scale
    # We'll cap each value at 200% for visualization
    # Create radar plots per basin (2x3 grid)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, subplot_kw=dict(polar=True), figsize=(14, 8))
    axes = axes.flatten()

    for ax, row in zip(axes, rows):
        values = row["metrics"]
        # Replace nans with 0 for plotting
        vals = [0 if (v is None or pd.isnull(v)) else min(float(v), 200.0) for v in values]
        vals += vals[:1]  # close the circle
        ax.plot(angles, vals, color="#1f77b4", linewidth=2)
        ax.fill(angles, vals, color="#1f77b4", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_yticklabels([])
        ax.set_ylim(0, 200)
        ax.set_title(row["basin"], y=1.1)
    fig.suptitle("Per-basin multi-variable diagnostics (normalized %; WD normalized)", fontsize=14)
    fig.tight_layout()
    safe_plot_save(fig, "radar_diagnostics.png")
    plt.close(fig)

# -------------------------
# Main Workflow
# -------------------------
def main():
    # Part 1: ELM fields
    print("Computing ELM climatology...")
    elm_clim = compute_elm_climatology()

    # Part 2: Observations
    print("Computing observation climatology...")
    obs_clim = compute_obs_climatology()

    # Part 3: Basin means
    print("Computing basin-averaged means...")
    try:
        basin_polys = load_basin_polygons(BASIN_GEOJSON)
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        return

    df_basin_means = compute_basin_means(elm_clim, obs_clim, basin_polys, BASINS)

    # Part 4: Streamflow metrics
    print("Computing streamflow metrics...")
    df_stream = compute_streamflow_metrics(BASINS)

    # Part 5: Combine
    print("Combining metrics and generating visualizations...")
    df = df_basin_means.merge(df_stream, on=["basin", "gauge_id"], how="left")

    # Derived metrics
    df["P_bias_abs_mmday"] = df["P_model_mmday"] - df["P_obs_mmday"]
    df["ET_bias_abs_mmday"] = df["ET_model_mmday"] - df["ET_obs_mmday"]
    df["Q_bias_abs_mmday"] = df["Q_model_mmday"] - df["Q_obs_mmday"]

    df["P_bias_pct"] = (df["P_bias_abs_mmday"] / df["P_obs_mmday"]) * 100.0
    df["ET_bias_pct"] = (df["ET_bias_abs_mmday"] / df["ET_obs_mmday"]) * 100.0
    df["Q_bias_pct"] = (df["Q_bias_abs_mmday"] / df["Q_obs_mmday"]) * 100.0

    df["WB_residual_mmday"] = (df["P_model_mmday"] - df["ET_model_mmday"] - df["Q_model_mmday"]) - \
                              (df["P_obs_mmday"] - df["ET_obs_mmday"] - df["Q_obs_mmday"])

    # Save summary table
    csv_path = os.path.join(OUTDIR_REL, "basin_metrics_1985_1989.csv")
    try:
        df.to_csv(csv_path, index=False, float_format="%.6g")
        save_to_outdirs(csv_path)
    except Exception as e:
        print(f"Warning: Failed to save metrics CSV: {e}")

    # Visualizations
    try:
        make_bar_chart(df)
    except Exception as e:
        print(f"Warning: Failed to create bar chart: {e}")
    try:
        make_radar_charts(df)
    except Exception as e:
        print(f"Warning: Failed to create radar charts: {e}")

    print("Analysis complete.")

if __name__ == "__main__":
    main()