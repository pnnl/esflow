import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_07_integrated_diagnostic/run1_debug/v1/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
CASE_NAME = "sample.v3.LR.historical"
ELM_DIR = "./data/sample/e3sm/lnd"
ROF_DIR = "./data/sample/e3sm/rof"
OBS_STREAMFLOW_DIR = "./data/sample/obs/streamflow"
GAUGE_META_PATH = "./data/sample/obs/gauge_metadata.csv"
BASIN_POLY_PATH = "./data/sample/obs/basin_polygons.geojson"

BASINS = {
    '3629000': 'Amazon',
    '4121801': 'Missouri',
    '4115200': 'Columbia',
    '6742900': 'Danube',
    '2969100': 'Mekong',
    '1159100': 'Orange',
}

YEARS = range(1985, 1990)
MONTHS = range(1, 13)

ILAMB_BASE = "https://www.ilamb.org/ILAMB-Data/DATA"
ILAMB_DATASETS = {
    'pr': {'path': 'pr/GPCCv2018/pr.nc', 'var': 'pr'},
    'et': {'path': 'evspsbl/MODIS/et_0.5x0.5.nc', 'var': 'et'},
    'mrro': {'path': 'mrro/LORA/LORA.nc', 'var': 'mrro'},
}

SEC_PER_DAY = 86400.0


def fix_polygon(polygon):
    """Fix invalid polygon geometry."""
    from shapely.validation import make_valid
    if not polygon.is_valid:
        try:
            polygon = make_valid(polygon)
        except Exception:
            polygon = polygon.buffer(0)
    return polygon


def load_basin_polygons():
    """Load basin polygons from GeoJSON."""
    try:
        from shapely.geometry import shape
        with open(BASIN_POLY_PATH, 'r') as f:
            geojson = json.load(f)

        basins = {}
        for feature in geojson['features']:
            gid = str(feature['properties'].get('grdc_no', feature['properties'].get('gauge_id', '')))
            if gid in BASINS:
                geom = shape(feature['geometry'])
                geom = fix_polygon(geom)
                basins[gid] = geom
                print(f"  Loaded basin polygon for {BASINS[gid]} (gauge {gid}), valid={geom.is_valid}")
        return basins
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        import traceback
        traceback.print_exc()
        return {}


def compute_basin_mean(lons, lats, data_2d, polygon):
    """Compute area-weighted basin mean from a 2D field using point-in-polygon."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    if data_2d is None or polygon is None:
        return np.nan

    # Ensure valid polygon
    polygon = fix_polygon(polygon)
    prepared_poly = prep(polygon)

    # Handle different array shapes
    if len(lons.shape) == 1 and len(lats.shape) == 1:
        lon_2d, lat_2d = np.meshgrid(lons, lats)
    else:
        lon_2d, lat_2d = lons, lats

    bounds = polygon.bounds  # (minx, miny, maxx, maxy)

    lon_flat = lon_2d.flatten()
    lat_flat = lat_2d.flatten()
    data_flat = data_2d.flatten()

    # Determine longitude convention adjustment
    # Try both conventions and pick the one that gives points in bounding box
    shifts_to_try = [0]
    if bounds[0] < 0 and np.nanmin(lon_flat) >= 0:
        shifts_to_try = [-360, 0]
    elif bounds[0] >= 0 and np.nanmin(lon_flat) < 0:
        shifts_to_try = [360, 0]
    elif bounds[2] > 180 and np.nanmax(lon_flat) <= 180:
        shifts_to_try = [360, 0]

    best_count = 0
    best_shift = 0
    for shift in shifts_to_try:
        lon_shifted = lon_flat + shift
        margin = 1.0
        bbox_mask = (
            (lon_shifted >= bounds[0] - margin) &
            (lon_shifted <= bounds[2] + margin) &
            (lat_flat >= bounds[1] - margin) &
            (lat_flat <= bounds[3] + margin) &
            np.isfinite(data_flat)
        )
        count = np.sum(bbox_mask)
        if count > best_count:
            best_count = count
            best_shift = shift

    lon_flat_shifted = lon_flat + best_shift
    margin = 1.0
    bbox_mask = (
        (lon_flat_shifted >= bounds[0] - margin) &
        (lon_flat_shifted <= bounds[2] + margin) &
        (lat_flat >= bounds[1] - margin) &
        (lat_flat <= bounds[3] + margin) &
        np.isfinite(data_flat)
    )

    if not np.any(bbox_mask):
        print(f"    No grid cells found in bounding box. Polygon bounds: {bounds}")
        print(f"    Data lon range: [{np.nanmin(lon_flat_shifted):.1f}, {np.nanmax(lon_flat_shifted):.1f}]")
        print(f"    Data lat range: [{np.nanmin(lat_flat):.1f}, {np.nanmax(lat_flat):.1f}]")
        return np.nan

    indices = np.where(bbox_mask)[0]

    # Point-in-polygon test using prepared geometry
    inside_indices = []
    for idx in indices:
        pt = Point(lon_flat_shifted[idx], lat_flat[idx])
        try:
            if prepared_poly.contains(pt):
                inside_indices.append(idx)
        except Exception:
            pass

    inside_indices = np.array(inside_indices)

    if len(inside_indices) == 0:
        print(f"    No grid cells inside polygon after point-in-polygon test ({len(indices)} candidates)")
        return np.nan

    # Area weighting by cos(lat)
    weights = np.cos(np.deg2rad(lat_flat[inside_indices]))
    values = data_flat[inside_indices]

    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return np.nan

    basin_mean = np.average(values[valid], weights=weights[valid])
    print(f"    Basin mean computed from {np.sum(valid)} grid cells: {basin_mean:.4f}")
    return basin_mean


# ============================================================
# PART 1: Extract ELM model fields
# ============================================================
print("=" * 70)
print("PART 1: Extracting ELM model fields (P, ET, Q)")
print("=" * 70)

elm_files = []
for year in YEARS:
    for month in MONTHS:
        fname = f"{CASE_NAME}.elm.h0.{year:04d}-{month:02d}.nc"
        fpath = os.path.join(ELM_DIR, fname)
        if os.path.exists(fpath):
            elm_files.append(fpath)

print(f"Found {len(elm_files)} ELM monthly files")

model_P_clim = None
model_ET_clim = None
model_Q_clim = None
model_lons = None
model_lats = None

try:
    if len(elm_files) > 0:
        ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
        print(f"ELM dataset dimensions: {dict(ds_elm.dims)}")
        print(f"ELM dataset variables: {list(ds_elm.data_vars)}")

        if 'lon' in ds_elm.coords:
            model_lons = ds_elm['lon'].values
        elif 'longitude' in ds_elm.coords:
            model_lons = ds_elm['longitude'].values

        if 'lat' in ds_elm.coords:
            model_lats = ds_elm['lat'].values
        elif 'latitude' in ds_elm.coords:
            model_lats = ds_elm['latitude'].values

        # Precipitation: RAIN + SNOW (kg/m2/s -> mm/day)
        if 'RAIN' in ds_elm and 'SNOW' in ds_elm:
            model_P = (ds_elm['RAIN'] + ds_elm['SNOW']) * SEC_PER_DAY
            model_P_clim = model_P.mean(dim='time').values
            print(f"Model P computed (RAIN+SNOW), shape: {model_P_clim.shape}, mean: {np.nanmean(model_P_clim):.3f} mm/day")

        # ET: QVEGE + QVEGT + QSOIL (mm/s -> mm/day)
        et_vars = ['QVEGE', 'QVEGT', 'QSOIL']
        if all(v in ds_elm for v in et_vars):
            model_ET = (ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']) * SEC_PER_DAY
            model_ET_clim = model_ET.mean(dim='time').values
            print(f"Model ET computed, shape: {model_ET_clim.shape}, mean: {np.nanmean(model_ET_clim):.3f} mm/day")

        # Runoff: QRUNOFF (mm/s -> mm/day)
        if 'QRUNOFF' in ds_elm:
            model_Q = ds_elm['QRUNOFF'] * SEC_PER_DAY
            model_Q_clim = model_Q.mean(dim='time').values
            print(f"Model Q computed, shape: {model_Q_clim.shape}, mean: {np.nanmean(model_Q_clim):.3f} mm/day")

        ds_elm.close()
    else:
        print("No ELM files found!")
except Exception as e:
    print(f"Error in Part 1: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# PART 2: Fetch and extract observation fields
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Fetching and extracting observation fields")
print("=" * 70)

obs_P_clim = None
obs_ET_clim = None
obs_Q_clim = None
obs_P_lons = None
obs_P_lats = None
obs_ET_lons = None
obs_ET_lats = None
obs_Q_lons = None
obs_Q_lats = None


def download_ilamb_file(rel_path, local_dir="./data/ilamb_cache"):
    """Download ILAMB file if not cached."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(rel_path))

    if os.path.exists(local_path):
        print(f"  Using cached: {local_path}")
        return local_path

    url = f"{ILAMB_BASE}/{rel_path}"
    print(f"  Downloading: {url}")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, local_path)
        print(f"  Saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def get_coord(ds, names):
    """Get coordinate array from dataset, trying multiple names."""
    for cname in names:
        if cname in ds.coords:
            return ds[cname].values
        if cname in ds.dims:
            return ds[cname].values
    return None


# GPCC Precipitation
print("\nFetching GPCC precipitation...")
try:
    pr_path = download_ilamb_file(ILAMB_DATASETS['pr']['path'])
    if pr_path:
        ds_pr = xr.open_dataset(pr_path, decode_times=True)
        print(f"  GPCC dimensions: {dict(ds_pr.dims)}")
        print(f"  GPCC variables: {list(ds_pr.data_vars)}")

        pr_var = ds_pr[ILAMB_DATASETS['pr']['var']]
        print(f"  PR dtype: {pr_var.dtype}, attrs: {dict(pr_var.attrs)}")

        # Print time range
        times = ds_pr['time'].values
        print(f"  Time range: {pd.Timestamp(times[0])} to {pd.Timestamp(times[-1])}")

        # Select time period 1985-1989
        try:
            pr_sel = pr_var.sel(time=slice('1985-01-01', '1989-12-31'))
            if len(pr_sel.time) == 0:
                print(f"  No data in 1985-1989, using all available times")
                pr_sel = pr_var
            else:
                print(f"  Selected {len(pr_sel.time)} timesteps for 1985-1989")
        except Exception:
            pr_sel = pr_var

        # Compute mean with proper masking
        obs_P_clim_raw = pr_sel.mean(dim='time').values

        # Check for unreasonable values and mask them
        # Precipitation should be between 0 and ~50 mm/day
        print(f"  Raw obs P stats: min={np.nanmin(obs_P_clim_raw):.4f}, max={np.nanmax(obs_P_clim_raw):.4f}, mean={np.nanmean(obs_P_clim_raw):.4f}")

        # Check units: GPCC pr might be in kg/m2/s
        units = pr_var.attrs.get('units', '')
        print(f"  PR units: '{units}'")

        if np.nanmean(np.abs(obs_P_clim_raw[np.isfinite(obs_P_clim_raw)])) > 1000:
            # Values are way too large - likely fill values not masked
            # Try re-opening with mask_and_scale
            ds_pr.close()
            ds_pr = xr.open_dataset(pr_path, decode_times=True, mask_and_scale=True)
            pr_var = ds_pr[ILAMB_DATASETS['pr']['var']]
            try:
                pr_sel = pr_var.sel(time=slice('1985-01-01', '1989-12-31'))
                if len(pr_sel.time) == 0:
                    pr_sel = pr_var
            except Exception:
                pr_sel = pr_var
            obs_P_clim_raw = pr_sel.mean(dim='time').values
            print(f"  After re-open with mask_and_scale: min={np.nanmin(obs_P_clim_raw):.6f}, max={np.nanmax(obs_P_clim_raw):.6f}, mean={np.nanmean(obs_P_clim_raw):.6f}")

        # Mask unreasonable values
        obs_P_clim_raw = np.where(np.abs(obs_P_clim_raw) > 1e10, np.nan, obs_P_clim_raw)

        # Convert units if needed
        if 'kg' in str(units) and 's' in str(units):
            # kg/m2/s -> mm/day
            obs_P_clim = obs_P_clim_raw * SEC_PER_DAY
            print(f"  Converted from kg/m2/s to mm/day")
        elif np.nanmean(np.abs(obs_P_clim_raw[np.isfinite(obs_P_clim_raw)])) < 0.01:
            # Likely in kg/m2/s
            obs_P_clim = obs_P_clim_raw * SEC_PER_DAY
            print(f"  Values very small, assuming kg/m2/s -> mm/day conversion")
        elif np.nanmean(np.abs(obs_P_clim_raw[np.isfinite(obs_P_clim_raw)])) > 100:
            # Likely in mm/month or something else
            obs_P_clim = obs_P_clim_raw / 30.0
            print(f"  Values large, assuming mm/month -> mm/day")
        else:
            obs_P_clim = obs_P_clim_raw

        obs_P_lons = get_coord(ds_pr, ['lon', 'longitude', 'x'])
        obs_P_lats = get_coord(ds_pr, ['lat', 'latitude', 'y'])

        print(f"  Obs P shape: {obs_P_clim.shape}, mean: {np.nanmean(obs_P_clim):.3f} mm/day")
        print(f"  Obs P lon range: [{np.nanmin(obs_P_lons):.1f}, {np.nanmax(obs_P_lons):.1f}]")
        ds_pr.close()
except Exception as e:
    print(f"Error fetching GPCC: {e}")
    import traceback
    traceback.print_exc()

# MODIS ET
print("\nFetching MODIS ET...")
try:
    et_path = download_ilamb_file(ILAMB_DATASETS['et']['path'])
    if et_path:
        ds_et = xr.open_dataset(et_path, decode_times=True, mask_and_scale=True)
        print(f"  MODIS ET dimensions: {dict(ds_et.dims)}")
        print(f"  MODIS ET variables: {list(ds_et.data_vars)}")

        et_var = ds_et[ILAMB_DATASETS['et']['var']]

        times = ds_et['time'].values
        print(f"  Time range: {pd.Timestamp(times[0])} to {pd.Timestamp(times[-1])}")

        try:
            et_sel = et_var.sel(time=slice('1985-01-01', '1989-12-31'))
            if len(et_sel.time) == 0:
                print(f"  No data in 1985-1989, using all available times")
                et_sel = et_var
            else:
                print(f"  Selected {len(et_sel.time)} timesteps")
        except Exception:
            et_sel = et_var

        obs_ET_clim_raw = et_sel.mean(dim='time').values
        obs_ET_clim_raw = np.where(np.abs(obs_ET_clim_raw) > 1e10, np.nan, obs_ET_clim_raw)

        units = et_var.attrs.get('units', '')
        print(f"  ET units: '{units}'")

        if 'kg' in str(units) and 's' in str(units):
            obs_ET_clim = obs_ET_clim_raw * SEC_PER_DAY
        elif np.nanmean(np.abs(obs_ET_clim_raw[np.isfinite(obs_ET_clim_raw)])) < 0.01:
            obs_ET_clim = obs_ET_clim_raw * SEC_PER_DAY
        else:
            obs_ET_clim = obs_ET_clim_raw

        obs_ET_lons = get_coord(ds_et, ['lon', 'longitude', 'x'])
        obs_ET_lats = get_coord(ds_et, ['lat', 'latitude', 'y'])

        print(f"  Obs ET shape: {obs_ET_clim.shape}, mean: {np.nanmean(obs_ET_clim):.3f} mm/day")
        ds_et.close()
except Exception as e:
    print(f"Error fetching MODIS ET: {e}")
    import traceback
    traceback.print_exc()

# LORA Runoff
print("\nFetching LORA runoff...")
try:
    mrro_path = download_ilamb_file(ILAMB_DATASETS['mrro']['path'])
    if mrro_path:
        ds_mrro = xr.open_dataset(mrro_path, decode_times=True, mask_and_scale=True)
        print(f"  LORA dimensions: {dict(ds_mrro.dims)}")
        print(f"  LORA variables: {list(ds_mrro.data_vars)}")

        mrro_var = ds_mrro[ILAMB_DATASETS['mrro']['var']]

        times = ds_mrro['time'].values
        print(f"  Time range: {pd.Timestamp(times[0])} to {pd.Timestamp(times[-1])}")

        try:
            mrro_sel = mrro_var.sel(time=slice('1985-01-01', '1989-12-31'))
            if len(mrro_sel.time) == 0:
                print(f"  No data in 1985-1989, using all available times")
                mrro_sel = mrro_var
            else:
                print(f"  Selected {len(mrro_sel.time)} timesteps")
        except Exception:
            mrro_sel = mrro_var

        obs_Q_clim_raw = mrro_sel.mean(dim='time').values
        obs_Q_clim_raw = np.where(np.abs(obs_Q_clim_raw) > 1e10, np.nan, obs_Q_clim_raw)

        units = mrro_var.attrs.get('units', '')
        print(f"  Runoff units: '{units}'")

        if 'kg' in str(units) and 's' in str(units):
            obs_Q_clim = obs_Q_clim_raw * SEC_PER_DAY
        elif np.nanmean(np.abs(obs_Q_clim_raw[np.isfinite(obs_Q_clim_raw)])) < 0.01:
            obs_Q_clim = obs_Q_clim_raw * SEC_PER_DAY
        else:
            obs_Q_clim = obs_Q_clim_raw

        obs_Q_lons = get_coord(ds_mrro, ['lon', 'longitude', 'x'])
        obs_Q_lats = get_coord(ds_mrro, ['lat', 'latitude', 'y'])

        print(f"  Obs Q shape: {obs_Q_clim.shape}, mean: {np.nanmean(obs_Q_clim):.3f} mm/day")
        ds_mrro.close()
except Exception as e:
    print(f"Error fetching LORA: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# PART 3: Clip to basin means
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Computing basin-averaged values")
print("=" * 70)

basin_polygons = load_basin_polygons()

basin_results = {}

for gauge_id, basin_name in BASINS.items():
    print(f"\n--- {basin_name} (gauge {gauge_id}) ---")

    result = {
        'basin_name': basin_name,
        'gauge_id': gauge_id,
        'model_P': np.nan,
        'model_ET': np.nan,
        'model_Q': np.nan,
        'obs_P': np.nan,
        'obs_ET': np.nan,
        'obs_Q': np.nan,
    }

    polygon = basin_polygons.get(gauge_id, None)
    if polygon is None:
        print(f"  No polygon found for {basin_name}")
        basin_results[gauge_id] = result
        continue

    # Model P
    if model_P_clim is not None and model_lons is not None:
        print(f"  Computing model P basin mean...")
        try:
            result['model_P'] = compute_basin_mean(model_lons, model_lats, model_P_clim, polygon)
        except Exception as e:
            print(f"    Error: {e}")

    # Model ET
    if model_ET_clim is not None and model_lons is not None:
        print(f"  Computing model ET basin mean...")
        try:
            result['model_ET'] = compute_basin_mean(model_lons, model_lats, model_ET_clim, polygon)
        except Exception as e:
            print(f"    Error: {e}")

    # Model Q (runoff)
    if model_Q_clim is not None and model_lons is not None:
        print(f"  Computing model Q basin mean...")
        try:
            result['model_Q'] = compute_basin_mean(model_lons, model_lats, model_Q_clim, polygon)
        except Exception as e:
            print(f"    Error: {e}")

    # Obs P
    if obs_P_clim is not None and obs_P_lons is not None:
        print(f"  Computing obs P basin mean...")
        try:
            result['obs_P'] = compute_basin_mean(obs_P_lons, obs_P_lats, obs_P_clim, polygon)
        except Exception as e:
            print(f"    Error: {e}")

    # Obs ET
    if obs_ET_clim is not None and obs_ET_lons is not None:
        print(f"  Computing obs ET basin mean...")
        try:
            result['obs_ET'] = compute_basin_mean(obs_ET_lons, obs_ET_lats, obs_ET_clim, polygon)
        except Exception as e:
            print(f"    Error: {e}")

    # Obs Q (runoff)
    if obs_Q_clim is not None and obs_Q_lons is not None:
        print(f"  Computing obs Q (runoff) basin mean...")
        try:
            result['obs_Q'] = compute_basin_mean(obs_Q_lons, obs_Q_lats, obs_Q_clim, polygon)
        except Exception as e:
            print(f"    Error: {e}")

    basin_results[gauge_id] = result
    print(f"  Model: P={result['model_P']:.3f}, ET={result['model_ET']:.3f}, Q={result['model_Q']:.3f} mm/day")
    print(f"  Obs:   P={result['obs_P']:.3f}, ET={result['obs_ET']:.3f}, Q={result['obs_Q']:.3f} mm/day")


# ============================================================
# PART 4: Streamflow FDC metrics
# ============================================================
print("\n" + "=" * 70)
print("PART 4: Streamflow flow duration curve metrics")
print("=" * 70)

# Load gauge metadata
try:
    gauge_meta = pd.read_csv(GAUGE_META_PATH)
    print(f"Loaded gauge metadata: {len(gauge_meta)} gauges")
    print(f"Columns: {list(gauge_meta.columns)}")
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    gauge_meta = None

# Load MOSART monthly discharge
mosart_files = []
for year in YEARS:
    for month in MONTHS:
        fname = f"{CASE_NAME}.mosart.h0.{year:04d}-{month:02d}.nc"
        fpath = os.path.join(ROF_DIR, fname)
        if os.path.exists(fpath):
            mosart_files.append(fpath)

# Also check for daily files
mosart_daily_files = []
if os.path.exists(ROF_DIR):
    all_rof_files = sorted(os.listdir(ROF_DIR))
    for f in all_rof_files:
        if '.mosart.h1.' in f and f.endswith('.nc'):
            # Check year
            for y in YEARS:
                if f".h1.{y}" in f:
                    mosart_daily_files.append(os.path.join(ROF_DIR, f))
                    break

print(f"Found {len(mosart_files)} MOSART monthly files")
print(f"Found {len(mosart_daily_files)} MOSART daily files in 1985-1989")

streamflow_results = {}

for gauge_id, basin_name in BASINS.items():
    print(f"\n--- {basin_name} (gauge {gauge_id}) ---")

    sf_result = {
        'volume_bias': np.nan,
        'wasserstein_dist': np.nan,
        'obs_mean_discharge': np.nan,
        'model_mean_discharge': np.nan,
    }

    # Load observed streamflow
    obs_sf_path = os.path.join(OBS_STREAMFLOW_DIR, f"{gauge_id}.csv")
    obs_discharge = None
    try:
        if os.path.exists(obs_sf_path):
            obs_sf = pd.read_csv(obs_sf_path, parse_dates=['date'])
            obs_sf = obs_sf[(obs_sf['date'].dt.year >= 1985) & (obs_sf['date'].dt.year <= 1989)]
            obs_discharge = obs_sf['discharge_m3s'].dropna().values
            obs_discharge = obs_discharge[np.isfinite(obs_discharge)]
            print(f"  Obs streamflow: {len(obs_discharge)} daily values")
            if len(obs_discharge) > 0:
                sf_result['obs_mean_discharge'] = np.nanmean(obs_discharge)
        else:
            print(f"  Obs streamflow file not found: {obs_sf_path}")
    except Exception as e:
        print(f"  Error loading obs streamflow: {e}")

    # Get gauge location
    gauge_lat = None
    gauge_lon = None
    if gauge_meta is not None:
        gm = gauge_meta[gauge_meta['gauge_id'].astype(str) == str(gauge_id)]
        if len(gm) > 0:
            gauge_lat = float(gm.iloc[0]['lat'])
            gauge_lon = float(gm.iloc[0]['lon'])
            print(f"  Gauge location: lat={gauge_lat:.2f}, lon={gauge_lon:.2f}")

    # Extract MOSART discharge at gauge location
    model_discharge = None

    # Try monthly files first
    if gauge_lat is not None and gauge_lon is not None and len(mosart_files) > 0:
        try:
            ds_rof = xr.open_mfdataset(mosart_files, combine='by_coords', decode_times=True)

            rof_var = 'RIVER_DISCHARGE_OVER_LAND_LIQ'
            if rof_var not in ds_rof:
                print(f"  {rof_var} not found. Available: {list(ds_rof.data_vars)[:10]}")
            else:
                rof_lats = ds_rof['lat'].values
                rof_lons = ds_rof['lon'].values

                gauge_lon_adj = gauge_lon
                if np.nanmin(rof_lons) >= 0 and gauge_lon < 0:
                    gauge_lon_adj = gauge_lon + 360
                elif np.nanmax(rof_lons) <= 180 and gauge_lon > 180:
                    gauge_lon_adj = gauge_lon - 360

                if len(rof_lats.shape) == 1 and len(rof_lons.shape) == 1:
                    lat_idx = int(np.argmin(np.abs(rof_lats - gauge_lat)))
                    lon_idx = int(np.argmin(np.abs(rof_lons - gauge_lon_adj)))

                    print(f"  Nearest MOSART grid: lat={rof_lats[lat_idx]:.2f}, lon={rof_lons[lon_idx]:.2f}")

                    discharge_ts = ds_rof[rof_var][:, lat_idx, lon_idx].values
                else:
                    if len(rof_lats.shape) == 2:
                        dist = (rof_lats - gauge_lat)**2 + (rof_lons - gauge_lon_adj)**2
                        idx = np.unravel_index(np.argmin(dist), dist.shape)
                        discharge_ts = ds_rof[rof_var][:, idx[0], idx[1]].values
                    else:
                        dist = (rof_lats - gauge_lat)**2 + (rof_lons - gauge_lon_adj)**2
                        idx = int(np.argmin(dist))
                        discharge_ts = ds_rof[rof_var][:, idx].values

                model_discharge = discharge_ts[np.isfinite(discharge_ts)]
                if len(model_discharge) > 0:
                    sf_result['model_mean_discharge'] = float(np.nanmean(model_discharge))
                    print(f"  Model discharge (monthly): {len(model_discharge)} values, mean={sf_result['model_mean_discharge']:.2f} m3/s")

            ds_rof.close()
        except Exception as e:
            print(f"  Error extracting MOSART discharge (monthly): {e}")
            import traceback
            traceback.print_exc()

    # Try daily files if monthly didn't work
    if (model_discharge is None or len(model_discharge) == 0) and gauge_lat is not None and gauge_lon is not None and len(mosart_daily_files) > 0:
        try:
            ds_rof_d = xr.open_mfdataset(mosart_daily_files[:100], combine='by_coords', decode_times=True)
            rof_var = 'RIVER_DISCHARGE_OVER_LAND_LIQ'
            if rof_var in ds_rof_d:
                rof_lats = ds_rof_d['lat'].values
                rof_lons = ds_rof_d['lon'].values

                gauge_lon_adj = gauge_lon
                if np.nanmin(rof_lons) >= 0 and gauge_lon < 0:
                    gauge_lon_adj = gauge_lon + 360

                if len(rof_lats.shape) == 1:
                    lat_idx = int(np.argmin(np.abs(rof_lats - gauge_lat)))
                    lon_idx = int(np.argmin(np.abs(rof_lons - gauge_lon_adj)))
                    discharge_ts = ds_rof_d[rof_var][:, lat_idx, lon_idx].values
                else:
                    dist = (rof_lats - gauge_lat)**2 + (rof_lons - gauge_lon_adj)**2
                    if len(rof_lats.shape) == 2:
                        idx = np.unravel_index(np.argmin(dist), dist.shape)
                        discharge_ts = ds_rof_d[rof_var][:, idx[0], idx[1]].values
                    else:
                        idx = int(np.argmin(dist))
                        discharge_ts = ds_rof_d[rof_var][:, idx].values

                model_discharge = discharge_ts[np.isfinite(discharge_ts)]
                if len(model_discharge) > 0:
                    sf_result['model_mean_discharge'] = float(np.nanmean(model_discharge))
                    print(f"  Model discharge (daily): {len(model_discharge)} values, mean={sf_result['model_mean_discharge']:.2f} m3/s")
            ds_rof_d.close()
        except Exception as e:
            print(f"  Error with daily MOSART: {e}")

    # Compute FDC metrics
    if obs_discharge is not None and len(obs_discharge) > 0 and model_discharge is not None and len(model_discharge) > 0:
        obs_mean = float(np.mean(obs_discharge))
        mod_mean = float(np.mean(model_discharge))
        if obs_mean != 0:
            sf_result['volume_bias'] = (mod_mean - obs_mean) / obs_mean * 100  # percent

        # Wasserstein distance on FDCs
        n_quantiles = 100
        obs_sorted = np.sort(obs_discharge)[::-1]
        mod_sorted = np.sort(model_discharge)[::-1]

        obs_fdc = np.interp(
            np.linspace(0, 1, n_quantiles),
            np.linspace(0, 1, len(obs_sorted)),
            obs_sorted
        )
        mod_fdc = np.interp(
            np.linspace(0, 1, n_quantiles),
            np.linspace(0, 1, len(mod_sorted)),
            mod_sorted
        )

        if obs_mean > 0:
            sf_result['wasserstein_dist'] = float(np.mean(np.abs(obs_fdc - mod_fdc)) / obs_mean)
        else:
            sf_result['wasserstein_dist'] = float(np.mean(np.abs(obs_fdc - mod_fdc)))

        print(f"  Volume bias: {sf_result['volume_bias']:.1f}%")
        print(f"  Wasserstein distance (normalized): {sf_result['wasserstein_dist']:.4f}")
    else:
        print(f"  Cannot compute FDC metrics (obs: {obs_discharge is not None and len(obs_discharge) > 0 if obs_discharge is not None else False}, model: {model_discharge is not None and len(model_discharge) > 0 if model_discharge is not None else False})")

    streamflow_results[gauge_id] = sf_result


# ============================================================
# PART 5: Combine and visualize
# ============================================================
print("\n" + "=" * 70)
print("PART 5: Combining metrics and creating visualizations")
print("=" * 70)

# Build summary table
summary_rows = []
for gauge_id, basin_name in BASINS.items():
    br = basin_results.get(gauge_id, {})
    sr = streamflow_results.get(gauge_id, {})

    model_P = br.get('model_P', np.nan)
    obs_P = br.get('obs_P', np.nan)
    model_ET = br.get('model_ET', np.nan)
    obs_ET = br.get('obs_ET', np.nan)
    model_Q = br.get('model_Q', np.nan)
    obs_Q = br.get('obs_Q', np.nan)

    P_bias = model_P - obs_P if np.isfinite(model_P) and np.isfinite(obs_P) else np.nan
    ET_bias = model_ET - obs_ET if np.isfinite(model_ET) and np.isfinite(obs_ET) else np.nan
    Q_bias = model_Q - obs_Q if np.isfinite(model_Q) and np.isfinite(obs_Q) else np.nan

    P_rel_bias = (P_bias / obs_P * 100) if np.isfinite(obs_P) and obs_P != 0 and np.isfinite(P_bias) else np.nan
    ET_rel_bias = (ET_bias / obs_ET * 100) if np.isfinite(obs_ET) and obs_ET != 0 and np.isfinite(ET_bias) else np.nan
    Q_rel_bias = (Q_bias / obs_Q * 100) if np.isfinite(obs_Q) and obs_Q != 0 and np.isfinite(Q_bias) else np.nan

    model_wb_residual = model_P - model_ET - model_Q if all(np.isfinite([model_P, model_ET, model_Q])) else np.nan
    obs_wb_residual = obs_P - obs_ET - obs_Q if all(np.isfinite([obs_P, obs_ET, obs_Q])) else np.nan

    summary_rows.append({
        'gauge_id': gauge_id,
        'basin_name': basin_name,
        'model_P_mm_day': model_P,
        'obs_P_mm_day': obs_P,
        'P_bias_mm_day': P_bias,
        'P_rel_bias_pct': P_rel_bias,
        'model_ET_mm_day': model_ET,
        'obs_ET_mm_day': obs_ET,
        'ET_bias_mm_day': ET_bias,
        'ET_rel_bias_pct': ET_rel_bias,
        'model_Q_mm_day': model_Q,
        'obs_Q_mm_day': obs_Q,
        'Q_bias_mm_day': Q_bias,
        'Q_rel_bias_pct': Q_rel_bias,
        'streamflow_volume_bias_pct': sr.get('volume_bias', np.nan),
        'wasserstein_distance': sr.get('wasserstein_dist', np.nan),
        'model_wb_residual_mm_day': model_wb_residual,
        'obs_wb_residual_mm_day': obs_wb_residual,
        'obs_mean_discharge_m3s': sr.get('obs_mean_discharge', np.nan),
        'model_mean_discharge_m3s': sr.get('model_mean_discharge', np.nan),
    })

summary_df = pd.DataFrame(summary_rows)

summary_csv_path = os.path.join(OUTPUT_DIR, "basin_diagnostic_summary.csv")
try:
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    print(f"\nSummary table saved to: {summary_csv_path}")
except Exception as e:
    print(f"Error saving summary CSV: {e}")

print("\n" + "-" * 70)
print("SUMMARY TABLE")
print("-" * 70)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(summary_df.to_string(index=False))

# ---- Bar Chart: Model vs Observation P, ET, Q ----
print("\nCreating bar chart...")
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    basin_names = [BASINS[gid] for gid in BASINS]
    x = np.arange(len(basin_names))
    width = 0.35

    variables = [
        ('P', 'model_P_mm_day', 'obs_P_mm_day', 'Precipitation (mm/day)'),
        ('ET', 'model_ET_mm_day', 'obs_ET_mm_day', 'Evapotranspiration (mm/day)'),
        ('Q', 'model_Q_mm_day', 'obs_Q_mm_day', 'Runoff (mm/day)'),
    ]

    colors_model = ['#2196F3', '#4CAF50', '#FF9800']
    colors_obs = ['#1565C0', '#2E7D32', '#E65100']

    for ax_idx, (var_name, mod_col, obs_col, ylabel) in enumerate(variables):
        ax = axes[ax_idx]

        mod_vals = summary_df[mod_col].values.copy()
        obs_vals = summary_df[obs_col].values.copy()

        # Replace NaN with 0 for plotting
        mod_vals_plot = np.where(np.isfinite(mod_vals), mod_vals, 0)
        obs_vals_plot = np.where(np.isfinite(obs_vals), obs_vals, 0)

        ax.bar(x - width / 2, mod_vals_plot, width, label='E3SM', color=colors_model[ax_idx], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, obs_vals_plot, width, label='Obs', color=colors_obs[ax_idx], alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{var_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basin_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        for i in range(len(basin_names)):
            if np.isfinite(mod_vals[i]) and np.isfinite(obs_vals[i]) and obs_vals[i] != 0:
                bias_pct = (mod_vals[i] - obs_vals[i]) / obs_vals[i] * 100
                max_val = max(mod_vals_plot[i], obs_vals_plot[i])
                ax.annotate(f'{bias_pct:+.0f}%',
                            xy=(i, max_val),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', fontsize=8, color='red')

    fig.suptitle('E3SM vs Observations: Basin-Averaged Water Cycle Components (1985-1989)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    bar_path = os.path.join(OUTPUT_DIR, "basin_water_cycle_bar_chart.png")
    fig.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Bar chart saved to: {bar_path}")
except Exception as e:
    print(f"Error creating bar chart: {e}")
    import traceback
    traceback.print_exc()

# ---- Radar Chart: Multi-variable diagnostic per basin ----
print("\nCreating radar chart...")
try:
    categories = ['P bias (%)', 'ET bias (%)', 'Runoff bias (%)',
                   'Streamflow\nbias (%)', 'Wasserstein\ndist (x100)', 'WB residual\n(mm/day x10)']
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(BASINS)))

    max_val_radar = 0
    for idx, (gauge_id, basin_name) in enumerate(BASINS.items()):
        row = summary_df[summary_df['gauge_id'] == gauge_id]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        values = [
            abs(row['P_rel_bias_pct']) if np.isfinite(row['P_rel_bias_pct']) else 0,
            abs(row['ET_rel_bias_pct']) if np.isfinite(row['ET_rel_bias_pct']) else 0,
            abs(row['Q_rel_bias_pct']) if np.isfinite(row['Q_rel_bias_pct']) else 0,
            abs(row['streamflow_volume_bias_pct']) if np.isfinite(row['streamflow_volume_bias_pct']) else 0,
            abs(row['wasserstein_distance']) * 100 if np.isfinite(row['wasserstein_distance']) else 0,
            abs(row['model_wb_residual_mm_day']) * 10 if np.isfinite(row['model_wb_residual_mm_day']) else 0,
        ]

        max_val_radar = max(max_val_radar, max(values))
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=basin_name, color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.08, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Multi-Variable Water Cycle Diagnostic\n(Absolute Bias Magnitudes)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)

    radar_path = os.path.join(OUTPUT_DIR, "basin_diagnostic_radar_chart.png")
    fig.savefig(radar_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Radar chart saved to: {radar_path}")
except Exception as e:
    print(f"Error creating radar chart: {e}")
    import traceback
    traceback.print_exc()

# ---- Additional: Bias attribution chart ----
print("\nCreating bias attribution chart...")
try:
    fig, ax = plt.subplots(figsize=(14, 7))

    basin_names_list = list(BASINS.values())
    x = np.arange(len(basin_names_list))
    width = 0.25

    p_biases = summary_df['P_bias_mm_day'].values
    et_biases = summary_df['ET_bias_mm_day'].values
    q_biases = summary_df['Q_bias_mm_day'].values

    p_biases_plot = np.where(np.isfinite(p_biases), p_biases, 0)
    et_biases_plot = np.where(np.isfinite(et_biases), et_biases, 0)
    q_biases_plot = np.where(np.isfinite(q_biases), q_biases, 0)

    ax.bar(x - width, p_biases_plot, width, label='P bias', color='#2196F3', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x, et_biases_plot, width, label='ET bias', color='#4CAF50', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + width, q_biases_plot, width, label='Q bias', color='#FF9800', alpha=0.85, edgecolor='black', linewidth=0.5)

    # Expected Q bias = P bias - ET bias
    expected_q_bias = p_biases_plot - et_biases_plot
    ax.scatter(x + width, expected_q_bias, color='red', s=80, zorder=5, label='Expected Q bias (P-ET)', marker='D', edgecolor='black')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(basin_names_list, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Bias (mm/day)', fontsize=12)
    ax.set_title('Water Cycle Bias Attribution by Basin (1985-1989)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    bias_path = os.path.join(OUTPUT_DIR, "basin_bias_attribution.png")
    fig.savefig(bias_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Bias attribution chart saved to: {bias_path}")
except Exception as e:
    print(f"Error creating bias attribution chart: {e}")
    import traceback
    traceback.print_exc()

# ---- Save text report ----
print("\nSaving detailed report...")
try:
    txt_path = os.path.join(OUTPUT_DIR, "basin_diagnostic_report.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("E3SM Per-Basin Integrated Water Cycle Evaluation (1985-1989)\n")
        f.write("=" * 80 + "\n\n")

        for _, row in summary_df.iterrows():
            f.write(f"Basin: {row['basin_name']} (gauge {row['gauge_id']})\n")
            f.write("-" * 50 + "\n")

            def fmt(v, prec=3):
                return f"{v:.{prec}f}" if np.isfinite(v) else "N/A"

            f.write(f"  Precipitation (mm/day):   Model={fmt(row['model_P_mm_day'])}  Obs={fmt(row['obs_P_mm_day'])}  Bias={fmt(row['P_bias_mm_day'])} ({fmt(row['P_rel_bias_pct'], 1)}%)\n")
            f.write(f"  Evapotranspiration:       Model={fmt(row['model_ET_mm_day'])}  Obs={fmt(row['obs_ET_mm_day'])}  Bias={fmt(row['ET_bias_mm_day'])} ({fmt(row['ET_rel_bias_pct'], 1)}%)\n")
            f.write(f"  Runoff (mm/day):          Model={fmt(row['model_Q_mm_day'])}  Obs={fmt(row['obs_Q_mm_day'])}  Bias={fmt(row['Q_bias_mm_day'])} ({fmt(row['Q_rel_bias_pct'], 1)}%)\n")
            f.write(f"  Streamflow vol bias:      {fmt(row['streamflow_volume_bias_pct'], 1)}%\n")
            f.write(f"  Wasserstein distance:     {fmt(row['wasserstein_distance'], 4)}\n")
            f.write(f"  Model WB residual (P-ET-Q): {fmt(row['model_wb_residual_mm_day'])} mm/day\n")
            f.write(f"  Obs WB residual (P-ET-Q):   {fmt(row['obs_wb_residual_mm_day'])} mm/day\n")
            f.write(f"  Obs mean discharge:       {fmt(row['obs_mean_discharge_m3s'], 1)} m3/s\n")
            f.write(f"  Model mean discharge:     {fmt(row['model_mean_discharge_m3s'], 1)} m3/s\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("- Positive P bias = model is too wet\n")
        f.write("- Positive ET bias = model evaporates too much\n")
        f.write("- Positive Q bias = model generates too much runoff\n")
        f.write("- WB residual should be close to 0 (P ~ ET + Q at long time scales)\n")
        f.write("- Large WB residual indicates storage change or model inconsistency\n")
        f.write("=" * 80 + "\n")

    print(f"Report saved to: {txt_path}")
except Exception as e:
    print(f"Error saving report: {e}")

print("\n" + "=" * 70)
print("TASK COMPLETE")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("=" * 70)

for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f} ({size_kb:.1f} KB)")