import os
import sys
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# Output directory
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_07_integrated_diagnostic/run4_output"
os.makedirs(output_dir, exist_ok=True)

# Basin definitions
basins = {
    '3629000': 'Amazon',
    '4121801': 'Missouri',
    '4115200': 'Columbia',
    '6742900': 'Danube',
    '2969100': 'Mekong',
    '1159100': 'Orange'
}

basin_ids = list(basins.keys())
basin_names = list(basins.values())

# Data paths
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"
rof_dir = "./data/sample/e3sm/rof"
gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
streamflow_obs_dir = "./data/sample/obs/streamflow"
basin_poly_path = "./data/sample/obs/basin_polygons.geojson"

years = range(1985, 1990)
months = range(1, 13)

# ============================================================
# PART 1: Extract model fields from ELM monthly output
# ============================================================
print("=" * 60)
print("PART 1: Extracting ELM model fields (1985-1989 climatology)")
print("=" * 60)

elm_files = []
for yr in years:
    for mo in months:
        fname = os.path.join(elm_dir, f"{case_name}.elm.h0.{yr:04d}-{mo:02d}.nc")
        if os.path.exists(fname):
            elm_files.append(fname)

print(f"Found {len(elm_files)} ELM files")

model_P = None
model_ET = None
model_Q = None
elm_lat = None
elm_lon = None

try:
    if len(elm_files) > 0:
        # Open all files
        ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
        
        # Precipitation: RAIN + SNOW (kg/m2/s -> mm/day)
        rain = ds_elm['RAIN'] if 'RAIN' in ds_elm else xr.zeros_like(ds_elm[list(ds_elm.data_vars)[0]])
        snow = ds_elm['SNOW'] if 'SNOW' in ds_elm else xr.zeros_like(rain)
        precip = rain + snow  # kg/m2/s = mm/s
        precip_mmday = precip * 86400.0  # mm/day
        
        # ET: QVEGE + QVEGT + QSOIL (mm/s -> mm/day)
        qvege = ds_elm['QVEGE'] if 'QVEGE' in ds_elm else xr.zeros_like(rain)
        qvegt = ds_elm['QVEGT'] if 'QVEGT' in ds_elm else xr.zeros_like(rain)
        qsoil = ds_elm['QSOIL'] if 'QSOIL' in ds_elm else xr.zeros_like(rain)
        et = qvege + qvegt + qsoil  # mm/s
        et_mmday = et * 86400.0  # mm/day
        
        # Total runoff: QRUNOFF (mm/s -> mm/day)
        qrunoff = ds_elm['QRUNOFF'] if 'QRUNOFF' in ds_elm else xr.zeros_like(rain)
        qrunoff_mmday = qrunoff * 86400.0  # mm/day
        
        # Compute time-mean climatology
        model_P = precip_mmday.mean(dim='time')
        model_ET = et_mmday.mean(dim='time')
        model_Q = qrunoff_mmday.mean(dim='time')
        
        # Get lat/lon
        if 'lat' in ds_elm.coords:
            elm_lat = ds_elm['lat'].values
            elm_lon = ds_elm['lon'].values
        elif 'lat' in ds_elm:
            elm_lat = ds_elm['lat'].values
            elm_lon = ds_elm['lon'].values
        
        print(f"Model P shape: {model_P.shape}, mean: {float(model_P.mean()):.3f} mm/day")
        print(f"Model ET shape: {model_ET.shape}, mean: {float(model_ET.mean()):.3f} mm/day")
        print(f"Model Q shape: {model_Q.shape}, mean: {float(model_Q.mean()):.3f} mm/day")
        
        ds_elm.close()
    else:
        print("WARNING: No ELM files found.")
except Exception as e:
    print(f"ERROR in Part 1: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# PART 2: Fetch and extract observation fields from ILAMB
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Fetching and extracting observation fields from ILAMB")
print("=" * 60)

ilamb_base = "https://www.ilamb.org/ILAMB-Data/DATA"
ilamb_cache = "./data/ilamb_cache"
os.makedirs(ilamb_cache, exist_ok=True)

def download_ilamb(url, local_path):
    """Download ILAMB data if not cached."""
    if os.path.exists(local_path):
        print(f"  Using cached: {local_path}")
        return True
    try:
        import urllib.request
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, local_path)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

# GPCC Precipitation
obs_P = None
obs_P_lat = None
obs_P_lon = None
try:
    gpcc_url = f"{ilamb_base}/pr/GPCCv2018/pr.nc"
    gpcc_local = os.path.join(ilamb_cache, "gpcc_pr.nc")
    if download_ilamb(gpcc_url, gpcc_local):
        ds_gpcc = xr.open_dataset(gpcc_local, decode_times=True)
        print(f"  GPCC variables: {list(ds_gpcc.data_vars)}")
        print(f"  GPCC dims: {dict(ds_gpcc.dims)}")
        
        pr_var = 'pr'
        if pr_var not in ds_gpcc:
            pr_var = list(ds_gpcc.data_vars)[0]
        
        # Select 1985-1989 if time dimension exists
        pr_data = ds_gpcc[pr_var]
        if 'time' in pr_data.dims:
            pr_data = pr_data.sel(time=slice('1985-01-01', '1989-12-31'))
            obs_P = pr_data.mean(dim='time')
        else:
            obs_P = pr_data
        
        # Get lat/lon
        lat_name = 'lat' if 'lat' in ds_gpcc.coords else list(ds_gpcc.coords)[0]
        lon_name = 'lon' if 'lon' in ds_gpcc.coords else list(ds_gpcc.coords)[1]
        obs_P_lat = ds_gpcc[lat_name].values
        obs_P_lon = ds_gpcc[lon_name].values
        
        # Handle 0-360 lon convention
        if obs_P_lon.max() > 180:
            obs_P_lon_shifted = np.where(obs_P_lon > 180, obs_P_lon - 360, obs_P_lon)
            sort_idx = np.argsort(obs_P_lon_shifted)
            obs_P_lon = obs_P_lon_shifted[sort_idx]
            obs_P = obs_P.isel(**{lon_name: sort_idx})
        
        print(f"  Obs P shape: {obs_P.shape}, mean: {float(obs_P.mean()):.3f} mm/day")
        ds_gpcc.close()
except Exception as e:
    print(f"ERROR fetching GPCC: {e}")
    import traceback
    traceback.print_exc()

# MODIS ET
obs_ET = None
obs_ET_lat = None
obs_ET_lon = None
try:
    modis_url = f"{ilamb_base}/evspsbl/MODIS/et_0.5x0.5.nc"
    modis_local = os.path.join(ilamb_cache, "modis_et.nc")
    if download_ilamb(modis_url, modis_local):
        ds_modis = xr.open_dataset(modis_local, decode_times=True)
        print(f"  MODIS variables: {list(ds_modis.data_vars)}")
        print(f"  MODIS dims: {dict(ds_modis.dims)}")
        
        et_var = 'et'
        if et_var not in ds_modis:
            for v in ds_modis.data_vars:
                if 'et' in v.lower() or 'evspsbl' in v.lower():
                    et_var = v
                    break
            else:
                et_var = list(ds_modis.data_vars)[0]
        
        et_data = ds_modis[et_var]
        if 'time' in et_data.dims:
            et_data = et_data.sel(time=slice('1985-01-01', '1989-12-31'))
            if len(et_data.time) == 0:
                print("  WARNING: No MODIS data in 1985-1989, using all available times")
                et_data = ds_modis[et_var]
            obs_ET = et_data.mean(dim='time')
        else:
            obs_ET = et_data
        
        lat_name = 'lat' if 'lat' in ds_modis.coords else list(ds_modis.coords)[0]
        lon_name = 'lon' if 'lon' in ds_modis.coords else list(ds_modis.coords)[1]
        obs_ET_lat = ds_modis[lat_name].values
        obs_ET_lon = ds_modis[lon_name].values
        
        # Convert kg/m2/s to mm/day if needed
        et_mean = float(obs_ET.mean())
        if abs(et_mean) < 0.01:
            obs_ET = obs_ET * 86400.0
            print(f"  Converted ET from kg/m2/s to mm/day")
        
        # Handle 0-360 lon convention
        if obs_ET_lon.max() > 180:
            obs_ET_lon_shifted = np.where(obs_ET_lon > 180, obs_ET_lon - 360, obs_ET_lon)
            sort_idx = np.argsort(obs_ET_lon_shifted)
            obs_ET_lon = obs_ET_lon_shifted[sort_idx]
            obs_ET = obs_ET.isel(**{lon_name: sort_idx})
        
        print(f"  Obs ET shape: {obs_ET.shape}, mean: {float(obs_ET.mean()):.3f} mm/day")
        ds_modis.close()
except Exception as e:
    print(f"ERROR fetching MODIS ET: {e}")
    import traceback
    traceback.print_exc()

# LORA Runoff
obs_Qr = None
obs_Qr_lat = None
obs_Qr_lon = None
try:
    lora_url = f"{ilamb_base}/mrro/LORA/LORA.nc"
    lora_local = os.path.join(ilamb_cache, "lora_mrro.nc")
    if download_ilamb(lora_url, lora_local):
        ds_lora = xr.open_dataset(lora_local, decode_times=True)
        print(f"  LORA variables: {list(ds_lora.data_vars)}")
        print(f"  LORA dims: {dict(ds_lora.dims)}")
        
        mrro_var = 'mrro'
        if mrro_var not in ds_lora:
            mrro_var = list(ds_lora.data_vars)[0]
        
        mrro_data = ds_lora[mrro_var]
        if 'time' in mrro_data.dims:
            mrro_data = mrro_data.sel(time=slice('1985-01-01', '1989-12-31'))
            if len(mrro_data.time) == 0:
                print("  WARNING: No LORA data in 1985-1989, using all available times")
                mrro_data = ds_lora[mrro_var]
            obs_Qr = mrro_data.mean(dim='time')
        else:
            obs_Qr = mrro_data
        
        lat_name = 'lat' if 'lat' in ds_lora.coords else list(ds_lora.coords)[0]
        lon_name = 'lon' if 'lon' in ds_lora.coords else list(ds_lora.coords)[1]
        obs_Qr_lat = ds_lora[lat_name].values
        obs_Qr_lon = ds_lora[lon_name].values
        
        # Convert kg/m2/s to mm/day if needed
        qr_mean = float(obs_Qr.mean())
        if abs(qr_mean) < 0.1:
            obs_Qr = obs_Qr * 86400.0
            print(f"  Converted runoff from kg/m2/s to mm/day")
        
        # Handle 0-360 lon convention
        if obs_Qr_lon.max() > 180:
            obs_Qr_lon_shifted = np.where(obs_Qr_lon > 180, obs_Qr_lon - 360, obs_Qr_lon)
            sort_idx = np.argsort(obs_Qr_lon_shifted)
            obs_Qr_lon = obs_Qr_lon_shifted[sort_idx]
            obs_Qr = obs_Qr.isel(**{lon_name: sort_idx})
        
        print(f"  Obs runoff shape: {obs_Qr.shape}, mean: {float(obs_Qr.mean()):.3f} mm/day")
        ds_lora.close()
except Exception as e:
    print(f"ERROR fetching LORA runoff: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# PART 3: Clip to basin means
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Computing basin-averaged values")
print("=" * 60)

# Load basin polygons
try:
    with open(basin_poly_path, 'r') as f:
        basin_geojson = json.load(f)
    print(f"Loaded basin polygons with {len(basin_geojson['features'])} features")
except Exception as e:
    print(f"ERROR loading basin polygons: {e}")
    basin_geojson = None

def get_basin_polygon(geojson, gauge_id):
    """Extract polygon coordinates for a given gauge_id."""
    if geojson is None:
        return None
    for feature in geojson['features']:
        fid = str(feature['properties'].get('grdc_no', ''))
        if fid == str(gauge_id):
            return feature['geometry']
    return None

def point_in_polygon_ray(px, py, polygon_coords):
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon_coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i]
        xj, yj = polygon_coords[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def get_basin_mask_gridded(lat, lon, geometry):
    """Create a boolean mask for a gridded dataset based on basin polygon.
    lat and lon are 1D arrays."""
    if geometry is None:
        return None
    
    geom_type = geometry['type']
    if geom_type == 'Polygon':
        polygons = [geometry['coordinates'][0]]
    elif geom_type == 'MultiPolygon':
        polygons = [poly[0] for poly in geometry['coordinates']]
    else:
        return None
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = np.zeros(lon2d.shape, dtype=bool)
    
    for poly_coords in polygons:
        poly_arr = np.array(poly_coords)
        # Quick bounding box check
        min_lon, max_lon = poly_arr[:, 0].min(), poly_arr[:, 0].max()
        min_lat, max_lat = poly_arr[:, 1].min(), poly_arr[:, 1].max()
        
        for i in range(lat2d.shape[0]):
            for j in range(lat2d.shape[1]):
                if mask[i, j]:
                    continue
                plat = lat2d[i, j]
                plon = lon2d[i, j]
                if min_lat <= plat <= max_lat and min_lon <= plon <= max_lon:
                    if point_in_polygon_ray(plon, plat, poly_coords):
                        mask[i, j] = True
    
    return mask

def get_basin_mask_unstructured(grid_lat, grid_lon, geometry):
    """Create a boolean mask for unstructured grid based on basin polygon."""
    if geometry is None:
        return None
    
    geom_type = geometry['type']
    if geom_type == 'Polygon':
        polygons = [geometry['coordinates'][0]]
    elif geom_type == 'MultiPolygon':
        polygons = [poly[0] for poly in geometry['coordinates']]
    else:
        return None
    
    # Adjust grid_lon to -180 to 180
    grid_lon_adj = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    
    mask = np.zeros(len(grid_lat), dtype=bool)
    
    for poly_coords in polygons:
        poly_arr = np.array(poly_coords)
        min_lon, max_lon = poly_arr[:, 0].min(), poly_arr[:, 0].max()
        min_lat, max_lat = poly_arr[:, 1].min(), poly_arr[:, 1].max()
        
        for i in range(len(grid_lat)):
            if mask[i]:
                continue
            plat = grid_lat[i]
            plon = grid_lon_adj[i]
            if min_lat <= plat <= max_lat and min_lon <= plon <= max_lon:
                if point_in_polygon_ray(plon, plat, poly_coords):
                    mask[i] = True
    
    return mask

def compute_basin_mean_gridded(data, lat, lon, mask):
    """Compute area-weighted basin mean for gridded data."""
    if mask is None or not mask.any():
        return np.nan
    
    data_vals = data.values if hasattr(data, 'values') else data
    
    # Create latitude weights
    lat_weights = np.cos(np.deg2rad(lat))
    weights_2d = np.broadcast_to(lat_weights[:, np.newaxis], data_vals.shape)
    
    masked_data = np.where(mask & np.isfinite(data_vals), data_vals, np.nan)
    masked_weights = np.where(mask & np.isfinite(data_vals), weights_2d, 0)
    
    total_weight = np.nansum(masked_weights)
    if total_weight == 0:
        return np.nan
    
    return np.nansum(masked_data * masked_weights) / total_weight

def compute_basin_mean_unstructured(data, grid_lat, mask):
    """Compute area-weighted basin mean for unstructured data."""
    if mask is None or not mask.any():
        return np.nan
    
    data_vals = data.values if hasattr(data, 'values') else data
    
    weights = np.cos(np.deg2rad(grid_lat[mask]))
    vals = data_vals[mask]
    
    valid = np.isfinite(vals)
    if not valid.any():
        return np.nan
    
    return np.average(vals[valid], weights=weights[valid])

# Determine if ELM data is structured or unstructured
elm_is_unstructured = False
if model_P is not None:
    if model_P.ndim == 1:
        elm_is_unstructured = True
        print("ELM data appears to be unstructured (1D)")
    else:
        print(f"ELM data appears to be structured ({model_P.ndim}D)")

# Store results
basin_results = {}

for gid in basin_ids:
    bname = basins[gid]
    print(f"\nProcessing basin: {bname} ({gid})")
    
    geometry = get_basin_polygon(basin_geojson, gid)
    if geometry is None:
        print(f"  WARNING: No polygon found for {bname}")
        basin_results[gid] = {
            'name': bname,
            'model_P': np.nan, 'model_ET': np.nan, 'model_Q': np.nan,
            'obs_P': np.nan, 'obs_ET': np.nan, 'obs_Q': np.nan
        }
        continue
    
    result = {'name': bname}
    
    # Model basin means
    if elm_is_unstructured and model_P is not None:
        mask_elm = get_basin_mask_unstructured(elm_lat, elm_lon, geometry)
        if mask_elm is not None:
            n_cells = mask_elm.sum()
            print(f"  ELM cells in basin: {n_cells}")
            result['model_P'] = compute_basin_mean_unstructured(model_P, elm_lat, mask_elm)
            result['model_ET'] = compute_basin_mean_unstructured(model_ET, elm_lat, mask_elm)
            result['model_Q'] = compute_basin_mean_unstructured(model_Q, elm_lat, mask_elm)
        else:
            result['model_P'] = result['model_ET'] = result['model_Q'] = np.nan
    elif model_P is not None and model_P.ndim == 2:
        # Get lat/lon coordinate names
        lat_dim = [d for d in model_P.dims if 'lat' in d.lower()]
        lon_dim = [d for d in model_P.dims if 'lon' in d.lower()]
        if lat_dim and lon_dim:
            lat_vals = model_P.coords[lat_dim[0]].values
            lon_vals = model_P.coords[lon_dim[0]].values
            # Shift lon to -180..180 if needed
            if lon_vals.max() > 180:
                lon_shifted = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
                sort_idx = np.argsort(lon_shifted)
                lon_vals = lon_shifted[sort_idx]
                model_P_shifted = model_P.isel(**{lon_dim[0]: sort_idx})
                model_ET_shifted = model_ET.isel(**{lon_dim[0]: sort_idx})
                model_Q_shifted = model_Q.isel(**{lon_dim[0]: sort_idx})
            else:
                model_P_shifted = model_P
                model_ET_shifted = model_ET
                model_Q_shifted = model_Q
            
            mask_elm = get_basin_mask_gridded(lat_vals, lon_vals, geometry)
            n_cells = mask_elm.sum() if mask_elm is not None else 0
            print(f"  ELM grid cells in basin: {n_cells}")
            result['model_P'] = compute_basin_mean_gridded(model_P_shifted, lat_vals, lon_vals, mask_elm)
            result['model_ET'] = compute_basin_mean_gridded(model_ET_shifted, lat_vals, lon_vals, mask_elm)
            result['model_Q'] = compute_basin_mean_gridded(model_Q_shifted, lat_vals, lon_vals, mask_elm)
        else:
            result['model_P'] = result['model_ET'] = result['model_Q'] = np.nan
    else:
        result['model_P'] = result['model_ET'] = result['model_Q'] = np.nan
    
    # Obs P basin mean
    if obs_P is not None:
        lat_name_p = [d for d in obs_P.dims if 'lat' in d.lower()]
        lon_name_p = [d for d in obs_P.dims if 'lon' in d.lower()]
        if lat_name_p and lon_name_p:
            mask_p = get_basin_mask_gridded(obs_P_lat, obs_P_lon, geometry)
            n_cells = mask_p.sum() if mask_p is not None else 0
            print(f"  GPCC grid cells in basin: {n_cells}")
            result['obs_P'] = compute_basin_mean_gridded(obs_P, obs_P_lat, obs_P_lon, mask_p)
        else:
            result['obs_P'] = np.nan
    else:
        result['obs_P'] = np.nan
    
    # Obs ET basin mean
    if obs_ET is not None:
        lat_name_e = [d for d in obs_ET.dims if 'lat' in d.lower()]
        lon_name_e = [d for d in obs_ET.dims if 'lon' in d.lower()]
        if lat_name_e and lon_name_e:
            mask_et = get_basin_mask_gridded(obs_ET_lat, obs_ET_lon, geometry)
            n_cells = mask_et.sum() if mask_et is not None else 0
            print(f"  MODIS grid cells in basin: {n_cells}")
            result['obs_ET'] = compute_basin_mean_gridded(obs_ET, obs_ET_lat, obs_ET_lon, mask_et)
        else:
            result['obs_ET'] = np.nan
    else:
        result['obs_ET'] = np.nan
    
    # Obs runoff basin mean
    if obs_Qr is not None:
        lat_name_q = [d for d in obs_Qr.dims if 'lat' in d.lower()]
        lon_name_q = [d for d in obs_Qr.dims if 'lon' in d.lower()]
        if lat_name_q and lon_name_q:
            mask_qr = get_basin_mask_gridded(obs_Qr_lat, obs_Qr_lon, geometry)
            n_cells = mask_qr.sum() if mask_qr is not None else 0
            print(f"  LORA grid cells in basin: {n_cells}")
            result['obs_Q'] = compute_basin_mean_gridded(obs_Qr, obs_Qr_lat, obs_Qr_lon, mask_qr)
        else:
            result['obs_Q'] = np.nan
    else:
        result['obs_Q'] = np.nan
    
    basin_results[gid] = result
    print(f"  Model P={result['model_P']:.3f}, ET={result['model_ET']:.3f}, Q={result['model_Q']:.3f} mm/day")
    print(f"  Obs   P={result['obs_P']:.3f}, ET={result['obs_ET']:.3f}, Q={result['obs_Q']:.3f} mm/day")

# ============================================================
# PART 4: Streamflow FDC metrics
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Streamflow flow duration curve analysis")
print("=" * 60)

# Load gauge metadata
try:
    gauge_meta = pd.read_csv(gauge_meta_path)
    print(f"Loaded gauge metadata: {len(gauge_meta)} gauges")
    print(f"Columns: {list(gauge_meta.columns)}")
except Exception as e:
    print(f"ERROR loading gauge metadata: {e}")
    gauge_meta = pd.DataFrame()

# Load MOSART data
mosart_files_monthly = []
mosart_files_daily = []
for yr in years:
    for mo in months:
        mf = os.path.join(rof_dir, f"{case_name}.mosart.h0.{yr:04d}-{mo:02d}.nc")
        if os.path.exists(mf):
            mosart_files_monthly.append(mf)

# Also check for daily files
for yr in years:
    for mo in months:
        for day in range(1, 32):
            df = os.path.join(rof_dir, f"{case_name}.mosart.h1.{yr:04d}-{mo:02d}-{day:02d}-00000.nc")
            if os.path.exists(df):
                mosart_files_daily.append(df)

print(f"Found {len(mosart_files_monthly)} MOSART monthly files")
print(f"Found {len(mosart_files_daily)} MOSART daily files")

# Use daily files if available, otherwise monthly
mosart_files = mosart_files_daily if len(mosart_files_daily) > 0 else mosart_files_monthly
mosart_freq = 'daily' if len(mosart_files_daily) > 0 else 'monthly'
print(f"Using {mosart_freq} MOSART files ({len(mosart_files)} files)")

ds_mosart = None
mosart_lat = None
mosart_lon = None

if len(mosart_files) > 0:
    try:
        ds_mosart = xr.open_mfdataset(mosart_files, combine='by_coords', decode_times=True)
        discharge_var = 'RIVER_DISCHARGE_OVER_LAND_LIQ'
        if discharge_var not in ds_mosart:
            print(f"  WARNING: {discharge_var} not found. Available: {list(ds_mosart.data_vars)}")
            # Try to find a similar variable
            for v in ds_mosart.data_vars:
                if 'discharge' in v.lower() or 'river' in v.lower():
                    discharge_var = v
                    break
        
        if 'lat' in ds_mosart:
            mosart_lat = ds_mosart['lat'].values
            mosart_lon = ds_mosart['lon'].values
        
        print(f"  MOSART discharge var: {discharge_var}")
        if mosart_lat is not None:
            print(f"  MOSART grid: {mosart_lat.shape}")
    except Exception as e:
        print(f"ERROR loading MOSART: {e}")
        import traceback
        traceback.print_exc()

def find_nearest_grid_cell(lat, lon, grid_lat, grid_lon):
    """Find nearest grid cell index for a given lat/lon."""
    grid_lon_adj = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    
    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        # Could be structured or unstructured
        if len(grid_lat) == len(grid_lon):
            # Unstructured
            dist = np.sqrt((grid_lat - lat)**2 + (grid_lon_adj - lon)**2)
            return np.argmin(dist)
        else:
            # Structured - find nearest lat and lon indices
            lat_idx = np.argmin(np.abs(grid_lat - lat))
            lon_idx = np.argmin(np.abs(grid_lon_adj - lon))
            return (lat_idx, lon_idx)
    elif grid_lat.ndim == 2:
        dist = np.sqrt((grid_lat - lat)**2 + (grid_lon_adj - lon)**2)
        return np.unravel_index(np.argmin(dist), grid_lat.shape)
    
    return None

def compute_wasserstein_distance(x, y):
    """Compute 1D Wasserstein distance between two empirical distributions."""
    try:
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(x, y)
    except:
        # Manual computation
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        # Interpolate to common quantiles
        n = min(len(x_sorted), len(y_sorted), 1000)
        quantiles = np.linspace(0, 1, n)
        x_q = np.quantile(x_sorted, quantiles)
        y_q = np.quantile(y_sorted, quantiles)
        return np.mean(np.abs(x_q - y_q))

# Process each basin for streamflow
streamflow_results = {}

for gid in basin_ids:
    bname = basins[gid]
    print(f"\nStreamflow analysis: {bname} ({gid})")
    
    result = {'name': bname}
    
    # Load observed discharge
    obs_file = os.path.join(streamflow_obs_dir, f"{gid}.csv")
    obs_discharge = None
    try:
        if os.path.exists(obs_file):
            obs_df = pd.read_csv(obs_file, parse_dates=['date'])
            obs_df = obs_df[(obs_df['date'] >= '1985-01-01') & (obs_df['date'] <= '1989-12-31')]
            obs_df = obs_df.dropna(subset=['discharge_m3s'])
            if len(obs_df) > 0:
                obs_discharge = obs_df['discharge_m3s'].values
                print(f"  Obs discharge: {len(obs_discharge)} values, mean={np.mean(obs_discharge):.1f} m3/s")
            else:
                print(f"  WARNING: No observed discharge in 1985-1989 for {gid}")
        else:
            print(f"  WARNING: Obs file not found: {obs_file}")
    except Exception as e:
        print(f"  ERROR loading obs discharge: {e}")
    
    # Get gauge location
    gauge_lat = None
    gauge_lon = None
    if not gauge_meta.empty:
        gauge_row = gauge_meta[gauge_meta['gauge_id'].astype(str) == str(gid)]
        if len(gauge_row) > 0:
            gauge_lat = float(gauge_row['lat'].iloc[0])
            gauge_lon = float(gauge_row['lon'].iloc[0])
            print(f"  Gauge location: lat={gauge_lat}, lon={gauge_lon}")
    
    # Extract simulated discharge
    sim_discharge = None
    if ds_mosart is not None and gauge_lat is not None and mosart_lat is not None:
        try:
            idx = find_nearest_grid_cell(gauge_lat, gauge_lon, mosart_lat, mosart_lon)
            print(f"  Nearest MOSART grid cell index: {idx}")
            
            discharge_data = ds_mosart[discharge_var]
            
            if isinstance(idx, tuple):
                # Structured grid
                lat_dim_name = discharge_data.dims[-2] if len(discharge_data.dims) >= 3 else discharge_data.dims[-1]
                lon_dim_name = discharge_data.dims[-1]
                sim_ts = discharge_data.isel(**{lat_dim_name: idx[0], lon_dim_name: idx[1]})
            else:
                # Unstructured
                # Find the spatial dimension name
                spatial_dims = [d for d in discharge_data.dims if d != 'time']
                if spatial_dims:
                    sim_ts = discharge_data.isel(**{spatial_dims[0]: idx})
                else:
                    sim_ts = discharge_data
            
            sim_discharge = sim_ts.values.flatten()
            sim_discharge = sim_discharge[np.isfinite(sim_discharge)]
            
            if len(sim_discharge) > 0:
                print(f"  Sim discharge: {len(sim_discharge)} values, mean={np.mean(sim_discharge):.1f} m3/s")
            else:
                print(f"  WARNING: No valid simulated discharge")
                sim_discharge = None
        except Exception as e:
            print(f"  ERROR extracting simulated discharge: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute FDC metrics
    if obs_discharge is not None and sim_discharge is not None and len(obs_discharge) > 0 and len(sim_discharge) > 0:
        obs_mean = np.mean(obs_discharge)
        sim_mean = np.mean(sim_discharge)
        volume_bias = (sim_mean - obs_mean) / obs_mean * 100 if obs_mean != 0 else np.nan
        
        wass_dist = compute_wasserstein_distance(sim_discharge, obs_discharge)
        
        result['obs_streamflow_mean'] = obs_mean
        result['sim_streamflow_mean'] = sim_mean
        result['volume_bias_pct'] = volume_bias
        result['wasserstein_distance'] = wass_dist
        
        print(f"  Volume bias: {volume_bias:.1f}%")
        print(f"  Wasserstein distance: {wass_dist:.1f} m3/s")
    else:
        result['obs_streamflow_mean'] = np.nan
        result['sim_streamflow_mean'] = np.nan
        result['volume_bias_pct'] = np.nan
        result['wasserstein_distance'] = np.nan
    
    streamflow_results[gid] = result

if ds_mosart is not None:
    ds_mosart.close()

# ============================================================
# PART 5: Combine and visualize
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Combining metrics and creating visualizations")
print("=" * 60)

# Build summary table
summary_data = []
for gid in basin_ids:
    bname = basins[gid]
    br = basin_results.get(gid, {})
    sr = streamflow_results.get(gid, {})
    
    mod_P = br.get('model_P', np.nan)
    obs_P_val = br.get('obs_P', np.nan)
    mod_ET = br.get('model_ET', np.nan)
    obs_ET_val = br.get('obs_ET', np.nan)
    mod_Q = br.get('model_Q', np.nan)
    obs_Q_val = br.get('obs_Q', np.nan)
    
    P_bias = (mod_P - obs_P_val) / obs_P_val * 100 if (np.isfinite(obs_P_val) and obs_P_val != 0) else np.nan
    ET_bias = (mod_ET - obs_ET_val) / obs_ET_val * 100 if (np.isfinite(obs_ET_val) and obs_ET_val != 0) else np.nan
    Q_bias = (mod_Q - obs_Q_val) / obs_Q_val * 100 if (np.isfinite(obs_Q_val) and obs_Q_val != 0) else np.nan
    
    # Water balance residual: P - ET - Q (should be ~0 in steady state)
    # Model residual
    mod_residual = mod_P - mod_ET - mod_Q if all(np.isfinite([mod_P, mod_ET, mod_Q])) else np.nan
    obs_residual = obs_P_val - obs_ET_val - obs_Q_val if all(np.isfinite([obs_P_val, obs_ET_val, obs_Q_val])) else np.nan
    
    vol_bias = sr.get('volume_bias_pct', np.nan)
    wass = sr.get('wasserstein_distance', np.nan)
    
    summary_data.append({
        'basin': bname,
        'gauge_id': gid,
        'model_P_mm_day': mod_P,
        'obs_P_mm_day': obs_P_val,
        'P_bias_pct': P_bias,
        'model_ET_mm_day': mod_ET,
        'obs_ET_mm_day': obs_ET_val,
        'ET_bias_pct': ET_bias,
        'model_runoff_mm_day': mod_Q,
        'obs_runoff_mm_day': obs_Q_val,
        'runoff_bias_pct': Q_bias,
        'streamflow_volume_bias_pct': vol_bias,
        'wasserstein_distance_m3s': wass,
        'model_wb_residual_mm_day': mod_residual,
        'obs_wb_residual_mm_day': obs_residual
    })

summary_df = pd.DataFrame(summary_data)

# Save summary table
try:
    summary_path = os.path.join(output_dir, "basin_summary_table.csv")
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"\nSaved summary table: {summary_path}")
    print("\n" + summary_df.to_string(index=False))
except Exception as e:
    print(f"ERROR saving summary table: {e}")

# ---- Bar chart: Model vs Observation P, ET, Q for each basin ----
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    x = np.arange(len(basin_ids))
    width = 0.35
    
    variables = [
        ('P', 'model_P_mm_day', 'obs_P_mm_day', 'Precipitation (mm/day)'),
        ('ET', 'model_ET_mm_day', 'obs_ET_mm_day', 'Evapotranspiration (mm/day)'),
        ('Q', 'model_runoff_mm_day', 'obs_runoff_mm_day', 'Runoff (mm/day)')
    ]
    
    for ax, (var_label, mod_col, obs_col, ylabel) in zip(axes, variables):
        mod_vals = summary_df[mod_col].values
        obs_vals = summary_df[obs_col].values
        
        bars1 = ax.bar(x - width/2, mod_vals, width, label='E3SM', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, obs_vals, width, label='Obs', color='darkorange', alpha=0.8)
        
        ax.set_xlabel('Basin')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{var_label}: Model vs Observation')
        ax.set_xticks(x)
        ax.set_xticklabels(basin_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add bias labels
        for i in range(len(basin_ids)):
            if np.isfinite(mod_vals[i]) and np.isfinite(obs_vals[i]) and obs_vals[i] != 0:
                bias_pct = (mod_vals[i] - obs_vals[i]) / obs_vals[i] * 100
                max_val = max(mod_vals[i], obs_vals[i])
                ax.text(x[i], max_val * 1.02, f'{bias_pct:+.0f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color='red' if abs(bias_pct) > 20 else 'green')
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "model_vs_obs_bar_chart.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved bar chart: {bar_path}")
except Exception as e:
    print(f"ERROR creating bar chart: {e}")
    import traceback
    traceback.print_exc()

# ---- Radar chart: Multi-variable diagnostic for each basin ----
try:
    # Variables for radar: P bias, ET bias, runoff bias, streamflow bias, |WB residual|, Wasserstein
    # Normalize each to a 0-1 scale (using max abs value across basins)
    
    radar_vars = ['P_bias_pct', 'ET_bias_pct', 'runoff_bias_pct', 
                  'streamflow_volume_bias_pct', 'model_wb_residual_mm_day', 'wasserstein_distance_m3s']
    radar_labels = ['P bias (%)', 'ET bias (%)', 'Runoff bias (%)', 
                    'Streamflow bias (%)', 'WB residual\n(mm/day)', 'Wasserstein\n(m³/s)']
    
    # Get absolute values for radar (showing magnitude of biases)
    radar_data = {}
    for gid in basin_ids:
        bname = basins[gid]
        row = summary_df[summary_df['gauge_id'] == gid].iloc[0]
        vals = []
        for var in radar_vars:
            v = row[var]
            vals.append(abs(v) if np.isfinite(v) else 0)
        radar_data[bname] = vals
    
    # Normalize each variable
    num_vars = len(radar_vars)
    max_vals = np.zeros(num_vars)
    for i in range(num_vars):
        all_vals = [radar_data[bname][i] for bname in basin_names]
        max_vals[i] = max(all_vals) if max(all_vals) > 0 else 1
    
    normalized_data = {}
    for bname in basin_names:
        normalized_data[bname] = [radar_data[bname][i] / max_vals[i] for i in range(num_vars)]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    
    for i, bname in enumerate(basin_names):
        values = normalized_data[bname] + normalized_data[bname][:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=bname, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.05, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax.set_title('Multi-Variable Diagnostic\n(Normalized |Bias| Across Basins)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Add actual values as annotations
    for i, bname in enumerate(basin_names):
        for j in range(num_vars):
            actual_val = radar_data[bname][j]
            if actual_val > 0:
                r = normalized_data[bname][j]
                theta = angles[j]
    
    plt.tight_layout()
    radar_path = os.path.join(output_dir, "radar_diagnostic.png")
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved radar chart: {radar_path}")
except Exception as e:
    print(f"ERROR creating radar chart: {e}")
    import traceback
    traceback.print_exc()

# ---- Additional: Heatmap of biases ----
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bias_cols = ['P_bias_pct', 'ET_bias_pct', 'runoff_bias_pct', 
                 'streamflow_volume_bias_pct']
    bias_labels = ['P bias (%)', 'ET bias (%)', 'Runoff bias (%)', 'Streamflow bias (%)']
    
    data_matrix = np.zeros((len(basin_ids), len(bias_cols)))
    for i, gid in enumerate(basin_ids):
        row = summary_df[summary_df['gauge_id'] == gid].iloc[0]
        for j, col in enumerate(bias_cols):
            data_matrix[i, j] = row[col] if np.isfinite(row[col]) else 0
    
    # Create heatmap
    vmax = max(abs(np.nanmin(data_matrix)), abs(np.nanmax(data_matrix)))
    vmax = min(vmax, 200)  # Cap at 200%
    
    im = ax.imshow(data_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    
    ax.set_xticks(range(len(bias_labels)))
    ax.set_xticklabels(bias_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(basin_names)))
    ax.set_yticklabels(basin_names)
    
    # Add text annotations
    for i in range(len(basin_names)):
        for j in range(len(bias_cols)):
            val = data_matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax, label='Bias (%)')
    ax.set_title('E3SM Water Cycle Biases by Basin (1985-1989)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "bias_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bias heatmap: {heatmap_path}")
except Exception as e:
    print(f"ERROR creating heatmap: {e}")
    import traceback
    traceback.print_exc()

# ---- Water balance diagram ----
try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, gid in enumerate(basin_ids):
        ax = axes[idx]
        bname = basins[gid]
        br = basin_results.get(gid, {})
        
        mod_vals = [br.get('model_P', 0), br.get('model_ET', 0), br.get('model_Q', 0)]
        obs_vals_plot = [br.get('obs_P', 0), br.get('obs_ET', 0), br.get('obs_Q', 0)]
        
        # Replace NaN with 0 for plotting
        mod_vals = [v if np.isfinite(v) else 0 for v in mod_vals]
        obs_vals_plot = [v if np.isfinite(v) else 0 for v in obs_vals_plot]
        
        categories = ['P', 'ET', 'Q']
        x_pos = np.arange(3)
        
        bars1 = ax.bar(x_pos - 0.2, mod_vals, 0.35, label='E3SM', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x_pos + 0.2, obs_vals_plot, 0.35, label='Obs', color='darkorange', alpha=0.8)
        
        # Add residual annotation
        mod_res = mod_vals[0] - mod_vals[1] - mod_vals[2]
        obs_res = obs_vals_plot[0] - obs_vals_plot[1] - obs_vals_plot[2]
        
        ax.set_title(f'{bname}\nModel Res={mod_res:.2f}, Obs Res={obs_res:.2f} mm/day', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)
        ax.set_ylabel('mm/day')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Per-Basin Water Balance: E3SM vs Observations (1985-1989)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    wb_path = os.path.join(output_dir, "water_balance_per_basin.png")
    plt.savefig(wb_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved water balance diagram: {wb_path}")
except Exception as e:
    print(f"ERROR creating water balance diagram: {e}")
    import traceback
    traceback.print_exc()

# Final summary print
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nOutput directory: {output_dir}")
print(f"Files created:")
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    fsize = os.path.getsize(fpath) / 1024
    print(f"  {f} ({fsize:.1f} KB)")

print("\n\nBasin Summary:")
print(summary_df[['basin', 'P_bias_pct', 'ET_bias_pct', 'runoff_bias_pct', 
                   'streamflow_volume_bias_pct', 'wasserstein_distance_m3s',
                   'model_wb_residual_mm_day']].to_string(index=False))

print("\nDone!")