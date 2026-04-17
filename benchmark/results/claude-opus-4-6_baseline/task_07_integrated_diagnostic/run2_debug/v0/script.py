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
OUTDIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_07_integrated_diagnostic/run2_output"
os.makedirs(OUTDIR, exist_ok=True)

# Data paths
E3SM_LND = "./data/sample/e3sm/lnd"
E3SM_ROF = "./data/sample/e3sm/rof"
CASE = "sample.v3.LR.historical"
BASIN_POLY = "./data/sample/obs/basin_polygons.geojson"
GAUGE_META = "./data/sample/obs/gauge_metadata.csv"
OBS_FLOW_DIR = "./data/sample/obs/streamflow"

# Basin definitions
BASINS = {
    '3629000': 'Amazon',
    '4121801': 'Missouri',
    '4115200': 'Columbia',
    '6742900': 'Danube',
    '2969100': 'Mekong',
    '1159100': 'Orange'
}

YEARS = range(1985, 1990)
MONTHS = range(1, 13)

# Conversion factors
# ELM: RAIN, SNOW in kg/m2/s = mm/s; QVEGE, QVEGT, QSOIL in mm/s; QRUNOFF in mm/s
# Convert mm/s to mm/day: * 86400
# GPCC pr: mm/day
# MODIS ET: variable 'et' - check units
# LORA mrro: kg/m2/s

def log(msg):
    print(f"[INFO] {msg}", flush=True)

###############################################################################
# Part 1: Extract model fields from ELM monthly output
###############################################################################
log("Part 1: Extracting model fields from ELM monthly output...")

elm_files = []
for yr in YEARS:
    for mo in MONTHS:
        fname = os.path.join(E3SM_LND, f"{CASE}.elm.h0.{yr:04d}-{mo:02d}.nc")
        if os.path.exists(fname):
            elm_files.append(fname)

log(f"Found {len(elm_files)} ELM files")

try:
    ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True, chunks={})
    log(f"ELM dataset loaded: {list(ds_elm.dims)}")
    
    # Check available variables
    avail_vars = list(ds_elm.data_vars)
    log(f"Available ELM variables (sample): {avail_vars[:20]}")
    
    # Precipitation = RAIN + SNOW (mm/s -> mm/day)
    if 'RAIN' in ds_elm and 'SNOW' in ds_elm:
        model_precip = (ds_elm['RAIN'] + ds_elm['SNOW']) * 86400.0  # mm/day
    elif 'QOVER' in ds_elm:
        log("RAIN/SNOW not found, looking for alternatives...")
        model_precip = None
    else:
        model_precip = None
    
    # ET = QVEGE + QVEGT + QSOIL (mm/s -> mm/day)
    et_vars = ['QVEGE', 'QVEGT', 'QSOIL']
    if all(v in ds_elm for v in et_vars):
        model_et = (ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']) * 86400.0  # mm/day
    else:
        log(f"Missing ET vars. Available: {[v for v in et_vars if v in ds_elm]}")
        model_et = None
    
    # Runoff = QRUNOFF (mm/s -> mm/day)
    if 'QRUNOFF' in ds_elm:
        model_runoff = ds_elm['QRUNOFF'] * 86400.0  # mm/day
    else:
        model_runoff = None
    
    # Compute climatological time-mean
    if model_precip is not None:
        model_precip_clim = model_precip.mean(dim='time').compute()
        log(f"Model P climatology shape: {model_precip_clim.shape}")
    if model_et is not None:
        model_et_clim = model_et.mean(dim='time').compute()
        log(f"Model ET climatology shape: {model_et_clim.shape}")
    if model_runoff is not None:
        model_runoff_clim = model_runoff.mean(dim='time').compute()
        log(f"Model Q climatology shape: {model_runoff_clim.shape}")
    
    # Get lat/lon from ELM
    if 'lat' in ds_elm:
        elm_lat = ds_elm['lat'].values
        elm_lon = ds_elm['lon'].values
    elif 'lsmlat' in ds_elm.dims:
        elm_lat = ds_elm['lsmlat'].values
        elm_lon = ds_elm['lsmlon'].values
    else:
        # Try to find coordinate variables
        for coord_name in ds_elm.coords:
            log(f"  ELM coord: {coord_name} shape={ds_elm[coord_name].shape}")
        elm_lat = None
        elm_lon = None
    
    ds_elm.close()
except Exception as e:
    log(f"Error loading ELM data: {e}")
    import traceback; traceback.print_exc()
    model_precip_clim = None
    model_et_clim = None
    model_runoff_clim = None
    elm_lat = None
    elm_lon = None

###############################################################################
# Part 2: Fetch and extract observation fields from ILAMB
###############################################################################
log("Part 2: Fetching observation data from ILAMB...")

ILAMB_BASE = "https://www.ilamb.org/ILAMB-Data/DATA"
OBS_CACHE = "./data/sample/obs/ilamb_cache"
os.makedirs(OBS_CACHE, exist_ok=True)

def fetch_ilamb(variable, dataset, filename):
    """Download ILAMB observation data if not cached."""
    url = f"{ILAMB_BASE}/{variable}/{dataset}/{filename}"
    local_path = os.path.join(OBS_CACHE, f"{variable}_{dataset}_{filename}")
    if os.path.exists(local_path):
        log(f"Using cached: {local_path}")
        return local_path
    log(f"Downloading: {url}")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, local_path)
        log(f"Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        log(f"Download failed: {e}")
        return None

# GPCC Precipitation
obs_precip_clim = None
try:
    pr_path = fetch_ilamb("pr", "GPCPv2.3", "pr.nc")
    if pr_path is None:
        pr_path = fetch_ilamb("pr", "GPCCv2018", "pr.nc")
    if pr_path is not None:
        ds_pr = xr.open_dataset(pr_path, decode_times=True)
        log(f"GPCC PR variables: {list(ds_pr.data_vars)}")
        log(f"GPCC PR coords: {list(ds_pr.coords)}")
        # Select 1985-1989
        pr_var = 'pr' if 'pr' in ds_pr else list(ds_pr.data_vars)[0]
        pr_data = ds_pr[pr_var]
        # Time selection
        try:
            pr_data = pr_data.sel(time=slice('1985-01-01', '1989-12-31'))
        except:
            log("Time selection failed for PR, using all times")
        obs_precip_clim = pr_data.mean(dim='time').compute()
        # Get coordinates
        obs_pr_lat = ds_pr['lat'].values if 'lat' in ds_pr else None
        obs_pr_lon = ds_pr['lon'].values if 'lon' in ds_pr else None
        # Handle 0-360 convention
        if obs_pr_lon is not None and np.any(obs_pr_lon > 180):
            log("GPCC PR uses 0-360 lon convention")
        log(f"Obs P climatology shape: {obs_precip_clim.shape}")
        ds_pr.close()
except Exception as e:
    log(f"Error loading GPCC data: {e}")
    import traceback; traceback.print_exc()

# MODIS ET
obs_et_clim = None
try:
    et_path = fetch_ilamb("evspsbl", "MODIS", "et_0.5x0.5.nc")
    if et_path is not None:
        ds_et = xr.open_dataset(et_path, decode_times=True)
        log(f"MODIS ET variables: {list(ds_et.data_vars)}")
        log(f"MODIS ET coords: {list(ds_et.coords)}")
        et_var = 'et' if 'et' in ds_et else 'evspsbl' if 'evspsbl' in ds_et else list(ds_et.data_vars)[0]
        et_data = ds_et[et_var]
        try:
            et_data = et_data.sel(time=slice('1985-01-01', '1989-12-31'))
        except:
            log("Time selection failed for ET, using all times")
        obs_et_clim = et_data.mean(dim='time').compute()
        # Convert units if needed (kg/m2/s -> mm/day)
        units = ds_et[et_var].attrs.get('units', '')
        log(f"MODIS ET units: {units}")
        if 'kg' in units.lower() or 'kg m-2 s-1' in units.lower():
            obs_et_clim = obs_et_clim * 86400.0
        log(f"Obs ET climatology shape: {obs_et_clim.shape}")
        ds_et.close()
except Exception as e:
    log(f"Error loading MODIS ET data: {e}")
    import traceback; traceback.print_exc()

# LORA Runoff
obs_runoff_clim = None
try:
    ro_path = fetch_ilamb("mrro", "LORA", "LORA.nc")
    if ro_path is not None:
        ds_ro = xr.open_dataset(ro_path, decode_times=True)
        log(f"LORA variables: {list(ds_ro.data_vars)}")
        ro_var = 'mrro' if 'mrro' in ds_ro else list(ds_ro.data_vars)[0]
        ro_data = ds_ro[ro_var]
        try:
            ro_data = ro_data.sel(time=slice('1985-01-01', '1989-12-31'))
        except:
            log("Time selection failed for runoff, using all times")
        obs_runoff_clim = ro_data.mean(dim='time').compute()
        units = ds_ro[ro_var].attrs.get('units', '')
        log(f"LORA runoff units: {units}")
        if 'kg' in units.lower() or 'kg m-2 s-1' in units.lower():
            obs_runoff_clim = obs_runoff_clim * 86400.0
        log(f"Obs runoff climatology shape: {obs_runoff_clim.shape}")
        ds_ro.close()
except Exception as e:
    log(f"Error loading LORA data: {e}")
    import traceback; traceback.print_exc()

###############################################################################
# Part 3: Clip to basin means using basin polygons
###############################################################################
log("Part 3: Clipping fields to basin-averaged values...")

# Load basin polygons
try:
    with open(BASIN_POLY, 'r') as f:
        basins_geojson = json.load(f)
    log(f"Loaded {len(basins_geojson['features'])} basin polygons")
except Exception as e:
    log(f"Error loading basin polygons: {e}")
    basins_geojson = None

def get_basin_polygon(geojson, gauge_id):
    """Extract polygon coordinates for a given gauge_id."""
    gauge_id_str = str(gauge_id)
    for feature in geojson['features']:
        props = feature['properties']
        fid = str(props.get('grdc_no', props.get('gauge_id', '')))
        if fid == gauge_id_str:
            return feature['geometry']
    return None

def point_in_polygon(x, y, polygon_coords):
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon_coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i]
        xj, yj = polygon_coords[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def create_basin_mask_2d(lat_2d, lon_2d, geometry):
    """Create a mask for a 2D grid given a polygon geometry."""
    mask = np.zeros(lat_2d.shape, dtype=bool)
    
    if geometry['type'] == 'Polygon':
        polygons = [geometry['coordinates']]
    elif geometry['type'] == 'MultiPolygon':
        polygons = geometry['coordinates']
    else:
        return mask
    
    for poly in polygons:
        exterior = poly[0]  # exterior ring
        coords = np.array(exterior)
        # Quick bounding box check
        min_lon, max_lon = coords[:, 0].min(), coords[:, 0].max()
        min_lat, max_lat = coords[:, 1].min(), coords[:, 1].max()
        
        for i in range(lat_2d.shape[0]):
            for j in range(lat_2d.shape[1]):
                if mask[i, j]:
                    continue
                lt = lat_2d[i, j]
                ln = lon_2d[i, j]
                if min_lat <= lt <= max_lat and min_lon <= ln <= max_lon:
                    if point_in_polygon(ln, lt, exterior):
                        mask[i, j] = True
    return mask

def create_basin_mask_1d(lats, lons, geometry):
    """Create a mask for a 1D unstructured grid given a polygon geometry."""
    mask = np.zeros(len(lats), dtype=bool)
    
    if geometry['type'] == 'Polygon':
        polygons = [geometry['coordinates']]
    elif geometry['type'] == 'MultiPolygon':
        polygons = geometry['coordinates']
    else:
        return mask
    
    for poly in polygons:
        exterior = poly[0]
        coords = np.array(exterior)
        min_lon, max_lon = coords[:, 0].min(), coords[:, 0].max()
        min_lat, max_lat = coords[:, 1].min(), coords[:, 1].max()
        
        for i in range(len(lats)):
            if mask[i]:
                continue
            lt = lats[i]
            ln = lons[i]
            if min_lat <= lt <= max_lat and min_lon <= ln <= max_lon:
                if point_in_polygon(ln, lt, exterior):
                    mask[i] = True
    return mask

def basin_mean_gridded(data_field, lat_coord, lon_coord, geometry, lon_360=False):
    """Compute basin-averaged value from a gridded (lat x lon) field."""
    if data_field is None or geometry is None:
        return np.nan
    
    # Get lat/lon values
    if isinstance(lat_coord, xr.DataArray):
        lats = lat_coord.values
    else:
        lats = np.array(lat_coord)
    if isinstance(lon_coord, xr.DataArray):
        lons = lon_coord.values
    else:
        lons = np.array(lon_coord)
    
    # Handle data
    if isinstance(data_field, xr.DataArray):
        data = data_field.values
    else:
        data = np.array(data_field)
    
    # Create 2D grid if 1D
    if lats.ndim == 1 and lons.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lons, lats)
    else:
        lat_2d = lats
        lon_2d = lons
    
    # Convert 0-360 to -180-180 if needed
    if lon_360:
        lon_2d_adj = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
    else:
        lon_2d_adj = lon_2d.copy()
    
    mask = create_basin_mask_2d(lat_2d, lon_2d_adj, geometry)
    
    if mask.sum() == 0:
        log(f"  No grid points found in basin (lon_360={lon_360})")
        # Try the other convention
        if not lon_360:
            lon_2d_adj2 = np.where(lon_2d_adj < 0, lon_2d_adj + 360, lon_2d_adj)
            mask = create_basin_mask_2d(lat_2d, lon_2d_adj2, geometry)
            if mask.sum() > 0:
                log(f"  Found {mask.sum()} points with adjusted lon")
    
    if mask.sum() == 0:
        return np.nan
    
    # Area weighting with cos(lat)
    weights = np.cos(np.deg2rad(lat_2d))
    weights = weights * mask
    
    valid = ~np.isnan(data) & mask
    if valid.sum() == 0:
        return np.nan
    
    weighted_mean = np.nansum(data[valid] * weights[valid]) / np.nansum(weights[valid])
    return float(weighted_mean)

def basin_mean_unstructured(data_field, lats, lons, geometry):
    """Compute basin-averaged value from an unstructured grid."""
    if data_field is None or geometry is None:
        return np.nan
    
    if isinstance(data_field, xr.DataArray):
        data = data_field.values
    else:
        data = np.array(data_field)
    
    if isinstance(lats, xr.DataArray):
        lats = lats.values
    if isinstance(lons, xr.DataArray):
        lons = lons.values
    
    lats = np.array(lats).flatten()
    lons = np.array(lons).flatten()
    data = data.flatten()
    
    # Convert lon to -180..180
    lons_adj = np.where(lons > 180, lons - 360, lons)
    
    mask = create_basin_mask_1d(lats, lons_adj, geometry)
    
    if mask.sum() == 0:
        return np.nan
    
    weights = np.cos(np.deg2rad(lats[mask]))
    valid = ~np.isnan(data[mask])
    if valid.sum() == 0:
        return np.nan
    
    weighted_mean = np.nansum(data[mask][valid] * weights[valid]) / np.nansum(weights[valid])
    return float(weighted_mean)

# Determine if ELM data is structured (lat x lon) or unstructured
elm_is_structured = False
elm_is_unstructured = False

if model_precip_clim is not None:
    if model_precip_clim.ndim == 2:
        elm_is_structured = True
        log("ELM data appears structured (2D)")
    elif model_precip_clim.ndim == 1:
        elm_is_unstructured = True
        log("ELM data appears unstructured (1D)")
    log(f"Model P clim dims: {model_precip_clim.dims}, shape: {model_precip_clim.shape}")

# Check obs data structure
if obs_precip_clim is not None:
    log(f"Obs P clim dims: {obs_precip_clim.dims}, shape: {obs_precip_clim.shape}")
    obs_pr_lat_vals = obs_precip_clim.coords.get('lat', None)
    obs_pr_lon_vals = obs_precip_clim.coords.get('lon', None)
    if obs_pr_lon_vals is not None:
        lon_vals = obs_pr_lon_vals.values
        log(f"Obs PR lon range: {lon_vals.min():.1f} to {lon_vals.max():.1f}")

# Compute basin means
basin_results = {}

for gid, bname in BASINS.items():
    log(f"\n  Processing basin: {bname} (gauge {gid})")
    geom = get_basin_polygon(basins_geojson, gid) if basins_geojson else None
    if geom is None:
        log(f"  WARNING: No polygon found for {bname}")
        basin_results[gid] = {
            'name': bname,
            'model_P': np.nan, 'model_ET': np.nan, 'model_Q': np.nan,
            'obs_P': np.nan, 'obs_ET': np.nan, 'obs_Q': np.nan
        }
        continue
    
    # Model basin means
    if elm_is_unstructured and elm_lat is not None:
        m_P = basin_mean_unstructured(model_precip_clim, elm_lat, elm_lon, geom) if model_precip_clim is not None else np.nan
        m_ET = basin_mean_unstructured(model_et_clim, elm_lat, elm_lon, geom) if model_et_clim is not None else np.nan
        m_Q = basin_mean_unstructured(model_runoff_clim, elm_lat, elm_lon, geom) if model_runoff_clim is not None else np.nan
    elif elm_is_structured and elm_lat is not None:
        m_P = basin_mean_gridded(model_precip_clim, elm_lat, elm_lon, geom) if model_precip_clim is not None else np.nan
        m_ET = basin_mean_gridded(model_et_clim, elm_lat, elm_lon, geom) if model_et_clim is not None else np.nan
        m_Q = basin_mean_gridded(model_runoff_clim, elm_lat, elm_lon, geom) if model_runoff_clim is not None else np.nan
    else:
        m_P = m_ET = m_Q = np.nan
    
    log(f"    Model P={m_P:.3f}, ET={m_ET:.3f}, Q={m_Q:.3f} mm/day")
    
    # Obs basin means
    if obs_precip_clim is not None and obs_precip_clim.ndim == 2:
        lat_c = obs_precip_clim.coords.get('lat', None)
        lon_c = obs_precip_clim.coords.get('lon', None)
        lon_is_360 = lon_c is not None and float(lon_c.values.max()) > 180
        o_P = basin_mean_gridded(obs_precip_clim, lat_c, lon_c, geom, lon_360=lon_is_360)
    else:
        o_P = np.nan
    
    if obs_et_clim is not None and obs_et_clim.ndim == 2:
        lat_c = obs_et_clim.coords.get('lat', None)
        lon_c = obs_et_clim.coords.get('lon', None)
        lon_is_360 = lon_c is not None and float(lon_c.values.max()) > 180
        o_ET = basin_mean_gridded(obs_et_clim, lat_c, lon_c, geom, lon_360=lon_is_360)
    else:
        o_ET = np.nan
    
    if obs_runoff_clim is not None and obs_runoff_clim.ndim == 2:
        lat_c = obs_runoff_clim.coords.get('lat', None)
        lon_c = obs_runoff_clim.coords.get('lon', None)
        lon_is_360 = lon_c is not None and float(lon_c.values.max()) > 180
        o_Q = basin_mean_gridded(obs_runoff_clim, lat_c, lon_c, geom, lon_360=lon_is_360)
    else:
        o_Q = np.nan
    
    log(f"    Obs P={o_P:.3f}, ET={o_ET:.3f}, Q={o_Q:.3f} mm/day")
    
    basin_results[gid] = {
        'name': bname,
        'model_P': m_P, 'model_ET': m_ET, 'model_Q': m_Q,
        'obs_P': o_P, 'obs_ET': o_ET, 'obs_Q': o_Q
    }

###############################################################################
# Part 4: Streamflow FDC metrics
###############################################################################
log("\nPart 4: Streamflow FDC metrics...")

# Load gauge metadata
try:
    gauge_meta = pd.read_csv(GAUGE_META)
    log(f"Gauge metadata: {len(gauge_meta)} gauges")
    log(f"Columns: {list(gauge_meta.columns)}")
except Exception as e:
    log(f"Error loading gauge metadata: {e}")
    gauge_meta = None

# Load MOSART monthly data
mosart_files = []
for yr in YEARS:
    for mo in MONTHS:
        fname = os.path.join(E3SM_ROF, f"{CASE}.mosart.h0.{yr:04d}-{mo:02d}.nc")
        if os.path.exists(fname):
            mosart_files.append(fname)

# Also try daily files
mosart_daily_files = []
for yr in YEARS:
    for mo in MONTHS:
        for day in range(1, 32):
            fname = os.path.join(E3SM_ROF, f"{CASE}.mosart.h1.{yr:04d}-{mo:02d}-{day:02d}-00000.nc")
            if os.path.exists(fname):
                mosart_daily_files.append(fname)

log(f"Found {len(mosart_files)} MOSART monthly files, {len(mosart_daily_files)} daily files")

# Use daily if available, else monthly
use_daily_mosart = len(mosart_daily_files) > 0

try:
    if use_daily_mosart:
        ds_mos = xr.open_mfdataset(mosart_daily_files, combine='by_coords', decode_times=True, chunks={})
        log(f"MOSART daily dataset loaded. Vars: {list(ds_mos.data_vars)[:10]}")
    elif len(mosart_files) > 0:
        ds_mos = xr.open_mfdataset(mosart_files, combine='by_coords', decode_times=True, chunks={})
        log(f"MOSART monthly dataset loaded. Vars: {list(ds_mos.data_vars)[:10]}")
    else:
        ds_mos = None
        log("No MOSART files found")
    
    if ds_mos is not None:
        log(f"MOSART dims: {dict(ds_mos.dims)}")
        log(f"MOSART coords: {list(ds_mos.coords)}")
        
        # Get grid coordinates
        if 'lat' in ds_mos:
            mos_lat = ds_mos['lat'].values
            mos_lon = ds_mos['lon'].values
        else:
            for c in ds_mos.coords:
                log(f"  MOSART coord {c}: shape={ds_mos[c].shape}")
            mos_lat = None
            mos_lon = None
        
        # Get discharge variable
        discharge_var = 'RIVER_DISCHARGE_OVER_LAND_LIQ'
        if discharge_var not in ds_mos:
            # Try alternatives
            for v in ds_mos.data_vars:
                if 'DISCHARGE' in v.upper() or 'FLOW' in v.upper():
                    discharge_var = v
                    break
            log(f"Using discharge variable: {discharge_var}")
        
except Exception as e:
    log(f"Error loading MOSART data: {e}")
    import traceback; traceback.print_exc()
    ds_mos = None
    mos_lat = None
    mos_lon = None

def find_nearest_grid_cell(target_lat, target_lon, grid_lat, grid_lon):
    """Find index of nearest grid cell."""
    if grid_lat is None or grid_lon is None:
        return None
    
    grid_lat = np.array(grid_lat).flatten()
    grid_lon = np.array(grid_lon).flatten()
    
    # Convert to -180..180
    grid_lon_adj = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    
    dist = np.sqrt((grid_lat - target_lat)**2 + (grid_lon_adj - target_lon)**2)
    idx = np.argmin(dist)
    log(f"  Nearest cell: idx={idx}, lat={grid_lat[idx]:.2f}, lon={grid_lon_adj[idx]:.2f}, dist={dist[idx]:.3f} deg")
    return idx

def compute_fdc(q):
    """Compute flow duration curve (sorted descending, with exceedance probabilities)."""
    q_sorted = np.sort(q)[::-1]
    n = len(q_sorted)
    exceedance = np.arange(1, n+1) / (n+1) * 100
    return exceedance, q_sorted

def wasserstein_1d(u, v):
    """Compute 1D Wasserstein distance between two samples."""
    try:
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(u, v)
    except:
        # Manual computation
        u_sorted = np.sort(u)
        v_sorted = np.sort(v)
        # Interpolate to same length
        n = max(len(u_sorted), len(v_sorted))
        u_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
        v_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)
        return np.mean(np.abs(u_interp - v_interp))

# Process each basin
for gid, bname in BASINS.items():
    log(f"\n  Processing streamflow for {bname} (gauge {gid})...")
    
    # Initialize defaults
    basin_results[gid]['obs_streamflow_mean'] = np.nan
    basin_results[gid]['model_streamflow_mean'] = np.nan
    basin_results[gid]['streamflow_volume_bias'] = np.nan
    basin_results[gid]['wasserstein_dist'] = np.nan
    
    # Get gauge info
    if gauge_meta is not None:
        gauge_row = gauge_meta[gauge_meta['gauge_id'].astype(str) == str(gid)]
        if len(gauge_row) == 0:
            log(f"  Gauge {gid} not found in metadata")
            continue
        gauge_lat = float(gauge_row['lat'].iloc[0])
        gauge_lon = float(gauge_row['lon'].iloc[0])
        log(f"  Gauge location: lat={gauge_lat}, lon={gauge_lon}")
    else:
        continue
    
    # Load observed discharge
    obs_flow_file = os.path.join(OBS_FLOW_DIR, f"{gid}.csv")
    if not os.path.exists(obs_flow_file):
        log(f"  Obs flow file not found: {obs_flow_file}")
        continue
    
    try:
        obs_flow = pd.read_csv(obs_flow_file, parse_dates=['date'])
        obs_flow = obs_flow[(obs_flow['date'].dt.year >= 1985) & (obs_flow['date'].dt.year <= 1989)]
        obs_q = obs_flow['discharge_m3s'].dropna().values
        log(f"  Obs flow: {len(obs_q)} daily values, mean={np.mean(obs_q):.1f} m3/s")
        basin_results[gid]['obs_streamflow_mean'] = float(np.mean(obs_q))
    except Exception as e:
        log(f"  Error loading obs flow: {e}")
        obs_q = None
        continue
    
    # Extract simulated discharge at nearest grid cell
    if ds_mos is not None and mos_lat is not None:
        idx = find_nearest_grid_cell(gauge_lat, gauge_lon, mos_lat, mos_lon)
        if idx is not None:
            try:
                # Determine dimension structure
                if 'gridcell' in ds_mos.dims:
                    sim_q_ts = ds_mos[discharge_var].isel(gridcell=idx).values
                elif len(ds_mos[discharge_var].dims) == 1:
                    # Only time dim? Unlikely...
                    sim_q_ts = ds_mos[discharge_var].values
                elif len(ds_mos[discharge_var].dims) == 2:
                    # (time, gridcell) or similar
                    dim_names = ds_mos[discharge_var].dims
                    non_time_dim = [d for d in dim_names if d != 'time'][0]
                    sim_q_ts = ds_mos[discharge_var].isel({non_time_dim: idx}).values
                elif len(ds_mos[discharge_var].dims) == 3:
                    # (time, lat, lon)
                    # Need to find i, j indices
                    # This requires 2D lat/lon
                    log("  3D discharge array - need 2D indexing")
                    sim_q_ts = None
                else:
                    sim_q_ts = None
                
                if sim_q_ts is not None:
                    sim_q_ts = sim_q_ts.flatten()
                    sim_q_ts = sim_q_ts[~np.isnan(sim_q_ts)]
                    log(f"  Sim flow: {len(sim_q_ts)} values, mean={np.mean(sim_q_ts):.1f} m3/s")
                    basin_results[gid]['model_streamflow_mean'] = float(np.mean(sim_q_ts))
                    
                    # FDC metrics
                    if obs_q is not None and len(obs_q) > 0 and len(sim_q_ts) > 0:
                        # Volume bias
                        vol_bias = (np.mean(sim_q_ts) - np.mean(obs_q)) / np.mean(obs_q) * 100
                        basin_results[gid]['streamflow_volume_bias'] = float(vol_bias)
                        
                        # Wasserstein distance
                        w_dist = wasserstein_1d(obs_q, sim_q_ts)
                        basin_results[gid]['wasserstein_dist'] = float(w_dist)
                        log(f"  Volume bias: {vol_bias:.1f}%, Wasserstein: {w_dist:.1f}")
            except Exception as e:
                log(f"  Error extracting sim discharge: {e}")
                import traceback; traceback.print_exc()

if ds_mos is not None:
    ds_mos.close()

###############################################################################
# Part 5: Combine and visualise
###############################################################################
log("\nPart 5: Combining metrics and creating visualizations...")

# Build summary table
summary_rows = []
for gid, bname in BASINS.items():
    r = basin_results[gid]
    
    # Biases in mm/day
    P_bias = r['model_P'] - r['obs_P'] if not (np.isnan(r['model_P']) or np.isnan(r['obs_P'])) else np.nan
    ET_bias = r['model_ET'] - r['obs_ET'] if not (np.isnan(r['model_ET']) or np.isnan(r['obs_ET'])) else np.nan
    Q_bias = r['model_Q'] - r['obs_Q'] if not (np.isnan(r['model_Q']) or np.isnan(r['obs_Q'])) else np.nan
    
    # Relative biases (%)
    P_bias_pct = P_bias / r['obs_P'] * 100 if not np.isnan(P_bias) and r['obs_P'] != 0 else np.nan
    ET_bias_pct = ET_bias / r['obs_ET'] * 100 if not np.isnan(ET_bias) and r['obs_ET'] != 0 else np.nan
    Q_bias_pct = Q_bias / r['obs_Q'] * 100 if not np.isnan(Q_bias) and r['obs_Q'] != 0 else np.nan
    
    # Water balance residual: P - ET - Q (mm/day)
    model_wb = r['model_P'] - r['model_ET'] - r['model_Q']
    obs_wb = r['obs_P'] - r['obs_ET'] - r['obs_Q']
    
    summary_rows.append({
        'gauge_id': gid,
        'basin_name': bname,
        'model_P_mm_day': r['model_P'],
        'obs_P_mm_day': r['obs_P'],
        'P_bias_mm_day': P_bias,
        'P_bias_pct': P_bias_pct,
        'model_ET_mm_day': r['model_ET'],
        'obs_ET_mm_day': r['obs_ET'],
        'ET_bias_mm_day': ET_bias,
        'ET_bias_pct': ET_bias_pct,
        'model_Q_mm_day': r['model_Q'],
        'obs_Q_mm_day': r['obs_Q'],
        'Q_bias_mm_day': Q_bias,
        'Q_bias_pct': Q_bias_pct,
        'model_streamflow_m3s': r.get('model_streamflow_mean', np.nan),
        'obs_streamflow_m3s': r.get('obs_streamflow_mean', np.nan),
        'streamflow_volume_bias_pct': r.get('streamflow_volume_bias', np.nan),
        'wasserstein_distance': r.get('wasserstein_dist', np.nan),
        'model_water_balance_residual': model_wb,
        'obs_water_balance_residual': obs_wb,
    })

df_summary = pd.DataFrame(summary_rows)

# Save summary table
try:
    csv_path = os.path.join(OUTDIR, "basin_diagnostic_summary.csv")
    df_summary.to_csv(csv_path, index=False, float_format='%.4f')
    log(f"Summary table saved to {csv_path}")
except Exception as e:
    log(f"Error saving CSV: {e}")

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(df_summary.to_string(index=False))
print("="*80)

# ---- Bar chart: Model vs Obs for P, ET, Q ----
log("Creating bar chart...")
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    basin_names = [BASINS[gid] for gid in BASINS]
    n_basins = len(basin_names)
    x = np.arange(n_basins)
    width = 0.35
    
    variables = [
        ('P', 'model_P_mm_day', 'obs_P_mm_day', 'Precipitation (mm/day)'),
        ('ET', 'model_ET_mm_day', 'obs_ET_mm_day', 'Evapotranspiration (mm/day)'),
        ('Q', 'model_Q_mm_day', 'obs_Q_mm_day', 'Runoff (mm/day)')
    ]
    
    colors_model = ['#2166ac', '#2166ac', '#2166ac']
    colors_obs = ['#b2182b', '#b2182b', '#b2182b']
    
    for idx, (vname, model_col, obs_col, ylabel) in enumerate(variables):
        ax = axes[idx]
        model_vals = df_summary[model_col].values
        obs_vals = df_summary[obs_col].values
        
        bars1 = ax.bar(x - width/2, model_vals, width, label='E3SM', color='steelblue', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, obs_vals, width, label='Observations', color='coral', edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{vname}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basin_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add bias annotation
        for i in range(n_basins):
            if not np.isnan(model_vals[i]) and not np.isnan(obs_vals[i]) and obs_vals[i] != 0:
                bias_pct = (model_vals[i] - obs_vals[i]) / obs_vals[i] * 100
                ymax = max(model_vals[i], obs_vals[i])
                ax.annotate(f'{bias_pct:+.0f}%', xy=(x[i], ymax),
                           fontsize=8, ha='center', va='bottom', color='darkgreen', fontweight='bold')
    
    plt.suptitle('E3SM vs Observations: Basin-Averaged Water Cycle Components (1985-1989)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, "basin_water_cycle_barchart.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f"Bar chart saved to {fig_path}")
except Exception as e:
    log(f"Error creating bar chart: {e}")
    import traceback; traceback.print_exc()

# ---- Radar chart: Multi-variable diagnostic per basin ----
log("Creating radar chart...")
try:
    # Metrics for radar: P bias%, ET bias%, Q bias%, streamflow bias%, water balance residual, Wasserstein
    categories = ['P Bias (%)', 'ET Bias (%)', 'Runoff Bias (%)', 
                  'Streamflow Bias (%)', 'WB Residual\n(mm/day)', 'Wasserstein\nDist (m³/s)']
    N = len(categories)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_basins))
    
    for i, (gid, bname) in enumerate(BASINS.items()):
        row = df_summary[df_summary['gauge_id'] == gid].iloc[0]
        
        values = [
            row['P_bias_pct'] if not np.isnan(row['P_bias_pct']) else 0,
            row['ET_bias_pct'] if not np.isnan(row['ET_bias_pct']) else 0,
            row['Q_bias_pct'] if not np.isnan(row['Q_bias_pct']) else 0,
            row['streamflow_volume_bias_pct'] if not np.isnan(row['streamflow_volume_bias_pct']) else 0,
            row['model_water_balance_residual'] if not np.isnan(row['model_water_balance_residual']) else 0,
            row['wasserstein_distance'] if not np.isnan(row['wasserstein_distance']) else 0,
        ]
        
        # Normalize to make radar chart readable
        # Use absolute values for the radar, and track signs separately
        abs_values = [abs(v) for v in values]
        
        # Close the polygon
        abs_values_closed = abs_values + abs_values[:1]
        
        ax.plot(angles, abs_values_closed, 'o-', linewidth=2, label=bname, color=colors[i])
        ax.fill(angles, abs_values_closed, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title('Multi-Variable Basin Diagnostic\n(Absolute Bias Magnitudes)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    fig_path = os.path.join(OUTDIR, "basin_radar_diagnostic.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f"Radar chart saved to {fig_path}")
except Exception as e:
    log(f"Error creating radar chart: {e}")
    import traceback; traceback.print_exc()

# ---- Alternative: Normalized radar chart ----
log("Creating normalized radar chart...")
try:
    categories_norm = ['|P Bias| (%)', '|ET Bias| (%)', '|Runoff Bias| (%)', 
                       '|Streamflow Bias| (%)', '|WB Residual|\n(mm/day)', 'Wasserstein\n(norm)']
    N = len(categories_norm)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # Collect raw values per metric for normalization
    raw_metrics = np.zeros((n_basins, N))
    for i, (gid, bname) in enumerate(BASINS.items()):
        row = df_summary[df_summary['gauge_id'] == gid].iloc[0]
        raw_metrics[i, 0] = abs(row['P_bias_pct']) if not np.isnan(row['P_bias_pct']) else 0
        raw_metrics[i, 1] = abs(row['ET_bias_pct']) if not np.isnan(row['ET_bias_pct']) else 0
        raw_metrics[i, 2] = abs(row['Q_bias_pct']) if not np.isnan(row['Q_bias_pct']) else 0
        raw_metrics[i, 3] = abs(row['streamflow_volume_bias_pct']) if not np.isnan(row['streamflow_volume_bias_pct']) else 0
        raw_metrics[i, 4] = abs(row['model_water_balance_residual']) if not np.isnan(row['model_water_balance_residual']) else 0
        raw_metrics[i, 5] = row['wasserstein_distance'] if not np.isnan(row['wasserstein_distance']) else 0
    
    # Normalize each metric to 0-1 range
    norm_metrics = np.zeros_like(raw_metrics)
    for j in range(N):
        col_max = raw_metrics[:, j].max()
        if col_max > 0:
            norm_metrics[:, j] = raw_metrics[:, j] / col_max
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, n_basins))
    
    for i, (gid, bname) in enumerate(BASINS.items()):
        values = norm_metrics[i, :].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=bname, color=colors[i], markersize=6)
        ax.fill(angles, values, alpha=0.08, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_norm, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    ax.set_title('Normalized Multi-Variable Basin Diagnostic\n(0 = best, 1 = worst among basins)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.grid(True)
    
    fig_path = os.path.join(OUTDIR, "basin_radar_diagnostic_normalized.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f"Normalized radar chart saved to {fig_path}")
except Exception as e:
    log(f"Error creating normalized radar chart: {e}")
    import traceback; traceback.print_exc()

# ---- Heatmap of biases ----
log("Creating bias heatmap...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bias_cols = ['P_bias_pct', 'ET_bias_pct', 'Q_bias_pct', 'streamflow_volume_bias_pct']
    bias_labels = ['P Bias (%)', 'ET Bias (%)', 'Runoff Bias (%)', 'Streamflow Bias (%)']
    
    bias_data = df_summary[bias_cols].values
    basin_labels = df_summary['basin_name'].values
    
    # Replace NaN with 0 for display
    bias_display = np.where(np.isnan(bias_data), 0, bias_data)
    
    vmax = np.nanmax(np.abs(bias_display)) if np.nanmax(np.abs(bias_display)) > 0 else 100
    im = ax.imshow(bias_display, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(bias_labels)))
    ax.set_xticklabels(bias_labels, fontsize=11)
    ax.set_yticks(range(len(basin_labels)))
    ax.set_yticklabels(basin_labels, fontsize=11)
    
    # Annotate cells
    for i in range(len(basin_labels)):
        for j in range(len(bias_labels)):
            val = bias_data[i, j]
            text = f'{val:.1f}%' if not np.isnan(val) else 'N/A'
            color = 'white' if abs(bias_display[i, j]) > vmax * 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=10, color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Bias (%)', shrink=0.8)
    ax.set_title('E3SM Water Cycle Bias by Basin (1985-1989)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(OUTDIR, "basin_bias_heatmap.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f"Bias heatmap saved to {fig_path}")
except Exception as e:
    log(f"Error creating heatmap: {e}")
    import traceback; traceback.print_exc()

log(f"\nAll outputs saved to: {OUTDIR}")
log("Script completed successfully.")