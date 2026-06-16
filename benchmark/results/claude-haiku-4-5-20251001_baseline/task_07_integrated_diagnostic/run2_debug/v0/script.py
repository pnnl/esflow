#!/usr/bin/env python3
"""
Per-basin integrated water cycle evaluation for E3SM over 1985-1989.
Diagnoses model biases across six major river basins.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial.distance import wasserstein_distance
from scipy.stats import gaussian_kde
import json
import urllib.request
import tempfile
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_07_integrated_diagnostic/run2_output"
DATA_DIR = "./data/sample"
E3SM_DIR = os.path.join(DATA_DIR, "e3sm")
OBS_DIR = os.path.join(DATA_DIR, "obs")
CASE_NAME = "sample.v3.LR.historical"

# Basin metadata: gauge_id -> (name, lat, lon)
BASINS = {
    3629000: ("Amazon", -3.0, -60.0),
    4121801: ("Missouri", 38.5, -92.0),
    4115200: ("Columbia", 46.0, -123.5),
    6742900: ("Danube", 45.0, 25.0),
    2969100: ("Mekong", 10.5, 105.0),
    1159100: ("Orange", -28.5, 20.0),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(msg):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def download_ilamb_data(variable, dataset, filename):
    """Download ILAMB observation data."""
    url = f"https://www.ilamb.org/ILAMB-Data/DATA/{variable}/{dataset}/{filename}"
    log_message(f"Downloading {url}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            return tmp.name
    except Exception as e:
        log_message(f"Failed to download {url}: {e}")
        return None

def load_elm_data(start_year=1985, end_year=1989):
    """Load ELM monthly data and compute climatological means."""
    log_message("Loading ELM data...")
    
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            fname = os.path.join(E3SM_DIR, "lnd", 
                f"{CASE_NAME}.elm.h0.{year:04d}-{month:02d}.nc")
            if os.path.exists(fname):
                files.append(fname)
    
    if not files:
        log_message(f"No ELM files found in {os.path.join(E3SM_DIR, 'lnd')}")
        return None
    
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        
        # Compute precipitation (kg/m2/s -> mm/day)
        precip = (ds['RAIN'] + ds['SNOW']) * 86400 / 1000
        
        # Compute evapotranspiration (mm/s -> mm/day)
        et = (ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']) * 86400
        
        # Total runoff (mm/s -> mm/day)
        runoff = ds['QRUNOFF'] * 86400
        
        # Compute climatological means
        precip_mean = precip.mean(dim='time')
        et_mean = et.mean(dim='time')
        runoff_mean = runoff.mean(dim='time')
        
        log_message(f"ELM data loaded: {len(files)} files")
        
        return {
            'precip': precip_mean,
            'et': et_mean,
            'runoff': runoff_mean,
            'ds': ds
        }
    except Exception as e:
        log_message(f"Error loading ELM data: {e}")
        return None

def load_obs_precip():
    """Load GPCC precipitation data."""
    log_message("Loading GPCC precipitation...")
    try:
        tmp_file = download_ilamb_data("pr", "GPCCv2018", "pr.nc")
        if tmp_file is None:
            return None
        
        ds = xr.open_dataset(tmp_file)
        # Convert from mm/day to mm/day (already in correct units)
        precip = ds['pr'].mean(dim='time')
        
        log_message("GPCC precipitation loaded")
        return precip
    except Exception as e:
        log_message(f"Error loading GPCC data: {e}")
        return None

def load_obs_et():
    """Load MODIS ET data."""
    log_message("Loading MODIS ET...")
    try:
        tmp_file = download_ilamb_data("evspsbl", "MODIS", "et_0.5x0.5.nc")
        if tmp_file is None:
            return None
        
        ds = xr.open_dataset(tmp_file)
        # Extract 'et' variable and convert to mm/day
        et = ds['et'].mean(dim='time')
        
        log_message("MODIS ET loaded")
        return et
    except Exception as e:
        log_message(f"Error loading MODIS ET: {e}")
        return None

def load_obs_runoff():
    """Load LORA runoff data."""
    log_message("Loading LORA runoff...")
    try:
        tmp_file = download_ilamb_data("mrro", "LORA", "LORA.nc")
        if tmp_file is None:
            return None
        
        ds = xr.open_dataset(tmp_file)
        runoff = ds['mrro'].mean(dim='time')
        
        log_message("LORA runoff loaded")
        return runoff
    except Exception as e:
        log_message(f"Error loading LORA runoff: {e}")
        return None

def load_basin_polygons():
    """Load basin polygons from GeoJSON."""
    log_message("Loading basin polygons...")
    try:
        geojson_file = os.path.join(OBS_DIR, "basin_polygons.geojson")
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        
        basins = {}
        for feature in geojson_data['features']:
            gauge_id = feature['properties'].get('grdc_no')
            if gauge_id:
                basins[gauge_id] = feature['geometry']
        
        log_message(f"Loaded {len(basins)} basin polygons")
        return basins
    except Exception as e:
        log_message(f"Error loading basin polygons: {e}")
        return {}

def clip_to_basin(data, basin_geom):
    """Clip gridded data to basin polygon and compute mean."""
    try:
        if data is None or basin_geom is None:
            return np.nan
        
        # Extract coordinates
        if 'lat' in data.coords and 'lon' in data.coords:
            lats = data.lat.values
            lons = data.lon.values
        else:
            return np.nan
        
        # Create 2D coordinate grids
        if len(lats.shape) == 1 and len(lons.shape) == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lat_grid, lon_grid = lats, lons
        
        # Create mask for points inside basin
        coords = list(basin_geom['coordinates'][0])
        mask = np.zeros(lat_grid.shape, dtype=bool)
        
        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                point = (lon_grid[i, j], lat_grid[i, j])
                mask[i, j] = point_in_polygon(point, coords)
        
        # Compute basin mean
        if mask.sum() > 0:
            return float(data.values[mask].mean())
        else:
            return np.nan
    except Exception as e:
        log_message(f"Error clipping to basin: {e}")
        return np.nan

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def load_gauge_metadata():
    """Load gauge metadata."""
    log_message("Loading gauge metadata...")
    try:
        metadata_file = os.path.join(OBS_DIR, "gauge_metadata.csv")
        df = pd.read_csv(metadata_file)
        return df
    except Exception as e:
        log_message(f"Error loading gauge metadata: {e}")
        return pd.DataFrame()

def load_streamflow_data(gauge_id, start_year=1985, end_year=1989):
    """Load observed streamflow for a gauge."""
    try:
        csv_file = os.path.join(OBS_DIR, "streamflow", f"{gauge_id}.csv")
        if not os.path.exists(csv_file):
            return None
        
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to period
        mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
        df = df[mask]
        
        return df
    except Exception as e:
        log_message(f"Error loading streamflow for gauge {gauge_id}: {e}")
        return None

def load_mosart_data(start_year=1985, end_year=1989):
    """Load MOSART monthly discharge data."""
    log_message("Loading MOSART data...")
    
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            fname = os.path.join(E3SM_DIR, "rof",
                f"{CASE_NAME}.mosart.h0.{year:04d}-{month:02d}.nc")
            if os.path.exists(fname):
                files.append(fname)
    
    if not files:
        log_message(f"No MOSART files found in {os.path.join(E3SM_DIR, 'rof')}")
        return None
    
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        discharge = ds['RIVER_DISCHARGE_OVER_LAND_LIQ']
        
        log_message(f"MOSART data loaded: {len(files)} files")
        return discharge
    except Exception as e:
        log_message(f"Error loading MOSART data: {e}")
        return None

def find_nearest_mosart_cell(lat, lon, mosart_ds):
    """Find nearest MOSART grid cell to a location."""
    try:
        if 'lat' not in mosart_ds.coords or 'lon' not in mosart_ds.coords:
            return None
        
        mosart_lats = mosart_ds.lat.values
        mosart_lons = mosart_ds.lon.values
        
        # Handle 2D coordinates
        if len(mosart_lats.shape) == 2:
            dist = np.sqrt((mosart_lats - lat)**2 + (mosart_lons - lon)**2)
            idx = np.unravel_index(np.argmin(dist), dist.shape)
        else:
            lat_idx = np.argmin(np.abs(mosart_lats - lat))
            lon_idx = np.argmin(np.abs(mosart_lons - lon))
            idx = (lat_idx, lon_idx)
        
        return idx
    except Exception as e:
        log_message(f"Error finding nearest MOSART cell: {e}")
        return None

def compute_fdc_metrics(obs_discharge, sim_discharge):
    """Compute flow duration curve metrics."""
    try:
        # Remove NaN values
        obs_clean = obs_discharge[~np.isnan(obs_discharge)]
        sim_clean = sim_discharge[~np.isnan(sim_discharge)]
        
        if len(obs_clean) == 0 or len(sim_clean) == 0:
            return {'volume_bias': np.nan, 'wasserstein': np.nan}
        
        # Volume bias (%)
        obs_mean = np.mean(obs_clean)
        sim_mean = np.mean(sim_clean)
        volume_bias = 100 * (sim_mean - obs_mean) / obs_mean if obs_mean != 0 else np.nan
        
        # Wasserstein distance (normalized)
        obs_sorted = np.sort(obs_clean)
        sim_sorted = np.sort(sim_clean)
        
        # Interpolate to same length
        if len(obs_sorted) != len(sim_sorted):
            common_len = min(len(obs_sorted), len(sim_sorted))
            obs_sorted = np.interp(np.linspace(0, 1, common_len),
                                   np.linspace(0, 1, len(obs_sorted)), obs_sorted)
            sim_sorted = np.interp(np.linspace(0, 1, common_len),
                                   np.linspace(0, 1, len(sim_sorted)), sim_sorted)
        
        wasserstein = wasserstein_distance(obs_sorted, sim_sorted)
        wasserstein_norm = wasserstein / obs_mean if obs_mean != 0 else np.nan
        
        return {
            'volume_bias': volume_bias,
            'wasserstein': wasserstein_norm
        }
    except Exception as e:
        log_message(f"Error computing FDC metrics: {e}")
        return {'volume_bias': np.nan, 'wasserstein': np.nan}

def main():
    """Main analysis workflow."""
    log_message("Starting per-basin integrated water cycle evaluation")
    
    # Part 1: Load model fields
    elm_data = load_elm_data()
    if elm_data is None:
        log_message("Failed to load ELM data")
        return
    
    # Part 2: Load observation fields
    obs_precip = load_obs_precip()
    obs_et = load_obs_et()
    obs_runoff = load_obs_runoff()
    
    # Load basin polygons
    basin_polygons = load_basin_polygons()
    
    # Part 3: Clip to basin means
    log_message("Computing basin-averaged metrics...")
    
    basin_results = []
    
    for gauge_id, (basin_name, lat, lon) in BASINS.items():
        log_message(f"Processing {basin_name} (gauge {gauge_id})")
        
        # Get basin polygon
        basin_geom = basin_polygons.get(gauge_id)
        
        # Clip model fields
        model_precip = clip_to_basin(elm_data['precip'], basin_geom) if basin_geom else np.nan
        model_et = clip_to_basin(elm_data['et'], basin_geom) if basin_geom else np.nan
        model_runoff = clip_to_basin(elm_data['runoff'], basin_geom) if basin_geom else np.nan
        
        # Clip observation fields
        obs_p = clip_to_basin(obs_precip, basin_geom) if basin_geom and obs_precip is not None else np.nan
        obs_e = clip_to_basin(obs_et, basin_geom) if basin_geom and obs_et is not None else np.nan
        obs_q = clip_to_basin(obs_runoff, basin_geom) if basin_geom and obs_runoff is not None else np.nan
        
        # Compute biases
        p_bias = 100 * (model_precip - obs_p) / obs_p if not np.isnan(obs_p) and obs_p != 0 else np.nan
        et_bias = 100 * (model_et - obs_e) / obs_e if not np.isnan(obs_e) and obs_e != 0 else np.nan
        q_bias = 100 * (model_runoff - obs_q) / obs_q if not np.isnan(obs_q) and obs_q != 0 else np.nan
        
        # Water balance residual
        water_balance = model_precip - model_et - model_runoff if not np.isnan(model_precip) else np.nan
        
        result = {
            'gauge_id': gauge_id,
            'basin_name': basin_name,
            'model_precip': model_precip,
            'obs_precip': obs_p,
            'p_bias': p_bias,
            'model_et': model_et,
            'obs_et': obs_e,
            'et_bias': et_bias,
            'model_runoff': model_runoff,
            'obs_runoff': obs_q,
            'q_bias': q_bias,
            'water_balance': water_balance,
        }
        
        basin_results.append(result)
    
    # Part 4: Streamflow FDC metrics
    log_message("Computing streamflow metrics...")
    
    gauge_metadata = load_gauge_metadata()
    mosart_discharge = load_mosart_data()
    
    for i, result in enumerate(basin_results):
        gauge_id = result['gauge_id']
        
        # Load observed streamflow
        obs_flow = load_streamflow_data(gauge_id)
        
        if obs_flow is not None and mosart_discharge is not None:
            try:
                # Find nearest MOSART cell
                idx = find_nearest_mosart_cell(BASINS[gauge_id][1], BASINS[gauge_id][2], mosart_discharge)
                
                if idx is not None:
                    # Extract simulated discharge
                    sim_flow = mosart_discharge.isel(x=idx[1], y=idx[0]).values
                    
                    # Compute metrics
                    metrics = compute_fdc_metrics(obs_flow['discharge_m3s'].values, sim_flow)
                    result['streamflow_bias'] = metrics['volume_bias']
                    result['wasserstein'] = metrics['wasserstein']
                else:
                    result['streamflow_bias'] = np.nan
                    result['wasserstein'] = np.nan
            except Exception as e:
                log_message(f"Error processing streamflow for {gauge_id}: {e}")
                result['streamflow_bias'] = np.nan
                result['wasserstein'] = np.nan
        else:
            result['streamflow_bias'] = np.nan
            result['wasserstein'] = np.nan
    
    # Part 5: Create summary table
    log_message("Creating summary table...")
    
    summary_df = pd.DataFrame(basin_results)
    summary_file = os.path.join(OUTPUT_DIR, "basin_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    log_message(f"Summary table saved to {summary_file}")
    
    # Create visualizations
    try:
        create_bar_chart(basin_results)
    except Exception as e:
        log_message(f"Error creating bar chart: {e}")
    
    try:
        create_radar_chart(basin_results)
    except Exception as e:
        log_message(f"Error creating radar chart: {e}")
    
    log_message("Analysis complete!")

def create_bar_chart(basin_results):
    """Create model vs observation bar chart."""
    log_message("Creating bar chart...")
    
    basin_names = [r['basin_name'] for r in basin_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precipitation
    model_p = [r['model_precip'] for r in basin_results]
    obs_p = [r['obs_precip'] for r in basin_results]
    
    x = np.arange(len(basin_names))
    width = 0.35
    
    axes[0].bar(x - width/2, model_p, width, label='Model', alpha=0.8)
    axes[0].bar(x + width/2, obs_p, width, label='Obs', alpha=0.8)
    axes[0].set_ylabel('Precipitation (mm/day)')
    axes[0].set_title('Precipitation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Evapotranspiration
    model_et = [r['model_et'] for r in basin_results]
    obs_et = [r['obs_et'] for r in basin_results]
    
    axes[1].bar(x - width/2, model_et, width, label='Model', alpha=0.8)
    axes[1].bar(x + width/2, obs_et, width, label='Obs', alpha=0.8)
    axes[1].set_ylabel('ET (mm/day)')
    axes[1].set_title('Evapotranspiration')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Runoff
    model_q = [r['model_runoff'] for r in basin_results]
    obs_q = [r['obs_runoff'] for r in basin_results]
    
    axes[2].bar(x - width/2, model_q, width, label='Model', alpha=0.8)
    axes[2].bar(x + width/2, obs_q, width, label='Obs', alpha=0.8)
    axes[2].set_ylabel('Runoff (mm/day)')
    axes[2].set_title('Total Runoff')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "model_vs_obs_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    log_message(f"Bar chart saved to {output_file}")

def create_radar_chart(basin_results):
    """Create radar chart for multi-variable diagnostics."""
    log_message("Creating radar chart...")
    
    # Normalize biases to 0-1 scale (0 = perfect, 1 = large error)
    categories = ['P Bias', 'ET Bias', 'Q Bias', 'Streamflow Bias']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for idx, result in enumerate(basin_results):
        ax = axes[idx]
        
        # Get biases
        p_bias = abs(result['p_bias']) if not np.isnan(result['p_bias']) else 0
        et_bias = abs(result['et_bias']) if not np.isnan(result['et_bias']) else 0
        q_bias = abs(result['q_bias']) if not np.isnan(result['q_bias']) else 0
        sf_bias = abs(result['streamflow_bias']) if not np.isnan(result['streamflow_bias']) else 0
        
        # Normalize to 0-100 scale
        values = [p_bias, et_bias, q_bias, sf_bias]
        
        # Close the plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['basin_name'])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(result['basin_name'], pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "radar_diagnostics.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    log_message(f"Radar chart saved to {output_file}")

if __name__ == '__main__':
    main()