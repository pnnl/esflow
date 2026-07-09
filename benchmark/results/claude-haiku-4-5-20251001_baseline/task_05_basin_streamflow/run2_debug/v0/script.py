#!/usr/bin/env python
"""
Basin-scale streamflow analysis for E3SM MOSART output.
Compares simulated discharge with observations for six major river basins (1985-1989).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import json
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

# Configuration
E3SM_DIR = "./data/sample/e3sm/"
OBS_DIR = "./data/sample/obs/"
CASE_NAME = "sample.v3.LR.historical"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_05_basin_streamflow/run2_output"
START_YEAR = 1985
END_YEAR = 1989

# Basin information: gauge_id, river_name
BASINS = {
    3629000: "Amazon",
    4121801: "Missouri",
    4115200: "Columbia",
    6742900: "Danube",
    2969100: "Mekong",
    1159100: "Orange"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_gauge_metadata():
    """Load gauge metadata from CSV."""
    try:
        metadata_file = os.path.join(OBS_DIR, "gauge_metadata.csv")
        df = pd.read_csv(metadata_file)
        return df
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        return None

def load_basin_polygons():
    """Load basin polygons from GeoJSON."""
    try:
        geojson_file = os.path.join(OBS_DIR, "basin_polygons.geojson")
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        return geojson_data
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        return None

def load_observation_data(gauge_id, start_year, end_year):
    """Load observation streamflow data for a gauge."""
    try:
        obs_file = os.path.join(OBS_DIR, "streamflow", f"{gauge_id}.csv")
        df = pd.read_csv(obs_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to date range
        mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
        df = df[mask].reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"Error loading observation data for gauge {gauge_id}: {e}")
        return None

def load_mosart_data(start_year, end_year):
    """Load MOSART monthly discharge data."""
    try:
        mosart_files = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                file_path = os.path.join(
                    E3SM_DIR, "rof",
                    f"{CASE_NAME}.mosart.h0.{year:04d}-{month:02d}.nc"
                )
                if os.path.exists(file_path):
                    mosart_files.append(file_path)
        
        if not mosart_files:
            print("No MOSART files found")
            return None
        
        # Load and concatenate
        ds_list = [xr.open_dataset(f) for f in sorted(mosart_files)]
        ds = xr.concat(ds_list, dim='time')
        
        return ds
    except Exception as e:
        print(f"Error loading MOSART data: {e}")
        return None

def find_nearest_grid_cell(lat, lon, grid_lats, grid_lons):
    """Find nearest grid cell to a given lat/lon point."""
    try:
        # Create KDTree for efficient nearest neighbor search
        points = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])
        tree = cKDTree(points)
        
        # Query nearest neighbor
        dist, idx = tree.query([lat, lon])
        
        # Convert flat index to 2D indices
        shape = grid_lats.shape
        row, col = np.unravel_index(idx, shape)
        
        return row, col
    except Exception as e:
        print(f"Error finding nearest grid cell: {e}")
        return None, None

def extract_discharge_at_gauge(ds, gauge_lat, gauge_lon):
    """Extract discharge time series at nearest grid cell to gauge."""
    try:
        # Get grid coordinates
        grid_lats = ds['lat'].values
        grid_lons = ds['lon'].values
        
        # Handle longitude convention (0-360 vs -180-180)
        if grid_lons.max() > 180:
            grid_lons = ((grid_lons + 180) % 360) - 180
        
        gauge_lon_adj = ((gauge_lon + 180) % 360) - 180
        
        # Find nearest cell
        row, col = find_nearest_grid_cell(gauge_lat, gauge_lon_adj, grid_lats, grid_lons)
        
        if row is None:
            return None
        
        # Extract discharge
        discharge = ds['RIVER_DISCHARGE_OVER_LAND_LIQ'].isel(lat=row, lon=col).values
        
        return discharge
    except Exception as e:
        print(f"Error extracting discharge: {e}")
        return None

def compute_metrics(obs, sim):
    """Compute validation metrics: RMSE, NSE, KGE, PBIAS."""
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(obs) | np.isnan(sim))
        obs_clean = obs[valid_mask]
        sim_clean = sim[valid_mask]
        
        if len(obs_clean) == 0:
            return None
        
        # RMSE
        rmse = np.sqrt(np.mean((sim_clean - obs_clean) ** 2))
        
        # NSE (Nash-Sutcliffe Efficiency)
        obs_mean = np.mean(obs_clean)
        nse = 1 - (np.sum((sim_clean - obs_clean) ** 2) / 
                   np.sum((obs_clean - obs_mean) ** 2))
        
        # KGE (Kling-Gupta Efficiency)
        sim_mean = np.mean(sim_clean)
        sim_std = np.std(sim_clean)
        obs_std = np.std(obs_clean)
        
        r = np.corrcoef(obs_clean, sim_clean)[0, 1]
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean
        
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        # PBIAS (Percent Bias)
        pbias = 100 * np.sum(sim_clean - obs_clean) / np.sum(obs_clean)
        
        return {
            'RMSE': rmse,
            'NSE': nse,
            'KGE': kge,
            'PBIAS': pbias,
            'n_samples': len(obs_clean)
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None

def get_basin_polygon(basin_geojson, gauge_id):
    """Extract polygon for a specific basin from GeoJSON."""
    try:
        for feature in basin_geojson['features']:
            if feature['properties'].get('grdc_no') == gauge_id:
                return feature['geometry']
        return None
    except Exception as e:
        print(f"Error getting basin polygon: {e}")
        return None

def plot_basin_analysis(gauge_id, basin_name, gauge_lat, gauge_lon, 
                        obs_dates, obs_discharge, sim_dates, sim_discharge,
                        metrics, basin_polygon):
    """Create figure with basin map and discharge time series."""
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # Map subplot
        ax_map = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
        
        # Set map extent with some padding
        lat_range = 5
        lon_range = 5
        ax_map.set_extent([gauge_lon - lon_range, gauge_lon + lon_range,
                           gauge_lat - lat_range, gauge_lat + lat_range],
                          crs=ccrs.PlateCarree())
        
        # Add map features
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_map.add_feature(cfeature.COASTLINE)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.add_feature(cfeature.LAKES, alpha=0.5)
        ax_map.add_feature(cfeature.RIVERS)
        ax_map.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot basin polygon if available
        if basin_polygon:
            try:
                if basin_polygon['type'] == 'Polygon':
                    coords = np.array(basin_polygon['coordinates'][0])
                    ax_map.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2,
                               transform=ccrs.PlateCarree(), label='Basin boundary')
                    ax_map.fill(coords[:, 0], coords[:, 1], alpha=0.2, color='blue',
                               transform=ccrs.PlateCarree())
            except Exception as e:
                print(f"Error plotting basin polygon: {e}")
        
        # Plot gauge location
        ax_map.plot(gauge_lon, gauge_lat, 'r*', markersize=20,
                   transform=ccrs.PlateCarree(), label='Gauge location')
        
        ax_map.set_title(f'{basin_name} Basin (Gauge ID: {gauge_id})', fontsize=14, fontweight='bold')
        ax_map.legend(loc='upper left')
        
        # Time series subplot
        ax_ts = plt.subplot(2, 1, 2)
        
        # Plot observations
        if len(obs_dates) > 0:
            ax_ts.plot(obs_dates, obs_discharge, 'b-', linewidth=1.5, 
                      label='Observations', alpha=0.8)
        
        # Plot simulations
        if len(sim_dates) > 0:
            ax_ts.plot(sim_dates, sim_discharge, 'r-', linewidth=1.5,
                      label='E3SM MOSART', alpha=0.8)
        
        ax_ts.set_xlabel('Date', fontsize=11)
        ax_ts.set_ylabel('Discharge (m³/s)', fontsize=11)
        ax_ts.set_title('Streamflow Comparison', fontsize=12, fontweight='bold')
        ax_ts.legend(loc='upper left', fontsize=10)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45)
        
        # Add metrics text
        if metrics:
            metrics_text = (f"RMSE: {metrics['RMSE']:.2f} m³/s\n"
                          f"NSE: {metrics['NSE']:.3f}\n"
                          f"KGE: {metrics['KGE']:.3f}\n"
                          f"PBIAS: {metrics['PBIAS']:.1f}%\n"
                          f"n: {metrics['n_samples']}")
            ax_ts.text(0.02, 0.98, metrics_text, transform=ax_ts.transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(OUTPUT_DIR, f"basin_{gauge_id}_{basin_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating plot for basin {gauge_id}: {e}")
        plt.close()

def interpolate_daily_to_monthly(obs_df):
    """Interpolate daily observations to monthly values (mean)."""
    try:
        obs_df['year_month'] = obs_df['date'].dt.to_period('M')
        monthly = obs_df.groupby('year_month')['discharge_m3s'].mean().reset_index()
        monthly['date'] = monthly['year_month'].dt.to_timestamp()
        return monthly[['date', 'discharge_m3s']].reset_index(drop=True)
    except Exception as e:
        print(f"Error interpolating to monthly: {e}")
        return None

def main():
    """Main analysis function."""
    print("Starting basin-scale streamflow analysis...")
    
    # Load metadata and polygons
    gauge_metadata = load_gauge_metadata()
    basin_geojson = load_basin_polygons()
    
    if gauge_metadata is None:
        print("Failed to load gauge metadata")
        return
    
    # Load MOSART data
    print(f"Loading MOSART data for {START_YEAR}-{END_YEAR}...")
    mosart_ds = load_mosart_data(START_YEAR, END_YEAR)
    
    if mosart_ds is None:
        print("Failed to load MOSART data")
        return
    
    # Convert MOSART time to datetime
    mosart_times = pd.to_datetime(mosart_ds['time'].values)
    
    # Results storage
    results = []
    
    # Process each basin
    for gauge_id, basin_name in BASINS.items():
        print(f"\nProcessing {basin_name} (Gauge ID: {gauge_id})...")
        
        # Get gauge metadata
        gauge_row = gauge_metadata[gauge_metadata['gauge_id'] == gauge_id]
        if gauge_row.empty:
            print(f"  Gauge {gauge_id} not found in metadata")
            continue
        
        gauge_lat = gauge_row['lat'].values[0]
        gauge_lon = gauge_row['lon'].values[0]
        
        print(f"  Gauge location: ({gauge_lat:.2f}, {gauge_lon:.2f})")
        
        # Load observation data
        obs_df = load_observation_data(gauge_id, START_YEAR, END_YEAR)
        if obs_df is None or len(obs_df) == 0:
            print(f"  No observation data available")
            continue
        
        # Convert daily observations to monthly for comparison with MOSART h0
        obs_monthly = interpolate_daily_to_monthly(obs_df)
        if obs_monthly is None or len(obs_monthly) == 0:
            print(f"  Failed to convert observations to monthly")
            continue
        
        # Extract simulated discharge
        sim_discharge = extract_discharge_at_gauge(mosart_ds, gauge_lat, gauge_lon)
        if sim_discharge is None:
            print(f"  Failed to extract simulated discharge")
            continue
        
        # Align time series
        obs_dates = obs_monthly['date'].values
        obs_vals = obs_monthly['discharge_m3s'].values
        sim_dates = mosart_times.values
        sim_vals = sim_discharge
        
        # Ensure same length
        min_len = min(len(obs_vals), len(sim_vals))
        obs_vals = obs_vals[:min_len]
        sim_vals = sim_vals[:min_len]
        obs_dates = obs_dates[:min_len]
        sim_dates = sim_dates[:min_len]
        
        print(f"  Comparing {min_len} monthly values")
        
        # Compute metrics
        metrics = compute_metrics(obs_vals, sim_vals)
        
        if metrics:
            print(f"  RMSE: {metrics['RMSE']:.2f} m³/s")
            print(f"  NSE: {metrics['NSE']:.3f}")
            print(f"  KGE: {metrics['KGE']:.3f}")
            print(f"  PBIAS: {metrics['PBIAS']:.1f}%")
            
            results.append({
                'gauge_id': gauge_id,
                'basin_name': basin_name,
                'gauge_lat': gauge_lat,
                'gauge_lon': gauge_lon,
                'RMSE': metrics['RMSE'],
                'NSE': metrics['NSE'],
                'KGE': metrics['KGE'],
                'PBIAS': metrics['PBIAS'],
                'n_samples': metrics['n_samples']
            })
        
        # Get basin polygon
        basin_polygon = None
        if basin_geojson:
            basin_polygon = get_basin_polygon(basin_geojson, gauge_id)
        
        # Create figure
        plot_basin_analysis(gauge_id, basin_name, gauge_lat, gauge_lon,
                          obs_dates, obs_vals, sim_dates, sim_vals,
                          metrics, basin_polygon)
    
    # Save results summary
    if results:
        results_df = pd.DataFrame(results)
        results_file = os.path.join(OUTPUT_DIR, "basin_metrics_summary.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved results summary: {results_file}")
        print("\nMetrics Summary:")
        print(results_df.to_string(index=False))
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()