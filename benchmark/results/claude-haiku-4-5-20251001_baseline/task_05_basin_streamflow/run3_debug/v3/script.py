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
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_05_basin_streamflow/run3_debug/v3/output"
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
        
        # Load first file to check structure
        ds_first = xr.open_dataset(sorted(mosart_files)[0])
        print(f"MOSART grid shape: lat={ds_first['lat'].shape}, lon={ds_first['lon'].shape}")
        print(f"RIVER_DISCHARGE_OVER_LAND_LIQ shape: {ds_first['RIVER_DISCHARGE_OVER_LAND_LIQ'].shape}")
        
        # Load and concatenate
        ds_list = [xr.open_dataset(f) for f in sorted(mosart_files)]
        ds = xr.concat(ds_list, dim='time')
        
        return ds
    except Exception as e:
        print(f"Error loading MOSART data: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_nearest_grid_cell(lat, lon, grid_lats, grid_lons):
    """Find nearest grid cell to a given lat/lon point."""
    try:
        # Create 2D mesh grid if 1D arrays
        if grid_lats.ndim == 1 and grid_lons.ndim == 1:
            lons_2d, lats_2d = np.meshgrid(grid_lons, grid_lats)
        else:
            lats_2d = grid_lats
            lons_2d = grid_lons
        
        # Flatten
        lats_flat = lats_2d.ravel()
        lons_flat = lons_2d.ravel()
        
        # Create KDTree for efficient nearest neighbor search
        points = np.column_stack([lats_flat, lons_flat])
        tree = cKDTree(points)
        
        # Query nearest neighbor
        dist, idx = tree.query([lat, lon])
        
        return idx
    except Exception as e:
        print(f"Error finding nearest grid cell: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_discharge_at_gauge(ds, gauge_lat, gauge_lon):
    """Extract discharge time series at nearest grid cell to gauge."""
    try:
        # Get grid coordinates
        grid_lats = ds['lat'].values
        grid_lons = ds['lon'].values
        
        print(f"  Grid lats shape: {grid_lats.shape}, lons shape: {grid_lons.shape}")
        
        # Handle longitude convention (0-360 vs -180-180)
        if grid_lons.max() > 180:
            grid_lons_adj = ((grid_lons + 180) % 360) - 180
        else:
            grid_lons_adj = grid_lons.copy()
        
        gauge_lon_adj = ((gauge_lon + 180) % 360) - 180
        
        # Find nearest cell
        idx = find_nearest_grid_cell(gauge_lat, gauge_lon_adj, grid_lats, grid_lons_adj)
        
        if idx is None:
            return None
        
        print(f"  Found nearest grid cell at index {idx}")
        
        # Extract discharge - handle both 1D and 2D cases
        discharge_var = ds['RIVER_DISCHARGE_OVER_LAND_LIQ']
        
        print(f"  Discharge variable shape: {discharge_var.shape}")
        
        if discharge_var.ndim == 2:
            # 1D grid (flattened) - time x grid
            discharge = discharge_var.values[:, idx]
        elif discharge_var.ndim == 3:
            # 2D grid - time x lat x lon
            # Convert flat index to 2D indices
            shape = (grid_lats.shape[0], grid_lons.shape[0])
            i, j = np.unravel_index(idx, shape)
            print(f"  Unraveled indices: i={i}, j={j}")
            discharge = discharge_var.values[:, i, j]
        else:
            print(f"Unexpected discharge variable dimensions: {discharge_var.ndim}")
            return None
        
        time = ds['time'].values
        
        return discharge, time
    except Exception as e:
        print(f"Error extracting discharge: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_metrics(obs_discharge, sim_discharge):
    """Compute validation metrics."""
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(obs_discharge) | np.isnan(sim_discharge))
        obs = obs_discharge[valid_mask]
        sim = sim_discharge[valid_mask]
        
        if len(obs) == 0:
            return None
        
        # RMSE
        rmse = np.sqrt(np.mean((sim - obs) ** 2))
        
        # NSE (Nash-Sutcliffe Efficiency)
        obs_mean = np.mean(obs)
        nse = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - obs_mean) ** 2))
        
        # KGE (Kling-Gupta Efficiency)
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        # PBIAS (Percent Bias)
        pbias = 100 * np.sum(sim - obs) / np.sum(obs)
        
        return {
            'rmse': rmse,
            'nse': nse,
            'kge': kge,
            'pbias': pbias,
            'n_samples': len(obs)
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None

def get_basin_polygon(gauge_id, basin_polygons):
    """Get polygon for a basin."""
    try:
        if basin_polygons is None:
            return None
        
        for feature in basin_polygons['features']:
            if feature['properties'].get('grdc_no') == gauge_id:
                return feature['geometry']
        
        return None
    except Exception as e:
        print(f"Error getting basin polygon: {e}")
        return None

def plot_basin_analysis(gauge_id, basin_name, obs_data, sim_data, sim_time, 
                        basin_polygon, metadata, metrics, output_dir):
    """Create figure with basin map and discharge time series."""
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # Get gauge location
        gauge_row = metadata[metadata['gauge_id'] == gauge_id]
        if gauge_row.empty:
            print(f"Gauge {gauge_id} not found in metadata")
            return
        
        gauge_lat = gauge_row['lat'].values[0]
        gauge_lon = gauge_row['lon'].values[0]
        
        # Left panel: Basin map
        ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        
        # Add map features
        ax1.add_feature(cfeature.LAND, facecolor='lightgray')
        ax1.add_feature(cfeature.OCEAN, facecolor='white')
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.add_feature(cfeature.LAKES, alpha=0.5)
        ax1.add_feature(cfeature.RIVERS, linewidth=0.5)
        
        # Plot basin polygon
        if basin_polygon is not None:
            try:
                if basin_polygon['type'] == 'Polygon':
                    coords = np.array(basin_polygon['coordinates'][0])
                    ax1.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2, 
                            label='Basin boundary', transform=ccrs.PlateCarree())
                    ax1.fill(coords[:, 0], coords[:, 1], alpha=0.2, color='blue',
                            transform=ccrs.PlateCarree())
            except Exception as e:
                print(f"Error plotting basin polygon: {e}")
        
        # Plot gauge location
        ax1.plot(gauge_lon, gauge_lat, 'r*', markersize=20, 
                label='Gauge location', transform=ccrs.PlateCarree())
        
        # Set map extent
        if basin_polygon is not None:
            try:
                coords = np.array(basin_polygon['coordinates'][0])
                lons = coords[:, 0]
                lats = coords[:, 1]
                margin = 5
                ax1.set_extent([lons.min() - margin, lons.max() + margin,
                               lats.min() - margin, lats.max() + margin],
                              crs=ccrs.PlateCarree())
            except:
                ax1.set_global()
        else:
            ax1.set_global()
        
        ax1.set_title(f'{basin_name} Basin (Gauge ID: {gauge_id})', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower left', fontsize=10)
        ax1.gridlines(draw_labels=True, alpha=0.3)
        
        # Right panel: Time series
        ax2 = plt.subplot(1, 2, 2)
        
        # Convert time to datetime
        sim_time_dt = pd.to_datetime(sim_time)
        
        # Plot observed discharge
        if obs_data is not None and len(obs_data) > 0:
            ax2.plot(obs_data['date'], obs_data['discharge_m3s'], 'b-', 
                    linewidth=1.5, label='Observed', alpha=0.8)
        
        # Plot simulated discharge
        if sim_data is not None and len(sim_data) > 0:
            ax2.plot(sim_time_dt, sim_data, 'r-', 
                    linewidth=1.5, label='Simulated (E3SM)', alpha=0.8)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Discharge (m³/s)', fontsize=12)
        ax2.set_title('Streamflow Comparison (1985-1989)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add metrics text
        if metrics is not None:
            metrics_text = (
                f"RMSE: {metrics['rmse']:.2f} m³/s\n"
                f"NSE: {metrics['nse']:.3f}\n"
                f"KGE: {metrics['kge']:.3f}\n"
                f"PBIAS: {metrics['pbias']:.1f}%\n"
                f"N: {metrics['n_samples']}"
            )
            ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"basin_{gauge_id}_{basin_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating basin plot for {basin_name}: {e}")
        import traceback
        traceback.print_exc()
        plt.close()

def main():
    """Main analysis function."""
    print("Starting basin-scale streamflow analysis...")
    
    # Load metadata and basin polygons
    print("Loading gauge metadata and basin polygons...")
    metadata = load_gauge_metadata()
    basin_polygons = load_basin_polygons()
    
    if metadata is None:
        print("Failed to load gauge metadata")
        return
    
    # Load MOSART data
    print(f"Loading MOSART data for {START_YEAR}-{END_YEAR}...")
    mosart_ds = load_mosart_data(START_YEAR, END_YEAR)
    
    if mosart_ds is None:
        print("Failed to load MOSART data")
        return
    
    # Results storage
    results = []
    
    # Process each basin
    for gauge_id, basin_name in BASINS.items():
        print(f"\nProcessing {basin_name} basin (gauge {gauge_id})...")
        
        try:
            # Get gauge metadata
            gauge_row = metadata[metadata['gauge_id'] == gauge_id]
            if gauge_row.empty:
                print(f"  Gauge {gauge_id} not found in metadata")
                continue
            
            gauge_lat = gauge_row['lat'].values[0]
            gauge_lon = gauge_row['lon'].values[0]
            print(f"  Gauge location: lat={gauge_lat}, lon={gauge_lon}")
            
            # Load observation data
            obs_data = load_observation_data(gauge_id, START_YEAR, END_YEAR)
            if obs_data is None or len(obs_data) == 0:
                print(f"  No observation data for gauge {gauge_id}")
                continue
            
            print(f"  Loaded {len(obs_data)} observation records")
            
            # Extract simulated discharge
            result = extract_discharge_at_gauge(mosart_ds, gauge_lat, gauge_lon)
            if result is None:
                print(f"  Failed to extract simulated discharge")
                continue
            
            sim_discharge, sim_time = result
            print(f"  Extracted {len(sim_discharge)} simulated records")
            
            # Convert simulated discharge from monthly to daily (simple interpolation)
            sim_time_dt = pd.to_datetime(sim_time)
            
            # Create daily time series from monthly data
            date_range = pd.date_range(start=sim_time_dt[0], end=sim_time_dt[-1], freq='D')
            sim_daily = np.interp(date_range.astype(np.int64), 
                                 sim_time_dt.astype(np.int64), 
                                 sim_discharge)
            
            # Align observation and simulation data
            obs_data_aligned = obs_data[
                (obs_data['date'] >= date_range[0]) & 
                (obs_data['date'] <= date_range[-1])
            ].reset_index(drop=True)
            
            if len(obs_data_aligned) == 0:
                print(f"  No overlapping data between obs and sim")
                continue
            
            # Create aligned arrays
            obs_discharge_aligned = obs_data_aligned['discharge_m3s'].values
            sim_discharge_aligned = np.interp(
                obs_data_aligned['date'].astype(np.int64),
                date_range.astype(np.int64),
                sim_daily
            )
            
            # Compute metrics
            metrics = compute_metrics(obs_discharge_aligned, sim_discharge_aligned)
            
            if metrics is not None:
                print(f"  RMSE: {metrics['rmse']:.2f} m³/s")
                print(f"  NSE: {metrics['nse']:.3f}")
                print(f"  KGE: {metrics['kge']:.3f}")
                print(f"  PBIAS: {metrics['pbias']:.1f}%")
                
                # Store results
                results.append({
                    'gauge_id': gauge_id,
                    'basin_name': basin_name,
                    'lat': gauge_lat,
                    'lon': gauge_lon,
                    'rmse': metrics['rmse'],
                    'nse': metrics['nse'],
                    'kge': metrics['kge'],
                    'pbias': metrics['pbias'],
                    'n_samples': metrics['n_samples']
                })
            
            # Get basin polygon
            basin_polygon = get_basin_polygon(gauge_id, basin_polygons)
            
            # Create figure
            plot_basin_analysis(gauge_id, basin_name, obs_data_aligned, 
                              sim_discharge_aligned, obs_data_aligned['date'].values,
                              basin_polygon, metadata, metrics, OUTPUT_DIR)
            
        except Exception as e:
            print(f"  Error processing {basin_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    if results:
        try:
            results_df = pd.DataFrame(results)
            results_file = os.path.join(OUTPUT_DIR, "basin_metrics.csv")
            results_df.to_csv(results_file, index=False)
            print(f"\nSaved metrics to: {results_file}")
            print("\nMetrics Summary:")
            print(results_df.to_string(index=False))
        except Exception as e:
            print(f"Error saving results: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()