#!/usr/bin/env python
"""
Compare E3SM simulated river discharge against observations for 1985-1989.
Compute flow duration curve (FDC) distributional metrics for each gauge.
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
import json

warnings.filterwarnings('ignore')

# Configuration
CASE_NAME = "sample.v3.LR.historical"
E3SM_ROOT = "./data/sample/e3sm/"
OBS_ROOT = "./data/sample/obs/"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_04_streamflow_fdc/run2_debug/v3/output"
START_YEAR = 1985
END_YEAR = 1989

# Target gauges for detailed analysis
TARGET_GAUGES = {
    'Amazon': 3629000,
    'Missouri': 4121801,
    'Columbia': 4115200,
    'Danube': 6742900,
    'Mekong': 2969100,
    'Orange': 1159100
}

def wasserstein_distance(u_values, v_values):
    """
    Compute Wasserstein distance (Earth Mover's Distance) between two distributions.
    Implementation of 1D Wasserstein distance.
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    
    # Interpolate to common grid
    n_samples = max(len(u_sorted), len(v_sorted))
    u_interp = np.interp(np.linspace(0, 1, n_samples), 
                         np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_interp = np.interp(np.linspace(0, 1, n_samples), 
                         np.linspace(0, 1, len(v_sorted)), v_sorted)
    
    # Compute L1 distance
    wd = np.mean(np.abs(u_interp - v_interp))
    return wd

def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_gauge_metadata():
    """Load gauge metadata from CSV."""
    try:
        metadata_file = os.path.join(OBS_ROOT, "gauge_metadata.csv")
        metadata = pd.read_csv(metadata_file)
        print(f"Loaded metadata for {len(metadata)} gauges")
        return metadata
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        return None

def load_observation_data(gauge_id, start_year, end_year):
    """Load observation data for a specific gauge."""
    try:
        obs_file = os.path.join(OBS_ROOT, "streamflow", f"{gauge_id}.csv")
        if not os.path.exists(obs_file):
            return None
        
        obs_data = pd.read_csv(obs_file)
        obs_data['date'] = pd.to_datetime(obs_data['date'])
        
        # Filter to date range
        mask = (obs_data['date'].dt.year >= start_year) & (obs_data['date'].dt.year <= end_year)
        obs_data = obs_data[mask].copy()
        
        if len(obs_data) == 0:
            return None
        
        obs_data = obs_data.sort_values('date').reset_index(drop=True)
        return obs_data
    except Exception as e:
        print(f"Error loading observation data for gauge {gauge_id}: {e}")
        return None

def load_mosart_data(start_year, end_year):
    """Load MOSART monthly discharge data."""
    try:
        mosart_files = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                pattern = os.path.join(
                    E3SM_ROOT, "rof",
                    f"{CASE_NAME}.mosart.h0.{year:04d}-{month:02d}.nc"
                )
                matching_files = glob.glob(pattern)
                mosart_files.extend(matching_files)
        
        if not mosart_files:
            print("No MOSART files found")
            return None
        
        mosart_files.sort()
        print(f"Found {len(mosart_files)} MOSART files")
        
        # Load and concatenate
        ds_list = []
        for f in mosart_files:
            try:
                ds = xr.open_dataset(f)
                ds_list.append(ds)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not ds_list:
            return None
        
        ds = xr.concat(ds_list, dim='time')
        return ds
    except Exception as e:
        print(f"Error loading MOSART data: {e}")
        return None

def find_nearest_grid_cell(lat, lon, grid_lats, grid_lons):
    """Find nearest grid cell to a given lat/lon."""
    try:
        # Handle longitude wrapping
        lon_grid = np.array(grid_lons, dtype=float)
        lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
        lon_target = lon if lon > -180 else lon + 360
        
        lat_grid = np.array(grid_lats, dtype=float)
        
        distances = np.sqrt((lat_grid - lat)**2 + (lon_grid - lon_target)**2)
        idx = np.nanargmin(distances)
        return idx
    except Exception as e:
        print(f"Error in find_nearest_grid_cell: {e}")
        return None

def extract_simulated_discharge(ds, gauge_lat, gauge_lon):
    """Extract simulated discharge time series for a gauge location."""
    try:
        if 'RIVER_DISCHARGE_OVER_LAND_LIQ' not in ds.data_vars:
            return None
        
        # Get grid coordinates
        discharge_var = ds['RIVER_DISCHARGE_OVER_LAND_LIQ']
        
        # Handle different coordinate naming conventions
        lat_name = None
        lon_name = None
        
        for name in ['lat', 'latitude', 'y']:
            if name in ds.coords:
                lat_name = name
                break
        
        for name in ['lon', 'longitude', 'x']:
            if name in ds.coords:
                lon_name = name
                break
        
        if lat_name is None or lon_name is None:
            print(f"Could not find lat/lon coordinates. Available: {list(ds.coords.keys())}")
            return None
        
        grid_lats = ds[lat_name].values
        grid_lons = ds[lon_name].values
        
        # Find nearest cell
        idx = find_nearest_grid_cell(gauge_lat, gauge_lon, grid_lats, grid_lons)
        if idx is None:
            return None
        
        # Extract discharge - handle different dimension orders
        discharge_data = discharge_var.values
        
        if discharge_data.ndim == 3:
            # Time x lat x lon or similar
            # Check if we need to unravel the index
            if discharge_data.shape[1] * discharge_data.shape[2] == len(grid_lats) * len(grid_lons):
                # 2D grid case
                i, j = np.unravel_index(idx, (discharge_data.shape[1], discharge_data.shape[2]))
                discharge = discharge_data[:, i, j]
            else:
                # 1D grid case
                discharge = discharge_data[:, idx, 0] if discharge_data.shape[2] == 1 else discharge_data[:, 0, idx]
        elif discharge_data.ndim == 2:
            # Time x grid_cells
            discharge = discharge_data[:, idx]
        else:
            print(f"Unexpected discharge data shape: {discharge_data.shape}")
            return None
        
        times = ds['time'].values
        
        # Create dataframe
        sim_data = pd.DataFrame({
            'date': pd.to_datetime(times),
            'discharge_m3s': discharge
        })
        
        return sim_data
    except Exception as e:
        print(f"Error extracting simulated discharge: {e}")
        import traceback
        traceback.print_exc()
        return None

def interpolate_monthly_to_daily(monthly_data):
    """Interpolate monthly data to daily."""
    try:
        # Create daily date range
        start_date = monthly_data['date'].min()
        end_date = monthly_data['date'].max()
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create daily dataframe
        daily_data = pd.DataFrame({'date': daily_dates})
        
        # Merge and interpolate
        merged = pd.merge_asof(
            daily_data, monthly_data.sort_values('date'),
            on='date', direction='nearest'
        )
        
        # Linear interpolation
        merged['discharge_m3s'] = merged['discharge_m3s'].interpolate(method='linear')
        
        return merged
    except Exception as e:
        print(f"Error interpolating monthly to daily: {e}")
        return None

def compute_fdc_metrics(obs_discharge, sim_discharge):
    """Compute FDC distributional metrics."""
    try:
        # Remove NaN values
        obs_clean = obs_discharge[~np.isnan(obs_discharge)]
        sim_clean = sim_discharge[~np.isnan(sim_discharge)]
        
        if len(obs_clean) < 10 or len(sim_clean) < 10:
            return None
        
        # Sort in descending order for FDC
        obs_sorted = np.sort(obs_clean)[::-1]
        sim_sorted = np.sort(sim_clean)[::-1]
        
        # Compute exceedance probabilities
        obs_prob = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
        sim_prob = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted)
        
        # Compute quantiles
        obs_q10 = np.percentile(obs_clean, 90)
        obs_q50 = np.percentile(obs_clean, 50)
        obs_q90 = np.percentile(obs_clean, 10)
        
        sim_q10 = np.percentile(sim_clean, 90)
        sim_q50 = np.percentile(sim_clean, 50)
        sim_q90 = np.percentile(sim_clean, 10)
        
        # Volume bias
        obs_mean = np.mean(obs_clean)
        sim_mean = np.mean(sim_clean)
        volume_bias = (sim_mean - obs_mean) / obs_mean if obs_mean > 0 else np.nan
        
        # Wasserstein distance (Earth Mover's Distance)
        # Normalize by mean for comparison
        obs_norm = obs_clean / obs_mean if obs_mean > 0 else obs_clean
        sim_norm = sim_clean / sim_mean if sim_mean > 0 else sim_clean
        
        wd = wasserstein_distance(obs_norm, sim_norm)
        
        # Quantile ratios
        q10_ratio = sim_q10 / obs_q10 if obs_q10 > 0 else np.nan
        q50_ratio = sim_q50 / obs_q50 if obs_q50 > 0 else np.nan
        q90_ratio = sim_q90 / obs_q90 if obs_q90 > 0 else np.nan
        
        metrics = {
            'obs_mean': obs_mean,
            'sim_mean': sim_mean,
            'volume_bias': volume_bias,
            'wasserstein_distance': wd,
            'obs_q10': obs_q10,
            'sim_q10': sim_q10,
            'q10_ratio': q10_ratio,
            'obs_q50': obs_q50,
            'sim_q50': sim_q50,
            'q50_ratio': q50_ratio,
            'obs_q90': obs_q90,
            'sim_q90': sim_q90,
            'q90_ratio': q90_ratio,
            'obs_sorted': obs_sorted,
            'sim_sorted': sim_sorted,
            'obs_prob': obs_prob,
            'sim_prob': sim_prob
        }
        
        return metrics
    except Exception as e:
        print(f"Error computing FDC metrics: {e}")
        return None

def process_all_gauges(metadata, mosart_ds):
    """Process all gauges and compute metrics."""
    results = []
    fdc_data = {}
    
    for idx, row in metadata.iterrows():
        gauge_id = row['gauge_id']
        gauge_lat = row['lat']
        gauge_lon = row['lon']
        river_name = row.get('river_name', 'Unknown')
        
        print(f"Processing gauge {gauge_id} ({river_name})...")
        
        # Load observations
        obs_data = load_observation_data(gauge_id, START_YEAR, END_YEAR)
        if obs_data is None or len(obs_data) == 0:
            print(f"  No observation data for gauge {gauge_id}")
            continue
        
        # Extract simulated discharge
        sim_data = extract_simulated_discharge(mosart_ds, gauge_lat, gauge_lon)
        if sim_data is None:
            print(f"  Could not extract simulated data for gauge {gauge_id}")
            continue
        
        # Interpolate simulated data to daily
        sim_data_daily = interpolate_monthly_to_daily(sim_data)
        if sim_data_daily is None:
            print(f"  Could not interpolate simulated data for gauge {gauge_id}")
            continue
        
        # Merge obs and sim on date
        merged = pd.merge(
            obs_data[['date', 'discharge_m3s']].rename(columns={'discharge_m3s': 'obs_discharge'}),
            sim_data_daily[['date', 'discharge_m3s']].rename(columns={'discharge_m3s': 'sim_discharge'}),
            on='date', how='inner'
        )
        
        if len(merged) < 10:
            print(f"  Insufficient data for gauge {gauge_id}")
            continue
        
        # Compute metrics
        metrics = compute_fdc_metrics(merged['obs_discharge'].values, merged['sim_discharge'].values)
        if metrics is None:
            print(f"  Could not compute metrics for gauge {gauge_id}")
            continue
        
        # Store results
        result_row = {
            'gauge_id': gauge_id,
            'river_name': river_name,
            'lat': gauge_lat,
            'lon': gauge_lon,
            'obs_mean': metrics['obs_mean'],
            'sim_mean': metrics['sim_mean'],
            'volume_bias': metrics['volume_bias'],
            'wasserstein_distance': metrics['wasserstein_distance'],
            'obs_q10': metrics['obs_q10'],
            'sim_q10': metrics['sim_q10'],
            'q10_ratio': metrics['q10_ratio'],
            'obs_q50': metrics['obs_q50'],
            'sim_q50': metrics['sim_q50'],
            'q50_ratio': metrics['q50_ratio'],
            'obs_q90': metrics['obs_q90'],
            'sim_q90': metrics['sim_q90'],
            'q90_ratio': metrics['q90_ratio'],
            'n_days': len(merged)
        }
        results.append(result_row)
        
        # Store FDC data
        fdc_data[gauge_id] = {
            'obs_sorted': metrics['obs_sorted'],
            'sim_sorted': metrics['sim_sorted'],
            'obs_prob': metrics['obs_prob'],
            'sim_prob': metrics['sim_prob'],
            'river_name': river_name
        }
        
        print(f"  Gauge {gauge_id}: WD={metrics['wasserstein_distance']:.4f}, VB={metrics['volume_bias']:.3f}")
    
    return pd.DataFrame(results), fdc_data

def save_metrics_csv(results_df):
    """Save metrics to CSV."""
    try:
        output_file = os.path.join(OUTPUT_DIR, "fdc_metrics_all_gauges.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved metrics to {output_file}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

def save_fdc_percentiles_csv(fdc_data):
    """Save FDC percentile data to CSV."""
    try:
        output_file = os.path.join(OUTPUT_DIR, "fdc_percentiles.csv")
        
        rows = []
        for gauge_id, data in fdc_data.items():
            # Compute percentiles
            percentiles = np.arange(0, 101, 1)
            for p in percentiles:
                obs_val = np.percentile(data['obs_sorted'], p)
                sim_val = np.percentile(data['sim_sorted'], p)
                rows.append({
                    'gauge_id': gauge_id,
                    'river_name': data['river_name'],
                    'percentile': p,
                    'obs_discharge': obs_val,
                    'sim_discharge': sim_val
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Saved FDC percentiles to {output_file}")
    except Exception as e:
        print(f"Error saving FDC percentiles CSV: {e}")

def create_fdc_map_figure(results_df, fdc_data):
    """Create figure with map and FDC panels."""
    try:
        # Filter to target gauges
        target_ids = list(TARGET_GAUGES.values())
        target_results = results_df[results_df['gauge_id'].isin(target_ids)].copy()
        
        if len(target_results) == 0:
            print("No target gauges found in results")
            return
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
        
        # Map panel (top left, spanning 2 rows and 2 columns)
        ax_map = fig.add_subplot(gs[0:2, 0:2], projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.coastlines(resolution='50m', linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax_map.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.5)
        ax_map.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot gauges on map with Wasserstein distance as color
        scatter = ax_map.scatter(
            target_results['lon'], target_results['lat'],
            c=target_results['wasserstein_distance'],
            s=300, cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=2,
            transform=ccrs.PlateCarree(), vmin=0, vmax=target_results['wasserstein_distance'].max()
        )
        
        # Add gauge labels
        for idx, row in target_results.iterrows():
            ax_map.text(
                row['lon'], row['lat'], str(row['gauge_id']),
                fontsize=8, ha='center', va='center', color='black', weight='bold',
                transform=ccrs.PlateCarree()
            )
        
        ax_map.set_title('Gauge Locations and Wasserstein Distance', fontsize=12, weight='bold')
        cbar = plt.colorbar(scatter, ax=ax_map, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Wasserstein Distance', fontsize=10)
        
        # FDC panels for each target gauge
        fdc_axes = [
            fig.add_subplot(gs[0, 2:]),
            fig.add_subplot(gs[1, 2:]),
            fig.add_subplot(gs[2, 0:2]),
            fig.add_subplot(gs[2, 2:])
        ]
        
        # Add more axes if needed
        for i in range(len(target_results) - len(fdc_axes)):
            fdc_axes.append(fig.add_subplot(gs[2, 2:]))
        
        # Plot FDCs
        for idx, (ax_idx, (_, row)) in enumerate(zip(range(len(fdc_axes)), target_results.iterrows())):
            if idx >= len(fdc_axes):
                break
            
            gauge_id = row['gauge_id']
            if gauge_id not in fdc_data:
                continue
            
            ax = fdc_axes[idx]
            data = fdc_data[gauge_id]
            
            # Plot FDCs
            ax.semilogy(data['obs_prob'] * 100, data['obs_sorted'], 'b-', linewidth=2, label='Observed', alpha=0.7)
            ax.semilogy(data['sim_prob'] * 100, data['sim_sorted'], 'r--', linewidth=2, label='Simulated', alpha=0.7)
            
            ax.set_xlabel('Exceedance Probability (%)', fontsize=10)
            ax.set_ylabel('Discharge (m³/s)', fontsize=10)
            ax.set_title(f"{data['river_name']} (ID: {gauge_id})\nWD={row['wasserstein_distance']:.4f}", 
                        fontsize=10, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            ax.set_xlim([0, 100])
        
        plt.suptitle(f'Flow Duration Curves: E3SM vs Observations (1985-1989)', 
                    fontsize=14, weight='bold', y=0.995)
        
        output_file = os.path.join(OUTPUT_DIR, "fdc_map_and_curves.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved FDC map figure to {output_file}")
        plt.close()
    except Exception as e:
        print(f"Error creating FDC map figure: {e}")

def main():
    """Main execution."""
    print("=" * 80)
    print("E3SM River Discharge vs Observations: FDC Analysis (1985-1989)")
    print("=" * 80)
    
    # Create output directory
    create_output_dir()
    
    # Load gauge metadata
    metadata = load_gauge_metadata()
    if metadata is None or len(metadata) == 0:
        print("Error: Could not load gauge metadata")
        return
    
    # Load MOSART data
    print("\nLoading MOSART data...")
    mosart_ds = load_mosart_data(START_YEAR, END_YEAR)
    if mosart_ds is None:
        print("Error: Could not load MOSART data")
        return
    
    print(f"MOSART data variables: {list(mosart_ds.data_vars.keys())}")
    print(f"MOSART data coords: {list(mosart_ds.coords.keys())}")
    if 'RIVER_DISCHARGE_OVER_LAND_LIQ' in mosart_ds.data_vars:
        print(f"MOSART discharge shape: {mosart_ds['RIVER_DISCHARGE_OVER_LAND_LIQ'].shape}")
    print(f"MOSART time range: {mosart_ds['time'].values[0]} to {mosart_ds['time'].values[-1]}")
    
    # Process all gauges
    print("\nProcessing gauges...")
    results_df, fdc_data = process_all_gauges(metadata, mosart_ds)
    
    if len(results_df) == 0:
        print("Error: No gauges processed successfully")
        return
    
    print(f"\nSuccessfully processed {len(results_df)} gauges")
    
    # Save results
    print("\nSaving results...")
    save_metrics_csv(results_df)
    save_fdc_percentiles_csv(fdc_data)
    
    # Create visualization
    print("\nCreating visualization...")
    create_fdc_map_figure(results_df, fdc_data)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Number of gauges processed: {len(results_df)}")
    print(f"\nWasserstein Distance:")
    print(f"  Mean: {results_df['wasserstein_distance'].mean():.4f}")
    print(f"  Std:  {results_df['wasserstein_distance'].std():.4f}")
    print(f"  Min:  {results_df['wasserstein_distance'].min():.4f}")
    print(f"  Max:  {results_df['wasserstein_distance'].max():.4f}")
    
    print(f"\nVolume Bias:")
    print(f"  Mean: {results_df['volume_bias'].mean():.3f}")
    print(f"  Std:  {results_df['volume_bias'].std():.3f}")
    print(f"  Min:  {results_df['volume_bias'].min():.3f}")
    print(f"  Max:  {results_df['volume_bias'].max():.3f}")
    
    print(f"\nQ50 Ratio (Median):")
    print(f"  Mean: {results_df['q50_ratio'].mean():.3f}")
    print(f"  Std:  {results_df['q50_ratio'].std():.3f}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()