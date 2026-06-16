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
from datetime import datetime
import cftime

warnings.filterwarnings('ignore')

# Configuration
CASE_NAME = "sample.v3.LR.historical"
E3SM_ROOT = "./data/sample/e3sm/"
OBS_ROOT = "./data/sample/obs/"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_04_streamflow_fdc/run3_debug/v3/output"
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
    Implementation using sorted quantiles.
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    
    # Interpolate to common quantiles
    n = max(len(u_sorted), len(v_sorted))
    u_quantiles = np.interp(np.linspace(0, 1, n), 
                            np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_quantiles = np.interp(np.linspace(0, 1, n), 
                            np.linspace(0, 1, len(v_sorted)), v_sorted)
    
    # Wasserstein distance is mean absolute difference of quantiles
    return np.mean(np.abs(u_quantiles - v_quantiles))

def cftime_to_datetime(cftime_obj):
    """Convert cftime object to datetime."""
    try:
        return pd.Timestamp(cftime_obj.year, cftime_obj.month, cftime_obj.day,
                           cftime_obj.hour, cftime_obj.minute, cftime_obj.second)
    except Exception as e:
        print(f"Error converting cftime: {e}")
        return None

def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_gauge_metadata():
    """Load gauge metadata from CSV."""
    try:
        metadata_file = os.path.join(OBS_ROOT, "gauge_metadata.csv")
        df = pd.read_csv(metadata_file)
        print(f"Loaded {len(df)} gauges from metadata")
        return df
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        return None

def load_observation_data(gauge_id, start_year, end_year):
    """Load observation streamflow data for a gauge."""
    try:
        obs_file = os.path.join(OBS_ROOT, "streamflow", f"{gauge_id}.csv")
        if not os.path.exists(obs_file):
            return None
        
        df = pd.read_csv(obs_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to date range
        mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
        df = df[mask].copy()
        
        if len(df) == 0:
            return None
        
        return df.set_index('date')
    except Exception as e:
        print(f"Error loading observation data for gauge {gauge_id}: {e}")
        return None

def load_mosart_monthly_data(start_year, end_year):
    """Load MOSART monthly discharge data."""
    try:
        files = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                pattern = os.path.join(
                    E3SM_ROOT, "rof",
                    f"{CASE_NAME}.mosart.h0.{year:04d}-{month:02d}.nc"
                )
                matching = glob.glob(pattern)
                files.extend(matching)
        
        if not files:
            print("No MOSART monthly files found")
            return None
        
        files.sort()
        print(f"Found {len(files)} MOSART monthly files")
        
        # Load and concatenate
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True)
        return ds
    except Exception as e:
        print(f"Error loading MOSART monthly data: {e}")
        return None

def load_mosart_daily_data(start_year, end_year):
    """Load MOSART daily discharge data."""
    try:
        files = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                pattern = os.path.join(
                    E3SM_ROOT, "rof",
                    f"{CASE_NAME}.mosart.h1.{year:04d}-{month:02d}-*.nc"
                )
                matching = glob.glob(pattern)
                files.extend(matching)
        
        if not files:
            print("No MOSART daily files found, will use monthly data")
            return None
        
        files.sort()
        print(f"Found {len(files)} MOSART daily files")
        
        # Load and concatenate
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True)
        return ds
    except Exception as e:
        print(f"Error loading MOSART daily data: {e}")
        return None

def find_nearest_grid_cell(lat, lon, grid_lats, grid_lons):
    """Find nearest grid cell to a given lat/lon."""
    try:
        # Get coordinate arrays
        lats = grid_lats.values
        lons = grid_lons.values
        
        # Handle 1D vs 2D coordinates
        if lats.ndim == 1 and lons.ndim == 1:
            # 1D coordinates - create meshgrid
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            # 2D coordinates
            lat_grid = lats
            lon_grid = lons
        
        # Handle longitude wrapping
        lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
        
        # Calculate distances
        distances = np.sqrt((lat_grid - lat)**2 + (lon_grid - lon)**2)
        
        # Find minimum
        idx = np.unravel_index(np.argmin(distances), distances.shape)
        return idx
    except Exception as e:
        print(f"Error in find_nearest_grid_cell: {e}")
        return None

def extract_discharge_at_gauge(ds, gauge_lat, gauge_lon):
    """Extract discharge time series at nearest grid cell to gauge."""
    try:
        if 'RIVER_DISCHARGE_OVER_LAND_LIQ' not in ds.data_vars:
            print("RIVER_DISCHARGE_OVER_LAND_LIQ not found in dataset")
            return None
        
        # Find nearest grid cell
        idx = find_nearest_grid_cell(gauge_lat, gauge_lon, ds['lat'], ds['lon'])
        if idx is None:
            return None
        
        # Extract discharge - handle different dimension names
        discharge_var = ds['RIVER_DISCHARGE_OVER_LAND_LIQ']
        
        # Get dimension names
        dims = discharge_var.dims
        
        # Try different indexing approaches
        try:
            # Try with x, y dimensions
            if 'x' in dims and 'y' in dims:
                discharge = discharge_var.isel(x=idx[1], y=idx[0])
            # Try with lon, lat dimensions
            elif 'lon' in dims and 'lat' in dims:
                discharge = discharge_var.isel(lon=idx[1], lat=idx[0])
            # Try with other common dimension names
            else:
                # Get the last two dimensions (usually spatial)
                spatial_dims = [d for d in dims if d not in ['time', 'bnds']]
                if len(spatial_dims) >= 2:
                    discharge = discharge_var.isel({spatial_dims[-2]: idx[0], spatial_dims[-1]: idx[1]})
                else:
                    print(f"Could not identify spatial dimensions: {dims}")
                    return None
        except Exception as e:
            print(f"Indexing error: {e}, dims: {dims}")
            return None
        
        return discharge
    except Exception as e:
        print(f"Error extracting discharge: {e}")
        return None

def compute_fdc_metrics(simulated, observed):
    """
    Compute flow duration curve metrics.
    
    Parameters:
    - simulated: simulated discharge time series (m3/s)
    - observed: observed discharge time series (m3/s)
    
    Returns:
    - dict with metrics
    """
    metrics = {}
    
    try:
        # Remove NaN values
        sim_clean = simulated.dropna()
        obs_clean = observed.dropna()
        
        if len(sim_clean) == 0 or len(obs_clean) == 0:
            return None
        
        # Volume bias
        sim_mean = sim_clean.mean()
        obs_mean = obs_clean.mean()
        volume_bias = (sim_mean - obs_mean) / obs_mean if obs_mean != 0 else np.nan
        metrics['volume_bias'] = volume_bias
        
        # Quantiles
        quantiles = [10, 50, 90]
        for q in quantiles:
            sim_q = np.percentile(sim_clean, q)
            obs_q = np.percentile(obs_clean, q)
            ratio = sim_q / obs_q if obs_q != 0 else np.nan
            metrics[f'Q{q}_ratio'] = ratio
            metrics[f'Q{q}_sim'] = sim_q
            metrics[f'Q{q}_obs'] = obs_q
        
        # Wasserstein distance (Earth Mover's Distance)
        # Normalize by observed mean for scale-independent comparison
        obs_mean_val = obs_clean.mean()
        if obs_mean_val > 0:
            sim_norm = sim_clean / obs_mean_val
            obs_norm = obs_clean / obs_mean_val
            wd = wasserstein_distance(sim_norm.values, obs_norm.values)
        else:
            wd = np.nan
        metrics['wasserstein_distance'] = wd
        
        # Additional statistics
        metrics['sim_mean'] = sim_mean
        metrics['obs_mean'] = obs_mean
        metrics['sim_std'] = sim_clean.std()
        metrics['obs_std'] = obs_clean.std()
        metrics['sim_min'] = sim_clean.min()
        metrics['obs_min'] = obs_clean.min()
        metrics['sim_max'] = sim_clean.max()
        metrics['obs_max'] = obs_clean.max()
        metrics['n_obs_days'] = len(obs_clean)
        metrics['n_sim_days'] = len(sim_clean)
        
        return metrics
    except Exception as e:
        print(f"Error computing FDC metrics: {e}")
        return None

def compute_fdc_percentiles(simulated, observed, percentiles=None):
    """Compute FDC percentile values."""
    if percentiles is None:
        percentiles = np.arange(0, 101, 5)
    
    try:
        sim_clean = simulated.dropna()
        obs_clean = observed.dropna()
        
        if len(sim_clean) == 0 or len(obs_clean) == 0:
            return None
        
        result = {'percentile': percentiles}
        result['sim_discharge'] = [np.percentile(sim_clean, p) for p in percentiles]
        result['obs_discharge'] = [np.percentile(obs_clean, p) for p in percentiles]
        
        return pd.DataFrame(result)
    except Exception as e:
        print(f"Error computing FDC percentiles: {e}")
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
        
        print(f"\nProcessing gauge {gauge_id} ({river_name})...")
        
        # Load observations
        obs_data = load_observation_data(gauge_id, START_YEAR, END_YEAR)
        if obs_data is None or len(obs_data) == 0:
            print(f"  No observation data available")
            continue
        
        # Extract simulated discharge
        sim_discharge = extract_discharge_at_gauge(mosart_ds, gauge_lat, gauge_lon)
        if sim_discharge is None:
            print(f"  Could not extract simulated discharge")
            continue
        
        # Convert to pandas Series with same index
        try:
            # Convert cftime to datetime
            time_values = sim_discharge.time.values
            datetime_index = []
            for t in time_values:
                if isinstance(t, cftime.datetime):
                    dt = cftime_to_datetime(t)
                    if dt is not None:
                        datetime_index.append(dt)
                else:
                    datetime_index.append(pd.Timestamp(t))
            
            if len(datetime_index) != len(sim_discharge.values):
                print(f"  Time index mismatch")
                continue
            
            sim_series = pd.Series(
                sim_discharge.values,
                index=pd.DatetimeIndex(datetime_index)
            )
        except Exception as e:
            print(f"  Error converting simulated data to series: {e}")
            continue
        
        # Align time series
        obs_discharge = obs_data['discharge_m3s']
        
        # Find common dates
        common_dates = obs_discharge.index.intersection(sim_series.index)
        if len(common_dates) == 0:
            print(f"  No common dates between simulation and observations")
            continue
        
        obs_aligned = obs_discharge[common_dates]
        sim_aligned = sim_series[common_dates]
        
        # Compute metrics
        metrics = compute_fdc_metrics(sim_aligned, obs_aligned)
        if metrics is None:
            print(f"  Could not compute metrics")
            continue
        
        # Compute FDC percentiles
        fdc_df = compute_fdc_percentiles(sim_aligned, obs_aligned)
        if fdc_df is not None:
            fdc_data[gauge_id] = fdc_df
        
        # Store results
        result_row = {
            'gauge_id': gauge_id,
            'river_name': river_name,
            'lat': gauge_lat,
            'lon': gauge_lon,
        }
        result_row.update(metrics)
        results.append(result_row)
        
        print(f"  Volume bias: {metrics['volume_bias']:.3f}")
        print(f"  Wasserstein distance: {metrics['wasserstein_distance']:.3f}")
        print(f"  Q50 ratio: {metrics['Q50_ratio']:.3f}")
    
    return pd.DataFrame(results), fdc_data

def save_metrics_csv(metrics_df):
    """Save metrics to CSV."""
    try:
        output_file = os.path.join(OUTPUT_DIR, "fdc_metrics_all_gauges.csv")
        metrics_df.to_csv(output_file, index=False)
        print(f"\nSaved metrics to {output_file}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

def save_fdc_percentiles_csv(fdc_data):
    """Save FDC percentiles to CSV files."""
    try:
        for gauge_id, fdc_df in fdc_data.items():
            output_file = os.path.join(OUTPUT_DIR, f"fdc_percentiles_gauge_{gauge_id}.csv")
            fdc_df.to_csv(output_file, index=False)
        print(f"Saved FDC percentiles for {len(fdc_data)} gauges")
    except Exception as e:
        print(f"Error saving FDC percentiles CSV: {e}")

def create_fdc_map_and_panels(metrics_df, fdc_data, mosart_ds):
    """Create map with Wasserstein distance and FDC comparison panels."""
    try:
        # Filter to target gauges
        target_ids = list(TARGET_GAUGES.values())
        target_metrics = metrics_df[metrics_df['gauge_id'].isin(target_ids)].copy()
        
        if len(target_metrics) == 0:
            print("No target gauges found in results")
            return
        
        # Create figure with map and FDC panels
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
        
        # Map subplot
        ax_map = fig.add_subplot(gs[0:2, 0:2], projection=ccrs.PlateCarree())
        ax_map.coastlines()
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot all gauges
        scatter = ax_map.scatter(
            metrics_df['lon'], metrics_df['lat'],
            c=metrics_df['wasserstein_distance'],
            s=100, cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5,
            transform=ccrs.PlateCarree(), vmin=0, vmax=1
        )
        
        # Highlight target gauges
        ax_map.scatter(
            target_metrics['lon'], target_metrics['lat'],
            s=300, facecolors='none', edgecolors='red', linewidth=2,
            transform=ccrs.PlateCarree(), label='Target gauges'
        )
        
        # Add labels for target gauges
        for idx, row in target_metrics.iterrows():
            ax_map.text(
                row['lon'], row['lat'],
                f"{row['gauge_id']}", fontsize=8,
                transform=ccrs.PlateCarree(),
                ha='center', va='bottom'
            )
        
        cbar = plt.colorbar(scatter, ax=ax_map, orientation='vertical', pad=0.02)
        cbar.set_label('Wasserstein Distance', fontsize=10)
        ax_map.set_title('Gauge Locations and Wasserstein Distance', fontsize=12, fontweight='bold')
        ax_map.legend(loc='lower left')
        
        # FDC comparison panels for target gauges
        panel_positions = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)]
        
        for (gauge_name, gauge_id), (row_idx, col_idx) in zip(TARGET_GAUGES.items(), panel_positions):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Get metrics for this gauge
            gauge_metrics = target_metrics[target_metrics['gauge_id'] == gauge_id]
            if len(gauge_metrics) == 0:
                ax.text(0.5, 0.5, f'{gauge_name}\nNo data', ha='center', va='center')
                ax.set_title(f'{gauge_name} ({gauge_id})', fontsize=10, fontweight='bold')
                continue
            
            # Get FDC data
            if gauge_id not in fdc_data:
                ax.text(0.5, 0.5, f'{gauge_name}\nNo FDC data', ha='center', va='center')
                ax.set_title(f'{gauge_name} ({gauge_id})', fontsize=10, fontweight='bold')
                continue
            
            fdc_df = fdc_data[gauge_id]
            
            # Plot FDC
            ax.plot(fdc_df['percentile'], fdc_df['obs_discharge'], 'b-', linewidth=2, label='Observed')
            ax.plot(fdc_df['percentile'], fdc_df['sim_discharge'], 'r--', linewidth=2, label='Simulated')
            
            ax.set_xlabel('Exceedance Percentile (%)', fontsize=9)
            ax.set_ylabel('Discharge (m³/s)', fontsize=9)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            
            # Add metrics text
            metrics_row = gauge_metrics.iloc[0]
            text_str = (
                f"WD: {metrics_row['wasserstein_distance']:.3f}\n"
                f"Vol bias: {metrics_row['volume_bias']:.3f}\n"
                f"Q50 ratio: {metrics_row['Q50_ratio']:.3f}"
            )
            ax.text(0.98, 0.02, text_str, transform=ax.transAxes,
                   fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'{gauge_name} ({gauge_id})', fontsize=10, fontweight='bold')
        
        # Save figure
        output_file = os.path.join(OUTPUT_DIR, "fdc_map_and_panels.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved FDC map and panels to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating FDC map and panels: {e}")

def create_summary_statistics_figure(metrics_df):
    """Create summary statistics figure."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Volume bias distribution
        ax = axes[0, 0]
        ax.hist(metrics_df['volume_bias'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect')
        ax.set_xlabel('Volume Bias', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Volume Bias', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Wasserstein distance distribution
        ax = axes[0, 1]
        ax.hist(metrics_df['wasserstein_distance'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Wasserstein Distance', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Wasserstein Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Q50 ratio distribution
        ax = axes[1, 0]
        ax.hist(metrics_df['Q50_ratio'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Perfect')
        ax.set_xlabel('Q50 Ratio (Sim/Obs)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Q50 Ratio', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter: Volume bias vs Wasserstein distance
        ax = axes[1, 1]
        ax.scatter(metrics_df['volume_bias'], metrics_df['wasserstein_distance'],
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Volume Bias', fontsize=11)
        ax.set_ylabel('Wasserstein Distance', fontsize=11)
        ax.set_title('Volume Bias vs Wasserstein Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, "summary_statistics.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved summary statistics to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating summary statistics figure: {e}")

def main():
    """Main execution function."""
    print("=" * 80)
    print("E3SM River Discharge vs Observations - FDC Analysis")
    print("=" * 80)
    
    # Create output directory
    create_output_dir()
    
    # Load gauge metadata
    print("\n1. Loading gauge metadata...")
    metadata = load_gauge_metadata()
    if metadata is None or len(metadata) == 0:
        print("ERROR: Could not load gauge metadata")
        return
    
    # Load MOSART data
    print("\n2. Loading MOSART discharge data...")
    
    # Try daily first, fall back to monthly
    mosart_ds = load_mosart_daily_data(START_YEAR, END_YEAR)
    if mosart_ds is None:
        print("  Falling back to monthly data...")
        mosart_ds = load_mosart_monthly_data(START_YEAR, END_YEAR)
    
    if mosart_ds is None:
        print("ERROR: Could not load MOSART data")
        return
    
    # Process all gauges
    print("\n3. Processing all gauges...")
    metrics_df, fdc_data = process_all_gauges(metadata, mosart_ds)
    
    if len(metrics_df) == 0:
        print("ERROR: No gauges processed successfully")
        return
    
    print(f"\nSuccessfully processed {len(metrics_df)} gauges")
    
    # Save results
    print("\n4. Saving results...")
    save_metrics_csv(metrics_df)
    save_fdc_percentiles_csv(fdc_data)
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    create_fdc_map_and_panels(metrics_df, fdc_data, mosart_ds)
    create_summary_statistics_figure(metrics_df)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()