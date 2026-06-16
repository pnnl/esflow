#!/usr/bin/env python
"""
Extract climatological mean runoff from E3SM ELM output (1985-1989)
and produce a global map with statistics.
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

# Configuration
DATA_DIR = "./data/sample/e3sm/lnd"
CASE_NAME = "sample.v3.LR.historical"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_02_seasonal_runoff/run3_output"
START_YEAR = 1985
END_YEAR = 1989

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_elm_data(data_dir, case_name, start_year, end_year):
    """Load ELM monthly data for specified years."""
    print(f"Loading ELM data from {data_dir}...")
    
    # Find all monthly files in the date range
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            pattern = os.path.join(
                data_dir,
                f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc"
            )
            matching_files = glob.glob(pattern)
            files.extend(matching_files)
    
    if not files:
        raise FileNotFoundError(
            f"No ELM files found matching pattern: {case_name}.elm.h0.YYYY-MM.nc"
        )
    
    files.sort()
    print(f"Found {len(files)} files")
    
    # Load all files
    ds_list = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            ds_list.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if not ds_list:
        raise ValueError("No files could be loaded successfully")
    
    # Concatenate along time dimension
    ds = xr.concat(ds_list, dim='time')
    return ds

def compute_runoff_climatology(ds):
    """Extract QRUNOFF and compute climatological statistics."""
    print("Computing runoff climatology...")
    
    # QRUNOFF is in mm/s, need to convert to mm/day for climatological mean
    if 'QRUNOFF' not in ds.data_vars:
        raise ValueError("QRUNOFF variable not found in dataset")
    
    qrunoff = ds['QRUNOFF']  # mm/s
    
    # Convert to mm/day
    qrunoff_daily = qrunoff * 86400  # seconds per day
    
    # Compute mean over time
    qrunoff_mean = qrunoff_daily.mean(dim='time')
    
    return qrunoff_mean, qrunoff_daily

def compute_area_weighted_mean(data, ds):
    """Compute area-weighted global mean."""
    print("Computing area-weighted global mean...")
    
    # Get grid cell areas if available
    if 'area' in ds.coords:
        area = ds['area']
    elif 'AREA' in ds.data_vars:
        area = ds['AREA'].isel(time=0)
    else:
        # Estimate area from lat/lon spacing
        lat = ds.coords['lat'].values
        lon = ds.coords['lon'].values
        
        # Create 2D lat/lon grids
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        
        # Compute area (simplified - assumes regular grid)
        dlat = np.abs(np.diff(lat)).mean()
        dlon = np.abs(np.diff(lon)).mean()
        
        # Area in m^2 (approximate)
        R_earth = 6.371e6  # meters
        area = (R_earth**2 * np.radians(dlat) * 
                np.radians(dlon) * np.cos(np.radians(lat_2d)))
        area = xr.DataArray(area, coords={'lat': lat, 'lon': lon}, 
                           dims=['lat', 'lon'])
    
    # Compute weighted mean
    valid_mask = ~np.isnan(data.values)
    if valid_mask.sum() == 0:
        return np.nan
    
    weighted_sum = (data * area).sum()
    area_sum = area.where(valid_mask).sum()
    
    global_mean = weighted_sum / area_sum
    
    return float(global_mean.values)

def create_runoff_map(qrunoff_mean, output_dir):
    """Create a map of mean runoff with statistics."""
    print("Creating runoff map...")
    
    try:
        # Create figure with map projection
        fig = plt.figure(figsize=(16, 10))
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Add coastlines and features
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        # Plot data with log scale for better visualization
        # Mask zero and negative values
        data_plot = qrunoff_mean.copy()
        data_plot = data_plot.where(data_plot > 0)
        
        # Use log scale for visualization
        im = ax.pcolormesh(
            qrunoff_mean.lon, qrunoff_mean.lat,
            data_plot,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=LogNorm(vmin=data_plot.min().values, 
                        vmax=data_plot.max().values),
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label('Mean Runoff (mm/day, log scale)', fontsize=12)
        
        # Compute statistics
        valid_data = qrunoff_mean.values[~np.isnan(qrunoff_mean.values)]
        valid_data = valid_data[valid_data > 0]
        
        global_mean = valid_data.mean()
        global_std = valid_data.std()
        global_min = valid_data.min()
        global_max = valid_data.max()
        
        # Add text box with statistics
        stats_text = (
            f'Global Runoff Statistics (1985-1989)\n'
            f'Mean: {global_mean:.4f} mm/day\n'
            f'Std Dev: {global_std:.4f} mm/day\n'
            f'Min: {global_min:.4f} mm/day\n'
            f'Max: {global_max:.4f} mm/day\n'
            f'Valid cells: {len(valid_data)}'
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title('E3SM ELM Climatological Mean Runoff (1985-1989)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, 'runoff_map.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved map to {output_file}")
        plt.close()
        
        return {
            'mean': global_mean,
            'std': global_std,
            'min': global_min,
            'max': global_max,
            'valid_cells': len(valid_data)
        }
        
    except Exception as e:
        print(f"Error creating map: {e}")
        return None

def save_statistics(stats_dict, qrunoff_mean, output_dir):
    """Save statistics to CSV and NetCDF."""
    print("Saving statistics...")
    
    try:
        # Save statistics to CSV
        stats_df = pd.DataFrame({
            'Statistic': ['Mean (mm/day)', 'Std Dev (mm/day)', 
                         'Min (mm/day)', 'Max (mm/day)', 'Valid Cells'],
            'Value': [
                stats_dict['mean'],
                stats_dict['std'],
                stats_dict['min'],
                stats_dict['max'],
                stats_dict['valid_cells']
            ]
        })
        
        csv_file = os.path.join(output_dir, 'runoff_statistics.csv')
        stats_df.to_csv(csv_file, index=False)
        print(f"Saved statistics to {csv_file}")
        
        # Save mean runoff field to NetCDF
        nc_file = os.path.join(output_dir, 'runoff_climatology.nc')
        qrunoff_mean.to_netcdf(nc_file)
        print(f"Saved runoff field to {nc_file}")
        
    except Exception as e:
        print(f"Error saving statistics: {e}")

def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("E3SM ELM Runoff Climatology Analysis")
        print("=" * 60)
        
        # Load data
        ds = load_elm_data(DATA_DIR, CASE_NAME, START_YEAR, END_YEAR)
        print(f"Dataset shape: {ds.dims}")
        print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        
        # Compute climatology
        qrunoff_mean, qrunoff_daily = compute_runoff_climatology(ds)
        print(f"Mean runoff shape: {qrunoff_mean.shape}")
        print(f"Mean runoff range: {qrunoff_mean.min().values:.6f} to "
              f"{qrunoff_mean.max().values:.6f} mm/day")
        
        # Compute area-weighted global mean
        global_mean = compute_area_weighted_mean(qrunoff_mean, ds)
        print(f"Area-weighted global mean: {global_mean:.6f} mm/day")
        
        # Create map
        stats_dict = create_runoff_map(qrunoff_mean, OUTPUT_DIR)
        
        # Save outputs
        if stats_dict:
            save_statistics(stats_dict, qrunoff_mean, OUTPUT_DIR)
        
        print("=" * 60)
        print("Analysis complete!")
        print(f"Output saved to: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in main execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()