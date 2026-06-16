#!/usr/bin/env python
"""
Extract climatological mean runoff from E3SM ELM output (1985-1989).
Compute area-weighted global mean and produce visualization.
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
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_02_seasonal_runoff/run2_output"
START_YEAR = 1985
END_YEAR = 1989

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_elm_data(data_dir, case_name, start_year, end_year):
    """Load ELM monthly data for specified years."""
    print(f"Loading ELM data from {data_dir}...")
    
    # Find all h0 files in the date range
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
        raise FileNotFoundError(f"No ELM files found matching pattern in {data_dir}")
    
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

def compute_global_mean_runoff(ds):
    """Compute area-weighted global mean runoff."""
    print("Computing global mean runoff...")
    
    # QRUNOFF is in mm/s, convert to mm/day for easier interpretation
    if 'QRUNOFF' not in ds.variables:
        raise ValueError("QRUNOFF variable not found in dataset")
    
    qrunoff = ds['QRUNOFF']  # mm/s
    qrunoff_daily = qrunoff * 86400  # Convert to mm/day
    
    # Get grid cell areas
    if 'area' in ds.variables:
        area = ds['area']
    elif 'AREA' in ds.variables:
        area = ds['AREA']
    else:
        # Compute area from lat/lon if not available
        print("Computing grid cell areas from lat/lon...")
        lat = ds['lat'].values
        lon = ds['lon'].values
        
        # Simple area weighting using cos(latitude)
        dlat = np.abs(np.gradient(lat))
        dlon = np.abs(np.gradient(lon))
        
        # Create 2D grid
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
        area = np.cos(np.radians(lat_2d)) * dlat[:, np.newaxis] * dlon[np.newaxis, :]
        area = xr.DataArray(area, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
    
    # Compute time mean first
    qrunoff_mean = qrunoff_daily.mean(dim='time')
    
    # Compute area-weighted global mean
    # Handle NaN values
    valid_mask = ~np.isnan(qrunoff_mean.values)
    
    if 'lat' in qrunoff_mean.dims and 'lon' in qrunoff_mean.dims:
        # Weight by area and latitude
        weights = area / area.sum()
        global_mean = (qrunoff_mean * weights).sum()
    else:
        # Fallback: simple mean
        global_mean = qrunoff_mean.mean()
    
    return qrunoff_mean, global_mean, qrunoff_daily

def compute_climatological_stats(qrunoff_daily):
    """Compute climatological statistics."""
    print("Computing climatological statistics...")
    
    # Time mean
    time_mean = qrunoff_daily.mean(dim='time')
    
    # Spatial statistics
    valid_data = time_mean.values[~np.isnan(time_mean.values)]
    
    stats_dict = {
        'global_mean': float(np.nanmean(time_mean.values)),
        'global_median': float(np.nanmedian(time_mean.values)),
        'global_std': float(np.nanstd(time_mean.values)),
        'global_min': float(np.nanmin(time_mean.values)),
        'global_max': float(np.nanmax(time_mean.values)),
        'valid_cells': int(np.sum(~np.isnan(time_mean.values))),
        'total_cells': int(time_mean.size),
    }
    
    return time_mean, stats_dict

def create_runoff_map(qrunoff_mean, stats_dict, output_dir):
    """Create map visualization of mean runoff."""
    print("Creating runoff map...")
    
    try:
        fig = plt.figure(figsize=(16, 10))
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Plot data
        qrunoff_plot = qrunoff_mean.copy()
        
        # Use log scale for better visualization
        vmin = np.nanpercentile(qrunoff_plot.values, 1)
        vmax = np.nanpercentile(qrunoff_plot.values, 99)
        vmin = max(vmin, 0.001)  # Avoid log(0)
        
        im = ax.contourf(
            qrunoff_plot.lon.values,
            qrunoff_plot.lat.values,
            qrunoff_plot.values,
            levels=20,
            cmap='YlGnBu',
            transform=ccrs.PlateCarree(),
            norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        
        # Add coastlines and features
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Mean Runoff (mm/day)', fontsize=12)
        
        # Title and statistics text
        title = f"Climatological Mean Runoff (1985-1989)\nE3SM ELM QRUNOFF"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics box
        stats_text = (
            f"Global Mean: {stats_dict['global_mean']:.4f} mm/day\n"
            f"Global Median: {stats_dict['global_median']:.4f} mm/day\n"
            f"Global Std Dev: {stats_dict['global_std']:.4f} mm/day\n"
            f"Min: {stats_dict['global_min']:.4f} mm/day\n"
            f"Max: {stats_dict['global_max']:.4f} mm/day\n"
            f"Valid Cells: {stats_dict['valid_cells']}/{stats_dict['total_cells']}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=props,
            family='monospace'
        )
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'runoff_map.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved map to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating map: {e}")

def save_netcdf_output(qrunoff_mean, output_dir):
    """Save mean runoff field to NetCDF."""
    print("Saving NetCDF output...")
    
    try:
        output_file = os.path.join(output_dir, 'runoff_climatology_1985-1989.nc')
        
        # Create dataset
        ds_out = xr.Dataset(
            {
                'QRUNOFF': qrunoff_mean,
            },
            attrs={
                'title': 'Climatological Mean Runoff (1985-1989)',
                'source': 'E3SM ELM',
                'case': 'sample.v3.LR.historical',
                'period': '1985-1989',
                'units': 'mm/day',
            }
        )
        
        ds_out.to_netcdf(output_file)
        print(f"Saved NetCDF to {output_file}")
        
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

def save_statistics_csv(stats_dict, output_dir):
    """Save statistics to CSV."""
    print("Saving statistics...")
    
    try:
        output_file = os.path.join(output_dir, 'runoff_statistics.csv')
        
        stats_df = pd.DataFrame([stats_dict])
        stats_df.to_csv(output_file, index=False)
        print(f"Saved statistics to {output_file}")
        
    except Exception as e:
        print(f"Error saving statistics: {e}")

def main():
    """Main execution."""
    print("=" * 70)
    print("E3SM ELM Climatological Mean Runoff Analysis (1985-1989)")
    print("=" * 70)
    
    try:
        # Load data
        ds = load_elm_data(DATA_DIR, CASE_NAME, START_YEAR, END_YEAR)
        print(f"Loaded dataset with shape: {ds.dims}")
        print(f"Variables: {list(ds.data_vars)}")
        
        # Compute runoff statistics
        qrunoff_mean, global_mean, qrunoff_daily = compute_global_mean_runoff(ds)
        print(f"Global mean runoff: {float(global_mean):.6f} mm/day")
        
        # Compute climatological statistics
        time_mean, stats_dict = compute_climatological_stats(qrunoff_daily)
        
        # Print statistics
        print("\n" + "=" * 70)
        print("RUNOFF STATISTICS (1985-1989)")
        print("=" * 70)
        for key, value in stats_dict.items():
            print(f"{key:20s}: {value}")
        print("=" * 70)
        
        # Create visualizations
        create_runoff_map(time_mean, stats_dict, OUTPUT_DIR)
        
        # Save outputs
        save_netcdf_output(time_mean, OUTPUT_DIR)
        save_statistics_csv(stats_dict, OUTPUT_DIR)
        
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()