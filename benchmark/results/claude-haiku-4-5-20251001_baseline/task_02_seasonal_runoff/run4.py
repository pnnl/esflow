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
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_02_seasonal_runoff/run4_output"
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

def compute_runoff_climatology(ds):
    """Extract QRUNOFF and compute climatological statistics."""
    print("Computing runoff climatology...")
    
    if 'QRUNOFF' not in ds.variables:
        raise ValueError("QRUNOFF not found in dataset")
    
    qrunoff = ds['QRUNOFF']  # mm/s
    
    # Convert to mm/day for easier interpretation
    qrunoff_daily = qrunoff * 86400  # mm/day
    
    # Compute mean over time
    qrunoff_mean = qrunoff_daily.mean(dim='time')
    
    # Compute standard deviation
    qrunoff_std = qrunoff_daily.std(dim='time')
    
    return qrunoff_mean, qrunoff_std, qrunoff_daily

def compute_area_weighted_mean(data, lat_var='lat', lon_var='lon'):
    """Compute area-weighted global mean."""
    print("Computing area-weighted global mean...")
    
    # Get latitude and longitude
    if lat_var in data.coords:
        lat = data.coords[lat_var].values
    elif lat_var in data.dims:
        lat = data[lat_var].values
    else:
        raise ValueError(f"Latitude variable '{lat_var}' not found")
    
    if lon_var in data.coords:
        lon = data.coords[lon_var].values
    elif lon_var in data.dims:
        lon = data[lon_var].values
    else:
        raise ValueError(f"Longitude variable '{lon_var}' not found")
    
    # Handle 2D lat/lon arrays
    if lat.ndim == 2 and lon.ndim == 2:
        # Already 2D, compute area weights
        dlat = np.abs(np.gradient(lat, axis=0))
        dlon = np.abs(np.gradient(lon, axis=1))
        weights = np.cos(np.radians(lat)) * dlat * dlon
    else:
        # 1D arrays, create 2D weights
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
        dlat = np.abs(np.gradient(lat))
        dlon = np.abs(np.gradient(lon))
        dlat_2d, dlon_2d = np.meshgrid(dlat, dlon, indexing='ij')
        weights = np.cos(np.radians(lat_2d)) * dlat_2d * dlon_2d
    
    # Normalize weights
    weights = weights / np.nansum(weights)
    
    # Compute weighted mean
    valid_data = data.where(np.isfinite(data), drop=False)
    weighted_mean = np.nansum(valid_data.values * weights)
    
    return weighted_mean, weights

def save_netcdf_output(qrunoff_mean, qrunoff_std, output_dir):
    """Save runoff climatology to NetCDF."""
    print("Saving NetCDF output...")
    
    try:
        # Create dataset
        ds_out = xr.Dataset({
            'QRUNOFF_mean': qrunoff_mean,
            'QRUNOFF_std': qrunoff_std,
        })
        
        # Add attributes
        ds_out['QRUNOFF_mean'].attrs['long_name'] = 'Mean runoff (1985-1989)'
        ds_out['QRUNOFF_mean'].attrs['units'] = 'mm/day'
        ds_out['QRUNOFF_std'].attrs['long_name'] = 'Std dev of runoff (1985-1989)'
        ds_out['QRUNOFF_std'].attrs['units'] = 'mm/day'
        
        output_file = os.path.join(output_dir, 'runoff_climatology_1985-1989.nc')
        ds_out.to_netcdf(output_file)
        print(f"Saved NetCDF to {output_file}")
        
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

def save_statistics_csv(qrunoff_mean, global_mean, output_dir):
    """Save global statistics to CSV."""
    print("Saving statistics CSV...")
    
    try:
        stats_dict = {
            'Statistic': [
                'Global Mean',
                'Global Median',
                'Global Std Dev',
                'Global Min',
                'Global Max',
                'Valid Grid Cells'
            ],
            'Value': [
                float(global_mean),
                float(np.nanmedian(qrunoff_mean.values)),
                float(np.nanstd(qrunoff_mean.values)),
                float(np.nanmin(qrunoff_mean.values)),
                float(np.nanmax(qrunoff_mean.values)),
                int(np.sum(np.isfinite(qrunoff_mean.values)))
            ],
            'Units': [
                'mm/day',
                'mm/day',
                'mm/day',
                'mm/day',
                'mm/day',
                'count'
            ]
        }
        
        df_stats = pd.DataFrame(stats_dict)
        output_file = os.path.join(output_dir, 'runoff_statistics_1985-1989.csv')
        df_stats.to_csv(output_file, index=False)
        print(f"Saved statistics to {output_file}")
        
    except Exception as e:
        print(f"Error saving statistics CSV: {e}")

def create_map_visualization(qrunoff_mean, global_mean, output_dir):
    """Create map of mean runoff with statistics overlay."""
    print("Creating map visualization...")
    
    try:
        # Create figure with map projection
        fig = plt.figure(figsize=(16, 10))
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Add coastlines and borders
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        # Get data for plotting
        data = qrunoff_mean.values
        
        # Handle coordinates
        if 'lat' in qrunoff_mean.coords and 'lon' in qrunoff_mean.coords:
            lat = qrunoff_mean.coords['lat'].values
            lon = qrunoff_mean.coords['lon'].values
        else:
            lat = qrunoff_mean.lat.values if hasattr(qrunoff_mean, 'lat') else np.arange(data.shape[0])
            lon = qrunoff_mean.lon.values if hasattr(qrunoff_mean, 'lon') else np.arange(data.shape[1])
        
        # Create 2D mesh
        if lat.ndim == 1 and lon.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lon, lat)
        else:
            lon_2d, lat_2d = lon, lat
        
        # Plot data with log scale for better visualization
        # Mask zero and negative values
        data_masked = np.ma.masked_where(data <= 0, data)
        
        # Use log scale for better visualization of small values
        im = ax.pcolormesh(
            lon_2d, lat_2d, data_masked,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=LogNorm(vmin=np.nanpercentile(data_masked, 1), 
                        vmax=np.nanpercentile(data_masked, 99)),
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Runoff (mm/day, log scale)', fontsize=12)
        
        # Add statistics text box
        stats_text = (
            f'Global Mean: {global_mean:.4f} mm/day\n'
            f'Median: {np.nanmedian(data):.4f} mm/day\n'
            f'Std Dev: {np.nanstd(data):.4f} mm/day\n'
            f'Min: {np.nanmin(data):.4f} mm/day\n'
            f'Max: {np.nanmax(data):.4f} mm/day\n'
            f'Period: 1985-1989'
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        ax.set_title('E3SM ELM Climatological Mean Runoff (1985-1989)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'runoff_map_1985-1989.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved map to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating map visualization: {e}")
        import traceback
        traceback.print_exc()

def create_histogram_plot(qrunoff_daily, output_dir):
    """Create histogram of runoff values."""
    print("Creating histogram plot...")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Flatten data
        data_flat = qrunoff_daily.values.flatten()
        data_valid = data_flat[np.isfinite(data_flat)]
        
        # Linear scale histogram
        axes[0].hist(data_valid, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Runoff (mm/day)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Runoff Distribution (Linear Scale)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Log scale histogram (for positive values only)
        data_positive = data_valid[data_valid > 0]
        if len(data_positive) > 0:
            axes[1].hist(np.log10(data_positive), bins=100, color='coral', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('log10(Runoff) [mm/day]', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title('Runoff Distribution (Log Scale)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'runoff_histogram_1985-1989.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating histogram: {e}")

def main():
    """Main execution function."""
    print("=" * 70)
    print("E3SM ELM Runoff Climatology Analysis (1985-1989)")
    print("=" * 70)
    
    try:
        # Load data
        ds = load_elm_data(DATA_DIR, CASE_NAME, START_YEAR, END_YEAR)
        print(f"Dataset shape: {ds.dims}")
        print(f"Variables: {list(ds.data_vars)}")
        
        # Compute climatology
        qrunoff_mean, qrunoff_std, qrunoff_daily = compute_runoff_climatology(ds)
        print(f"Mean runoff shape: {qrunoff_mean.shape}")
        print(f"Mean runoff range: {float(np.nanmin(qrunoff_mean)):.6f} to {float(np.nanmax(qrunoff_mean)):.6f} mm/day")
        
        # Compute area-weighted global mean
        global_mean, weights = compute_area_weighted_mean(qrunoff_mean)
        print(f"Area-weighted global mean runoff: {global_mean:.6f} mm/day")
        
        # Save outputs
        save_netcdf_output(qrunoff_mean, qrunoff_std, OUTPUT_DIR)
        save_statistics_csv(qrunoff_mean, global_mean, OUTPUT_DIR)
        create_map_visualization(qrunoff_mean, global_mean, OUTPUT_DIR)
        create_histogram_plot(qrunoff_daily, OUTPUT_DIR)
        
        print("=" * 70)
        print("Analysis complete!")
        print(f"All outputs saved to: {OUTPUT_DIR}")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()