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
import cartopy.crs as ccrs
from datetime import datetime

# Configuration
DATA_DIR = "./data/sample/e3sm/lnd"
CASE_NAME = "sample.v3.LR.historical"
START_YEAR = 1985
END_YEAR = 1989
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_02_seasonal_runoff/run1_debug/v2/output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_elm_data(data_dir, case_name, start_year, end_year):
    """Load ELM monthly data for specified years."""
    print(f"Loading ELM data from {data_dir}")
    
    # Find all monthly files in the date range
    all_files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            pattern = os.path.join(
                data_dir,
                f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc"
            )
            matching_files = glob.glob(pattern)
            all_files.extend(matching_files)
    
    if not all_files:
        raise FileNotFoundError(
            f"No ELM files found matching pattern: {case_name}.elm.h0.YYYY-MM.nc"
        )
    
    all_files.sort()
    print(f"Found {len(all_files)} files")
    
    # Load all files
    datasets = []
    for filepath in all_files:
        try:
            ds = xr.open_dataset(filepath)
            datasets.append(ds)
            print(f"  Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No datasets loaded successfully")
    
    # Concatenate along time dimension
    data = xr.concat(datasets, dim='time')
    return data

def compute_area_weights(ds):
    """Compute area weights for grid cells."""
    # Try different possible area variable names
    area_names = ['area', 'AREA', 'cell_area', 'CELL_AREA']
    area = None
    
    for name in area_names:
        if name in ds.data_vars or name in ds.coords:
            area = ds[name]
            print(f"Using area variable: {name}")
            break
    
    if area is None:
        # Compute area from lat/lon if available
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lat = ds.coords['lat'].values
            lon = ds.coords['lon'].values
            
            # Create 2D lat/lon grids
            if lat.ndim == 1 and lon.ndim == 1:
                lon_2d, lat_2d = np.meshgrid(lon, lat)
            else:
                lat_2d = lat
                lon_2d = lon
            
            # Compute area using lat/lon spacing
            # Approximate: area = R^2 * dlat * dlon * cos(lat)
            R = 6.371e6  # Earth radius in meters
            dlat = np.abs(np.gradient(lat_2d, axis=0))
            dlon = np.abs(np.gradient(lon_2d, axis=1))
            area = R**2 * np.radians(dlat) * np.radians(dlon) * np.cos(np.radians(lat_2d))
            area = xr.DataArray(area, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
            print("Computed area from lat/lon spacing")
        else:
            print("Warning: Could not determine grid cell areas, using uniform weights")
            area = None
    
    return area

def extract_and_process_runoff(data):
    """Extract QRUNOFF and compute climatological mean."""
    print("\nProcessing QRUNOFF data...")
    
    # Check for QRUNOFF variable
    if 'QRUNOFF' not in data.data_vars:
        raise ValueError(f"QRUNOFF not found in dataset. Available variables: {list(data.data_vars.keys())}")
    
    qrunoff = data['QRUNOFF']  # mm/s
    print(f"QRUNOFF shape: {qrunoff.shape}")
    print(f"QRUNOFF units: mm/s (from ELM)")
    
    # Convert mm/s to mm/day for climatological analysis
    qrunoff_daily = qrunoff * 86400  # seconds per day
    
    # Compute climatological mean (mean over all time steps)
    qrunoff_mean = qrunoff_daily.mean(dim='time')
    print(f"Mean QRUNOFF: {float(qrunoff_mean.mean()):.6f} mm/day")
    
    return qrunoff_mean, qrunoff_daily

def compute_global_statistics(qrunoff_mean, area):
    """Compute area-weighted global mean runoff."""
    print("\nComputing global statistics...")
    
    if area is not None:
        # Area-weighted mean
        weights = area / area.sum()
        global_mean = (qrunoff_mean * weights).sum()
        print(f"Area-weighted global mean runoff: {float(global_mean):.6f} mm/day")
    else:
        # Simple mean
        global_mean = qrunoff_mean.mean()
        print(f"Simple global mean runoff: {float(global_mean):.6f} mm/day")
    
    # Additional statistics
    global_min = float(qrunoff_mean.min())
    global_max = float(qrunoff_mean.max())
    global_std = float(qrunoff_mean.std())
    
    stats = {
        'global_mean': float(global_mean),
        'global_min': global_min,
        'global_max': global_max,
        'global_std': global_std,
        'period': f"{START_YEAR}-{END_YEAR}"
    }
    
    print(f"  Min: {global_min:.6f} mm/day")
    print(f"  Max: {global_max:.6f} mm/day")
    print(f"  Std: {global_std:.6f} mm/day")
    
    return stats

def create_map(qrunoff_mean, stats, output_dir):
    """Create map visualization of mean runoff."""
    print("\nCreating map visualization...")
    
    try:
        fig = plt.figure(figsize=(14, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Plot data
        im = ax.contourf(
            qrunoff_mean.lon, qrunoff_mean.lat, qrunoff_mean,
            levels=20, cmap='viridis', transform=ccrs.PlateCarree()
        )
        
        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Runoff (mm/day)', fontsize=11)
        
        # Title with statistics
        title = f"Climatological Mean Runoff ({START_YEAR}-{END_YEAR})\n"
        title += f"Global Mean: {stats['global_mean']:.4f} mm/day | "
        title += f"Range: [{stats['global_min']:.4f}, {stats['global_max']:.4f}] mm/day"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Save figure
        output_file = os.path.join(output_dir, "runoff_climatology_map.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved map to: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating map: {e}")

def save_outputs(qrunoff_mean, stats, output_dir):
    """Save outputs to files."""
    print("\nSaving outputs...")
    
    # Save statistics to CSV
    try:
        stats_file = os.path.join(output_dir, "runoff_statistics.csv")
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(stats_file, index=False)
        print(f"Saved statistics to: {stats_file}")
    except Exception as e:
        print(f"Error saving statistics: {e}")
    
    # Save mean runoff field to NetCDF
    try:
        output_file = os.path.join(output_dir, "runoff_climatology.nc")
        qrunoff_mean.to_netcdf(output_file)
        print(f"Saved runoff field to: {output_file}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}")
    
    # Save mean runoff field to CSV (flatten the 2D array)
    try:
        output_file = os.path.join(output_dir, "runoff_climatology.csv")
        # Flatten the 2D data
        lat_vals = qrunoff_mean.lat.values
        lon_vals = qrunoff_mean.lon.values
        data_vals = qrunoff_mean.values
        
        # Create lists for output
        lats = []
        lons = []
        runoffs = []
        
        for i, lat in enumerate(lat_vals):
            for j, lon in enumerate(lon_vals):
                lats.append(lat)
                lons.append(lon)
                runoffs.append(data_vals[i, j])
        
        df = pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'runoff_mm_day': runoffs
        })
        df.to_csv(output_file, index=False)
        print(f"Saved runoff field to: {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def main():
    """Main execution function."""
    print("=" * 70)
    print("E3SM ELM Climatological Runoff Analysis")
    print("=" * 70)
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    try:
        # Load data
        data = load_elm_data(DATA_DIR, CASE_NAME, START_YEAR, END_YEAR)
        
        # Compute area weights
        area = compute_area_weights(data)
        
        # Extract and process runoff
        qrunoff_mean, qrunoff_daily = extract_and_process_runoff(data)
        
        # Compute global statistics
        stats = compute_global_statistics(qrunoff_mean, area)
        
        # Create visualization
        create_map(qrunoff_mean, stats, OUTPUT_DIR)
        
        # Save outputs
        save_outputs(qrunoff_mean, stats, OUTPUT_DIR)
        
        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()