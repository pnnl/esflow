#!/usr/bin/env python
"""
Benchmark E3SM land model evapotranspiration against MODIS satellite observations.
Uses 5-year climatology from 1985-1989.
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats
from urllib.request import urlretrieve
import tempfile

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_03_et_benchmark/run4_output"
E3SM_DATA_DIR = "./data/sample/e3sm/"
CASE_NAME = "sample.v3.LR.historical"
YEARS = [1985, 1986, 1987, 1988, 1989]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_ilamb_data(variable, dataset, filename):
    """Download ILAMB observation data."""
    url = f"https://www.ilamb.org/ILAMB-Data/DATA/{variable}/{dataset}/{filename}"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        print(f"Downloading {url}...")
        urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def load_elm_data(case_name, years):
    """Load ELM data for specified years and compute ET."""
    elm_files = []
    for year in years:
        for month in range(1, 13):
            pattern = os.path.join(
                E3SM_DATA_DIR, "lnd",
                f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc"
            )
            matches = glob.glob(pattern)
            elm_files.extend(matches)
    
    if not elm_files:
        print(f"No ELM files found matching pattern")
        return None
    
    elm_files.sort()
    print(f"Found {len(elm_files)} ELM files")
    
    try:
        # Load all files
        ds_list = [xr.open_dataset(f) for f in elm_files]
        ds = xr.concat(ds_list, dim='time')
        
        # ET is sum of QVEGE, QVEGT, QSOIL (in mm/s)
        # Convert to mm/day for easier interpretation
        et = (ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']) * 86400  # mm/day
        
        # Close individual datasets
        for d in ds_list:
            d.close()
        
        return et
    except Exception as e:
        print(f"Error loading ELM data: {e}")
        return None

def load_modis_data(filepath):
    """Load MODIS ET data."""
    try:
        ds = xr.open_dataset(filepath)
        
        # Extract ET variable (should be 'et' based on task description)
        if 'et' in ds.data_vars:
            et = ds['et']
        elif 'evspsbl' in ds.data_vars:
            et = ds['evspsbl']
        else:
            print(f"Available variables: {list(ds.data_vars)}")
            et = ds[list(ds.data_vars)[0]]
        
        return et
    except Exception as e:
        print(f"Error loading MODIS data: {e}")
        return None

def compute_climatology(data):
    """Compute time-mean climatology."""
    try:
        # Handle different time dimension names
        if 'time' in data.dims:
            climatology = data.mean(dim='time')
        else:
            climatology = data.mean()
        return climatology
    except Exception as e:
        print(f"Error computing climatology: {e}")
        return None

def regrid_to_common_grid(model_et, obs_et):
    """Regrid model and observations to a common grid."""
    try:
        # Use observation grid as target
        obs_grid = obs_et.copy()
        
        # Regrid model to observation grid using nearest neighbor
        model_regridded = model_et.interp_like(obs_grid, method='nearest')
        
        return model_regridded, obs_grid
    except Exception as e:
        print(f"Error regridding: {e}")
        return None, None

def compute_bias_and_correlation(model_et, obs_et):
    """Compute spatial bias and correlation."""
    try:
        # Flatten spatial dimensions
        model_flat = model_et.values.flatten()
        obs_flat = obs_et.values.flatten()
        
        # Remove NaN values
        valid_idx = ~(np.isnan(model_flat) | np.isnan(obs_flat))
        model_valid = model_flat[valid_idx]
        obs_valid = obs_flat[valid_idx]
        
        if len(model_valid) == 0:
            print("No valid data for comparison")
            return None, None, None
        
        # Compute bias
        bias = model_valid - obs_valid
        global_mean_bias = np.mean(bias)
        
        # Compute correlation
        correlation = np.corrcoef(model_valid, obs_valid)[0, 1]
        
        # Compute RMSE
        rmse = np.sqrt(np.mean(bias**2))
        
        return global_mean_bias, correlation, rmse
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None, None, None

def create_bias_map(model_et, obs_et, output_path):
    """Create a map of the bias field."""
    try:
        # Compute bias
        bias = model_et - obs_et
        
        # Create figure with map projection
        fig = plt.figure(figsize=(14, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Plot bias
        im = ax.contourf(
            bias.lon, bias.lat, bias,
            levels=20, cmap='RdBu_r', transform=ccrs.PlateCarree(),
            vmin=-np.nanpercentile(np.abs(bias), 95),
            vmax=np.nanpercentile(np.abs(bias), 95)
        )
        
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('ET Bias (mm/day, Model - Obs)')
        
        plt.title('E3SM ELM vs MODIS ET Bias (1985-1989 Climatology)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved bias map to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating bias map: {e}")

def create_comparison_map(model_et, obs_et, output_dir):
    """Create comparison maps of model, observations, and bias."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Model ET
        im1 = axes[0].contourf(
            model_et.lon, model_et.lat, model_et,
            levels=20, cmap='viridis', transform=ccrs.PlateCarree()
        )
        axes[0].coastlines()
        axes[0].set_title('E3SM ELM ET')
        cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05)
        cbar1.set_label('mm/day')
        
        # Observations
        im2 = axes[1].contourf(
            obs_et.lon, obs_et.lat, obs_et,
            levels=20, cmap='viridis', transform=ccrs.PlateCarree()
        )
        axes[1].coastlines()
        axes[1].set_title('MODIS ET')
        cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05)
        cbar2.set_label('mm/day')
        
        # Bias
        bias = model_et - obs_et
        vmax = np.nanpercentile(np.abs(bias), 95)
        im3 = axes[2].contourf(
            bias.lon, bias.lat, bias,
            levels=20, cmap='RdBu_r', transform=ccrs.PlateCarree(),
            vmin=-vmax, vmax=vmax
        )
        axes[2].coastlines()
        axes[2].set_title('Bias (Model - Obs)')
        cbar3 = plt.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.05)
        cbar3.set_label('mm/day')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'et_comparison_maps.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison maps to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison maps: {e}")

def create_scatter_plot(model_et, obs_et, output_dir):
    """Create scatter plot of model vs observations."""
    try:
        model_flat = model_et.values.flatten()
        obs_flat = obs_et.values.flatten()
        
        valid_idx = ~(np.isnan(model_flat) | np.isnan(obs_flat))
        model_valid = model_flat[valid_idx]
        obs_valid = obs_flat[valid_idx]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(obs_valid, model_valid, alpha=0.3, s=1)
        
        # Add 1:1 line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        
        ax.set_xlabel('MODIS ET (mm/day)')
        ax.set_ylabel('E3SM ELM ET (mm/day)')
        ax.set_title('E3SM ELM vs MODIS ET (1985-1989 Climatology)')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        corr = np.corrcoef(model_valid, obs_valid)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'et_scatter_plot.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating scatter plot: {e}")

def save_results_to_netcdf(model_et, obs_et, bias, output_dir):
    """Save results to NetCDF file."""
    try:
        ds = xr.Dataset({
            'model_et': model_et,
            'obs_et': obs_et,
            'bias': bias
        })
        
        output_path = os.path.join(output_dir, 'et_benchmark_results.nc')
        ds.to_netcdf(output_path)
        print(f"Saved NetCDF results to {output_path}")
        
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

def main():
    """Main analysis function."""
    print("=" * 70)
    print("E3SM ELM Evapotranspiration Benchmark against MODIS")
    print("=" * 70)
    
    # Step 1: Download MODIS data
    print("\n[1/5] Downloading MODIS ET data from ILAMB...")
    modis_file = download_ilamb_data('evspsbl', 'MODIS', 'et_0.5x0.5.nc')
    
    if modis_file is None or not os.path.exists(modis_file):
        print("Warning: Could not download MODIS data. Attempting to use local copy if available...")
        # Try to find local copy
        local_modis = os.path.join(OUTPUT_DIR, 'et_0.5x0.5.nc')
        if os.path.exists(local_modis):
            modis_file = local_modis
        else:
            print("Error: MODIS data not available")
            return
    
    # Step 2: Load ELM data
    print("\n[2/5] Loading E3SM ELM data (1985-1989)...")
    elm_et = load_elm_data(CASE_NAME, YEARS)
    if elm_et is None:
        print("Error: Could not load ELM data")
        return
    print(f"ELM ET shape: {elm_et.shape}")
    
    # Step 3: Load MODIS data
    print("\n[3/5] Loading MODIS ET data...")
    modis_et_raw = load_modis_data(modis_file)
    if modis_et_raw is None:
        print("Error: Could not load MODIS data")
        return
    print(f"MODIS ET shape: {modis_et_raw.shape}")
    
    # Step 4: Compute climatologies
    print("\n[4/5] Computing climatologies...")
    elm_et_clim = compute_climatology(elm_et)
    modis_et_clim = compute_climatology(modis_et_raw)
    
    if elm_et_clim is None or modis_et_clim is None:
        print("Error: Could not compute climatologies")
        return
    
    print(f"ELM ET climatology shape: {elm_et_clim.shape}")
    print(f"MODIS ET climatology shape: {modis_et_clim.shape}")
    
    # Step 5: Regrid to common grid
    print("\n[5/5] Regridding to common grid...")
    elm_et_regrid, modis_et_regrid = regrid_to_common_grid(elm_et_clim, modis_et_clim)
    
    if elm_et_regrid is None:
        print("Error: Could not regrid data")
        return
    
    # Compute metrics
    print("\nComputing bias and correlation metrics...")
    global_mean_bias, correlation, rmse = compute_bias_and_correlation(
        elm_et_regrid, modis_et_regrid
    )
    
    # Compute bias field
    bias = elm_et_regrid - modis_et_regrid
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Global Mean Bias (Model - Obs): {global_mean_bias:.4f} mm/day")
    print(f"Spatial Correlation: {correlation:.4f}")
    print(f"RMSE: {rmse:.4f} mm/day")
    print(f"Model ET Mean: {elm_et_regrid.mean().values:.4f} mm/day")
    print(f"Obs ET Mean: {modis_et_regrid.mean().values:.4f} mm/day")
    print("=" * 70)
    
    # Save results to CSV
    try:
        results_df = pd.DataFrame({
            'Metric': ['Global Mean Bias (mm/day)', 'Spatial Correlation', 'RMSE (mm/day)',
                      'Model ET Mean (mm/day)', 'Obs ET Mean (mm/day)'],
            'Value': [global_mean_bias, correlation, rmse,
                     elm_et_regrid.mean().values, modis_et_regrid.mean().values]
        })
        
        csv_path = os.path.join(OUTPUT_DIR, 'et_benchmark_metrics.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        create_bias_map(elm_et_regrid, modis_et_regrid, 
                       os.path.join(OUTPUT_DIR, 'et_bias_map.png'))
    except Exception as e:
        print(f"Error creating bias map: {e}")
    
    try:
        create_comparison_map(elm_et_regrid, modis_et_regrid, OUTPUT_DIR)
    except Exception as e:
        print(f"Error creating comparison maps: {e}")
    
    try:
        create_scatter_plot(elm_et_regrid, modis_et_regrid, OUTPUT_DIR)
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
    
    # Save NetCDF results
    try:
        save_results_to_netcdf(elm_et_regrid, modis_et_regrid, bias, OUTPUT_DIR)
    except Exception as e:
        print(f"Error saving NetCDF: {e}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()