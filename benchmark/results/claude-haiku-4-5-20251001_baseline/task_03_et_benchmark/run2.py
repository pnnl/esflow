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
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_03_et_benchmark/run2_output"
E3SM_DATA_DIR = "./data/sample/e3sm"
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
    """Load ELM data for specified years and compute ET climatology."""
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
        
        # Compute ET: sum of QVEGE, QVEGT, QSOIL (mm/s)
        # Convert to mm/day for comparison with observations
        et_components = ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']
        et_mm_day = et_components * 86400  # Convert from mm/s to mm/day
        
        # Compute climatological mean
        et_clim = et_mm_day.mean(dim='time')
        
        print(f"ELM ET climatology shape: {et_clim.shape}")
        print(f"ELM ET climatology range: {float(et_clim.min()):.2f} to {float(et_clim.max()):.2f} mm/day")
        
        return et_clim, ds
    except Exception as e:
        print(f"Error loading ELM data: {e}")
        return None, None

def load_modis_data(modis_file):
    """Load MODIS ET data and compute climatology."""
    try:
        ds = xr.open_dataset(modis_file)
        
        # Extract ET variable (should be 'et' according to instructions)
        if 'et' in ds.data_vars:
            et = ds['et']
        elif 'evspsbl' in ds.data_vars:
            et = ds['evspsbl']
        else:
            print(f"Available variables: {list(ds.data_vars)}")
            return None
        
        # Compute climatological mean
        et_clim = et.mean(dim='time')
        
        print(f"MODIS ET climatology shape: {et_clim.shape}")
        print(f"MODIS ET climatology range: {float(et_clim.min()):.2f} to {float(et_clim.max()):.2f}")
        
        return et_clim
    except Exception as e:
        print(f"Error loading MODIS data: {e}")
        return None

def regrid_to_common_grid(et_model, et_obs):
    """Regrid both datasets to a common grid for comparison."""
    try:
        # Use observation grid as target
        et_model_regridded = et_model.interp_like(et_obs, method='nearest')
        return et_model_regridded, et_obs
    except Exception as e:
        print(f"Error regridding: {e}")
        return None, None

def compute_bias_and_correlation(et_model, et_obs):
    """Compute bias and spatial correlation."""
    try:
        # Flatten arrays for correlation calculation
        model_flat = et_model.values.flatten()
        obs_flat = et_obs.values.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(model_flat) | np.isnan(obs_flat))
        model_valid = model_flat[valid_mask]
        obs_valid = obs_flat[valid_mask]
        
        if len(model_valid) == 0:
            print("No valid data for comparison")
            return None, None, None
        
        # Compute bias
        bias = et_model - et_obs
        global_mean_bias = float(bias.mean())
        
        # Compute spatial correlation
        correlation = float(np.corrcoef(model_valid, obs_valid)[0, 1])
        
        # Compute RMSE
        rmse = float(np.sqrt(np.mean((model_valid - obs_valid) ** 2)))
        
        return global_mean_bias, correlation, rmse, bias
    except Exception as e:
        print(f"Error computing bias and correlation: {e}")
        return None, None, None, None

def plot_bias_map(bias, output_file):
    """Create a map of the bias field."""
    try:
        fig = plt.figure(figsize=(14, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Plot bias
        im = ax.contourf(
            bias.lon, bias.lat, bias,
            levels=20, cmap='RdBu_r', transform=ccrs.PlateCarree(),
            vmin=-np.nanpercentile(np.abs(bias.values), 95),
            vmax=np.nanpercentile(np.abs(bias.values), 95)
        )
        
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('ET Bias (mm/day, Model - Obs)')
        
        plt.title('E3SM ELM vs MODIS ET Bias (1985-1989 Climatology)')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved bias map to {output_file}")
        plt.close()
    except Exception as e:
        print(f"Error plotting bias map: {e}")

def plot_comparison_maps(et_model, et_obs, bias, output_dir):
    """Create comparison maps of model, observations, and bias."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Model ET
        im1 = axes[0].contourf(
            et_model.lon, et_model.lat, et_model,
            levels=20, cmap='viridis', transform=ccrs.PlateCarree()
        )
        axes[0].coastlines()
        axes[0].set_title('E3SM ELM ET')
        cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05)
        cbar1.set_label('mm/day')
        
        # Observations
        im2 = axes[1].contourf(
            et_obs.lon, et_obs.lat, et_obs,
            levels=20, cmap='viridis', transform=ccrs.PlateCarree()
        )
        axes[1].coastlines()
        axes[1].set_title('MODIS ET')
        cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05)
        cbar2.set_label('mm/day')
        
        # Bias
        vmax = np.nanpercentile(np.abs(bias.values), 95)
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
        output_file = os.path.join(output_dir, 'et_comparison_maps.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved comparison maps to {output_file}")
        plt.close()
    except Exception as e:
        print(f"Error creating comparison maps: {e}")

def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    try:
        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def save_bias_to_netcdf(bias, output_file):
    """Save bias field to NetCDF."""
    try:
        bias.to_netcdf(output_file)
        print(f"Saved bias field to {output_file}")
    except Exception as e:
        print(f"Error saving bias to NetCDF: {e}")

def main():
    """Main analysis function."""
    print("=" * 80)
    print("E3SM ELM Evapotranspiration Benchmark against MODIS")
    print("=" * 80)
    
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
    
    # Step 2: Load MODIS data
    print("\n[2/5] Loading MODIS ET climatology...")
    et_obs = load_modis_data(modis_file)
    if et_obs is None:
        print("Error: Could not load MODIS data")
        return
    
    # Step 3: Load E3SM ELM data
    print("\n[3/5] Loading E3SM ELM ET climatology...")
    et_model, ds_elm = load_elm_data(CASE_NAME, YEARS)
    if et_model is None:
        print("Error: Could not load ELM data")
        return
    
    # Step 4: Regrid to common grid
    print("\n[4/5] Regridding to common grid...")
    et_model_regrid, et_obs_regrid = regrid_to_common_grid(et_model, et_obs)
    if et_model_regrid is None:
        print("Error: Could not regrid data")
        return
    
    # Step 5: Compute bias and correlation
    print("\n[5/5] Computing bias and correlation...")
    global_mean_bias, correlation, rmse, bias = compute_bias_and_correlation(
        et_model_regrid, et_obs_regrid
    )
    
    if global_mean_bias is None:
        print("Error: Could not compute bias and correlation")
        return
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Global Mean Bias (Model - Obs): {global_mean_bias:.4f} mm/day")
    print(f"Spatial Correlation: {correlation:.4f}")
    print(f"RMSE: {rmse:.4f} mm/day")
    print(f"Model ET range: {float(et_model_regrid.min()):.2f} to {float(et_model_regrid.max()):.2f} mm/day")
    print(f"Obs ET range: {float(et_obs_regrid.min()):.2f} to {float(et_obs_regrid.max()):.2f} mm/day")
    print("=" * 80)
    
    # Save results
    results = {
        'global_mean_bias_mm_day': global_mean_bias,
        'spatial_correlation': correlation,
        'rmse_mm_day': rmse,
        'model_et_min': float(et_model_regrid.min()),
        'model_et_max': float(et_model_regrid.max()),
        'obs_et_min': float(et_obs_regrid.min()),
        'obs_et_max': float(et_obs_regrid.max()),
        'years': f"{YEARS[0]}-{YEARS[-1]}"
    }
    
    # Save to CSV
    csv_file = os.path.join(OUTPUT_DIR, 'et_benchmark_results.csv')
    save_results_to_csv(results, csv_file)
    
    # Save bias to NetCDF
    bias_nc_file = os.path.join(OUTPUT_DIR, 'et_bias.nc')
    save_bias_to_netcdf(bias, bias_nc_file)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Bias map
    bias_map_file = os.path.join(OUTPUT_DIR, 'et_bias_map.png')
    plot_bias_map(bias, bias_map_file)
    
    # Comparison maps
    plot_comparison_maps(et_model_regrid, et_obs_regrid, bias, OUTPUT_DIR)
    
    # Create scatter plot
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        model_flat = et_model_regrid.values.flatten()
        obs_flat = et_obs_regrid.values.flatten()
        valid_mask = ~(np.isnan(model_flat) | np.isnan(obs_flat))
        
        ax.scatter(obs_flat[valid_mask], model_flat[valid_mask], alpha=0.3, s=1)
        
        # Add 1:1 line
        min_val = min(obs_flat[valid_mask].min(), model_flat[valid_mask].min())
        max_val = max(obs_flat[valid_mask].max(), model_flat[valid_mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
        
        ax.set_xlabel('MODIS ET (mm/day)')
        ax.set_ylabel('E3SM ELM ET (mm/day)')
        ax.set_title(f'ET Comparison (r={correlation:.3f}, bias={global_mean_bias:.3f} mm/day)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        scatter_file = os.path.join(OUTPUT_DIR, 'et_scatter_plot.png')
        plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to {scatter_file}")
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All output files saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()