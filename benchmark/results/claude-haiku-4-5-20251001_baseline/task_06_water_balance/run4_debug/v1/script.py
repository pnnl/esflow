#!/usr/bin/env python
"""
E3SM Land Model Water Balance Closure Analysis
Analyzes monthly ELM output for 1985-1989 to check water balance closure.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "./data/sample/e3sm"
OBS_DIR = "./data/sample/obs"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_06_water_balance/run4_debug/v1/output"
CASE_NAME = "sample.v3.LR.historical"
COMPONENT = "elm"
START_YEAR = 1985
END_YEAR = 1989

# Basin IDs and names
BASINS = {
    3629000: "Amazon",
    4121801: "Missouri",
    4115200: "Columbia",
    6742900: "Danube",
    2969100: "Mekong",
    1159100: "Orange"
}

def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_elm_data():
    """Load and concatenate ELM monthly data for 1985-1989."""
    print("Loading ELM data...")
    data_list = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            filename = f"{CASE_NAME}.{COMPONENT}.h0.{year:04d}-{month:02d}.nc"
            filepath = os.path.join(DATA_DIR, "lnd", filename)
            
            if os.path.exists(filepath):
                try:
                    ds = xr.open_dataset(filepath)
                    data_list.append(ds)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
            else:
                print(f"File not found: {filepath}")
    
    if not data_list:
        raise FileNotFoundError("No ELM data files found")
    
    # Concatenate along time dimension
    elm_data = xr.concat(data_list, dim='time')
    print(f"Loaded ELM data with shape: {elm_data.dims}")
    return elm_data

def compute_water_balance_components(elm_data):
    """Compute water balance components from ELM data."""
    print("Computing water balance components...")
    
    # Precipitation: RAIN + SNOW (kg/m2/s -> mm/day)
    rain = elm_data.get('RAIN', None)
    snow = elm_data.get('SNOW', None)
    
    if rain is not None and snow is not None:
        precip = (rain + snow) * 86400 / 1000  # Convert to mm/day
    else:
        raise ValueError("RAIN or SNOW not found in ELM data")
    
    # Evapotranspiration: QVEGE + QVEGT + QSOIL (mm/s -> mm/day)
    qvege = elm_data.get('QVEGE', None)
    qvegt = elm_data.get('QVEGT', None)
    qsoil = elm_data.get('QSOIL', None)
    
    if qvege is not None and qvegt is not None and qsoil is not None:
        et = (qvege + qvegt + qsoil) * 86400  # Convert to mm/day
    else:
        raise ValueError("QVEGE, QVEGT, or QSOIL not found in ELM data")
    
    # Total runoff: QRUNOFF (mm/s -> mm/day)
    qrunoff = elm_data.get('QRUNOFF', None)
    if qrunoff is not None:
        runoff = qrunoff * 86400  # Convert to mm/day
    else:
        raise ValueError("QRUNOFF not found in ELM data")
    
    # Water balance residual: P - ET - Q
    residual = precip - et - runoff
    
    return {
        'precip': precip,
        'et': et,
        'runoff': runoff,
        'residual': residual
    }

def compute_climatological_means(components):
    """Compute time-mean fields."""
    print("Computing climatological means...")
    
    clim_means = {}
    for key, data in components.items():
        clim_means[key] = data.mean(dim='time')
    
    return clim_means

def compute_area_weighted_global_mean(data):
    """Compute area-weighted global mean."""
    # Get latitude for area weighting
    if 'lat' in data.dims:
        lat = data.lat.values
        weights = np.cos(np.radians(lat))
        weights = weights / weights.sum()
        
        # Reshape weights to match data dimensions
        if 'lon' in data.dims:
            weights_2d = np.tile(weights[:, np.newaxis], (1, len(data.lon)))
            weights_2d = weights_2d / weights_2d.sum()
            global_mean = (data * weights_2d).sum()
        else:
            global_mean = (data * weights).sum()
    else:
        global_mean = data.mean()
    
    return float(global_mean)

def load_basin_polygons():
    """Load basin polygons from GeoJSON."""
    print("Loading basin polygons...")
    
    geojson_path = os.path.join(OBS_DIR, "basin_polygons.geojson")
    
    if not os.path.exists(geojson_path):
        print(f"Warning: Basin polygons file not found at {geojson_path}")
        return {}
    
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        basins = {}
        for feature in geojson_data.get('features', []):
            grdc_no = feature.get('properties', {}).get('grdc_no')
            if grdc_no and grdc_no in BASINS:
                basins[grdc_no] = feature.get('geometry', {})
        
        return basins
    except Exception as e:
        print(f"Warning: Could not load basin polygons: {e}")
        return {}

def clip_to_basin(data, basin_geometry):
    """Clip data to a basin polygon using simple lat/lon bounding box."""
    if not basin_geometry or basin_geometry.get('type') != 'Polygon':
        return data
    
    try:
        coords = basin_geometry.get('coordinates', [[]])[0]
        if not coords:
            return data
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)
        
        # Clip data
        clipped = data.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
            drop=True
        )
        
        return clipped
    except Exception as e:
        print(f"Warning: Could not clip to basin: {e}")
        return data

def compute_basin_statistics(components, basin_polygons):
    """Compute basin-averaged statistics."""
    print("Computing basin statistics...")
    
    basin_stats = {}
    
    for basin_id, basin_name in BASINS.items():
        basin_stats[basin_id] = {'name': basin_name}
        
        basin_geom = basin_polygons.get(basin_id, {})
        
        for key, data in components.items():
            clipped = clip_to_basin(data, basin_geom)
            
            # Compute area-weighted mean
            if 'lat' in clipped.dims and 'lon' in clipped.dims:
                lat = clipped.lat.values
                lon = clipped.lon.values
                
                # Create 2D weight array
                weights = np.cos(np.radians(lat))
                weights_2d = np.tile(weights[:, np.newaxis], (1, len(lon)))
                weights_2d = weights_2d / weights_2d.sum()
                
                # Apply weights
                weighted_data = clipped * weights_2d
                mean_val = float(weighted_data.sum().values)
            else:
                mean_val = float(clipped.mean().values)
            
            basin_stats[basin_id][key] = mean_val
    
    return basin_stats

def create_global_residual_map(residual_field, basin_polygons):
    """Create global residual map with basin outlines."""
    print("Creating global residual map...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot residual field
    residual_plot = residual_field.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-5,
        vmax=5,
        cbar_kwargs={'label': 'Water Balance Residual (mm/day)', 'shrink': 0.8}
    )
    
    # Add basin outlines
    colors = plt.cm.Set1(np.linspace(0, 1, len(BASINS)))
    
    for idx, (basin_id, basin_name) in enumerate(BASINS.items()):
        basin_geom = basin_polygons.get(basin_id, {})
        
        if basin_geom and basin_geom.get('type') == 'Polygon':
            coords = basin_geom.get('coordinates', [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                ax.plot(lons, lats, color=colors[idx], linewidth=2, 
                       label=basin_name, transform=ccrs.PlateCarree())
    
    ax.coastlines()
    ax.gridlines(draw_labels=True, alpha=0.3)
    ax.set_global()
    ax.legend(loc='lower left', fontsize=10)
    ax.set_title('E3SM Water Balance Residual (P - ET - Q) with Basin Outlines', 
                fontsize=14, fontweight='bold')
    
    return fig

def create_basin_bar_charts(basin_stats):
    """Create per-basin bar charts."""
    print("Creating basin bar charts...")
    
    basin_names = [basin_stats[bid]['name'] for bid in sorted(BASINS.keys())]
    precip_vals = [basin_stats[bid]['precip'] for bid in sorted(BASINS.keys())]
    et_vals = [basin_stats[bid]['et'] for bid in sorted(BASINS.keys())]
    runoff_vals = [basin_stats[bid]['runoff'] for bid in sorted(BASINS.keys())]
    residual_vals = [basin_stats[bid]['residual'] for bid in sorted(BASINS.keys())]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(basin_names))
    width = 0.6
    
    # Precipitation
    axes[0, 0].bar(x, precip_vals, width, color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('Precipitation (mm/day)', fontsize=11)
    axes[0, 0].set_title('Precipitation by Basin', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Evapotranspiration
    axes[0, 1].bar(x, et_vals, width, color='green', alpha=0.8)
    axes[0, 1].set_ylabel('Evapotranspiration (mm/day)', fontsize=11)
    axes[0, 1].set_title('Evapotranspiration by Basin', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Runoff
    axes[1, 0].bar(x, runoff_vals, width, color='navy', alpha=0.8)
    axes[1, 0].set_ylabel('Runoff (mm/day)', fontsize=11)
    axes[1, 0].set_title('Runoff by Basin', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Residual
    colors = ['red' if v < 0 else 'blue' for v in residual_vals]
    axes[1, 1].bar(x, residual_vals, width, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Residual (mm/day)', fontsize=11)
    axes[1, 1].set_title('Water Balance Residual (P - ET - Q) by Basin', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(basin_names, rotation=45, ha='right')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def save_results(components_clim, basin_stats, basin_polygons):
    """Save all results to output directory."""
    print("Saving results...")
    
    # Save global statistics
    global_stats = {}
    for key, data in components_clim.items():
        global_mean = compute_area_weighted_global_mean(data)
        global_stats[key] = global_mean
        print(f"Global mean {key}: {global_mean:.4f} mm/day")
    
    # Save to CSV
    stats_df = pd.DataFrame({
        'Component': ['Precipitation', 'Evapotranspiration', 'Runoff', 'Residual'],
        'Global Mean (mm/day)': [
            global_stats['precip'],
            global_stats['et'],
            global_stats['runoff'],
            global_stats['residual']
        ]
    })
    
    stats_csv = os.path.join(OUTPUT_DIR, 'global_water_balance_stats.csv')
    try:
        stats_df.to_csv(stats_csv, index=False)
        print(f"Saved global statistics to {stats_csv}")
    except Exception as e:
        print(f"Error saving global statistics: {e}")
    
    # Save basin statistics
    basin_df_data = []
    for basin_id in sorted(BASINS.keys()):
        stats = basin_stats[basin_id]
        basin_df_data.append({
            'Basin ID': basin_id,
            'Basin Name': stats['name'],
            'Precipitation (mm/day)': stats['precip'],
            'Evapotranspiration (mm/day)': stats['et'],
            'Runoff (mm/day)': stats['runoff'],
            'Residual (mm/day)': stats['residual']
        })
    
    basin_df = pd.DataFrame(basin_df_data)
    basin_csv = os.path.join(OUTPUT_DIR, 'basin_water_balance_stats.csv')
    try:
        basin_df.to_csv(basin_csv, index=False)
        print(f"Saved basin statistics to {basin_csv}")
    except Exception as e:
        print(f"Error saving basin statistics: {e}")
    
    # Save NetCDF files
    for key, data in components_clim.items():
        nc_file = os.path.join(OUTPUT_DIR, f'{key}_climatological_mean.nc')
        try:
            data.to_netcdf(nc_file)
            print(f"Saved {key} to {nc_file}")
        except Exception as e:
            print(f"Error saving {key} NetCDF: {e}")

def main():
    """Main analysis function."""
    print("=" * 80)
    print("E3SM Land Model Water Balance Closure Analysis")
    print("=" * 80)
    
    try:
        create_output_dir()
        
        # Load data
        elm_data = load_elm_data()
        
        # Compute water balance components
        components = compute_water_balance_components(elm_data)
        
        # Compute climatological means
        components_clim = compute_climatological_means(components)
        
        # Load basin polygons
        basin_polygons = load_basin_polygons()
        
        # Compute basin statistics
        basin_stats = compute_basin_statistics(components_clim, basin_polygons)
        
        # Save results
        save_results(components_clim, basin_stats, basin_polygons)
        
        # Create figures
        print("Creating figures...")
        
        # Global residual map
        try:
            fig_global = create_global_residual_map(components_clim['residual'], basin_polygons)
            global_map_file = os.path.join(OUTPUT_DIR, 'global_residual_map.png')
            fig_global.savefig(global_map_file, dpi=150, bbox_inches='tight')
            print(f"Saved global residual map to {global_map_file}")
            plt.close(fig_global)
        except Exception as e:
            print(f"Error creating global residual map: {e}")
        
        # Basin bar charts
        try:
            fig_basins = create_basin_bar_charts(basin_stats)
            basin_chart_file = os.path.join(OUTPUT_DIR, 'basin_water_balance_charts.png')
            fig_basins.savefig(basin_chart_file, dpi=150, bbox_inches='tight')
            print(f"Saved basin charts to {basin_chart_file}")
            plt.close(fig_basins)
        except Exception as e:
            print(f"Error creating basin charts: {e}")
        
        # Composite figure
        try:
            fig = plt.figure(figsize=(18, 14))
            
            # Top: Global residual map
            ax_map = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
            residual_plot = components_clim['residual'].plot(
                ax=ax_map,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',
                vmin=-5,
                vmax=5,
                cbar_kwargs={'label': 'Water Balance Residual (mm/day)', 'shrink': 0.8}
            )
            
            # Add basin outlines
            colors = plt.cm.Set1(np.linspace(0, 1, len(BASINS)))
            for idx, (basin_id, basin_name) in enumerate(BASINS.items()):
                basin_geom = basin_polygons.get(basin_id, {})
                if basin_geom and basin_geom.get('type') == 'Polygon':
                    coords = basin_geom.get('coordinates', [[]])[0]
                    if coords:
                        lons = [c[0] for c in coords]
                        lats = [c[1] for c in coords]
                        ax_map.plot(lons, lats, color=colors[idx], linewidth=2,
                                   label=basin_name, transform=ccrs.PlateCarree())
            
            ax_map.coastlines()
            ax_map.gridlines(draw_labels=True, alpha=0.3)
            ax_map.set_global()
            ax_map.legend(loc='lower left', fontsize=9, ncol=2)
            ax_map.set_title('E3SM Water Balance Residual (P - ET - Q) with Basin Outlines',
                            fontsize=14, fontweight='bold')
            
            # Bottom: Basin bar charts
            basin_names = [basin_stats[bid]['name'] for bid in sorted(BASINS.keys())]
            precip_vals = [basin_stats[bid]['precip'] for bid in sorted(BASINS.keys())]
            et_vals = [basin_stats[bid]['et'] for bid in sorted(BASINS.keys())]
            runoff_vals = [basin_stats[bid]['runoff'] for bid in sorted(BASINS.keys())]
            residual_vals = [basin_stats[bid]['residual'] for bid in sorted(BASINS.keys())]
            
            ax_bar = plt.subplot(2, 1, 2)
            x = np.arange(len(basin_names))
            width = 0.2
            
            ax_bar.bar(x - 1.5*width, precip_vals, width, label='Precipitation', color='steelblue', alpha=0.8)
            ax_bar.bar(x - 0.5*width, et_vals, width, label='Evapotranspiration', color='green', alpha=0.8)
            ax_bar.bar(x + 0.5*width, runoff_vals, width, label='Runoff', color='navy', alpha=0.8)
            ax_bar.bar(x + 1.5*width, residual_vals, width, label='Residual', color='red', alpha=0.8)
            
            ax_bar.set_ylabel('Water Flux (mm/day)', fontsize=12)
            ax_bar.set_title('Basin-Averaged Water Balance Components', fontsize=12, fontweight='bold')
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(basin_names, rotation=45, ha='right')
            ax_bar.legend(fontsize=10, loc='upper left')
            ax_bar.grid(axis='y', alpha=0.3)
            ax_bar.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            composite_file = os.path.join(OUTPUT_DIR, 'composite_water_balance_analysis.png')
            fig.savefig(composite_file, dpi=150, bbox_inches='tight')
            print(f"Saved composite figure to {composite_file}")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating composite figure: {e}")
        
        print("=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()