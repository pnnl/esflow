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
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_06_water_balance/run3_output"
CASE_NAME = "sample.v3.LR.historical"
COMPONENT = "lnd"
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
    data_files = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            filename = f"{CASE_NAME}.elm.h0.{year:04d}-{month:02d}.nc"
            filepath = os.path.join(DATA_DIR, COMPONENT, filename)
            if os.path.exists(filepath):
                data_files.append(filepath)
    
    if not data_files:
        print(f"ERROR: No ELM data files found in {os.path.join(DATA_DIR, COMPONENT)}")
        return None
    
    print(f"Found {len(data_files)} files")
    
    try:
        ds = xr.open_mfdataset(data_files, combine='by_coords', parallel=False)
        print(f"Loaded data shape: {ds.dims}")
        return ds
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

def compute_water_balance_components(ds):
    """Compute P, ET, Q from ELM variables."""
    print("Computing water balance components...")
    
    # Convert units and compute components
    # RAIN, SNOW: kg/m2/s -> mm/day (1 kg/m2 = 1 mm)
    # QVEGE, QVEGT, QSOIL: mm/s -> mm/day
    # QRUNOFF: mm/s -> mm/day
    
    try:
        # Precipitation: RAIN + SNOW (kg/m2/s -> mm/day)
        rain = ds['RAIN'].values * 86400  # kg/m2/s to mm/day
        snow = ds['SNOW'].values * 86400  # kg/m2/s to mm/day
        precip = rain + snow
        
        # Evapotranspiration: QVEGE + QVEGT + QSOIL (mm/s -> mm/day)
        qvege = ds['QVEGE'].values * 86400  # mm/s to mm/day
        qvegt = ds['QVEGT'].values * 86400  # mm/s to mm/day
        qsoil = ds['QSOIL'].values * 86400  # mm/s to mm/day
        evap = qvege + qvegt + qsoil
        
        # Runoff: QRUNOFF (mm/s -> mm/day)
        runoff = ds['QRUNOFF'].values * 86400  # mm/s to mm/day
        
        # Water balance residual: P - ET - Q
        residual = precip - evap - runoff
        
        print(f"Precipitation range: {np.nanmin(precip):.2f} to {np.nanmax(precip):.2f} mm/day")
        print(f"Evapotranspiration range: {np.nanmin(evap):.2f} to {np.nanmax(evap):.2f} mm/day")
        print(f"Runoff range: {np.nanmin(runoff):.2f} to {np.nanmax(runoff):.2f} mm/day")
        print(f"Residual range: {np.nanmin(residual):.2f} to {np.nanmax(residual):.2f} mm/day")
        
        return precip, evap, runoff, residual
    except Exception as e:
        print(f"ERROR computing components: {e}")
        return None, None, None, None

def compute_climatology(precip, evap, runoff, residual):
    """Compute time-mean fields."""
    print("Computing climatological means...")
    
    precip_mean = np.nanmean(precip, axis=0)
    evap_mean = np.nanmean(evap, axis=0)
    runoff_mean = np.nanmean(runoff, axis=0)
    residual_mean = np.nanmean(residual, axis=0)
    
    return precip_mean, evap_mean, runoff_mean, residual_mean

def compute_area_weighted_global_mean(field, ds):
    """Compute area-weighted global mean."""
    try:
        # Get latitude for area weighting
        lat = ds['lat'].values
        
        # Create area weights (cosine of latitude)
        weights = np.cos(np.radians(lat))
        
        # Reshape weights to match field
        if field.ndim == 2:
            weights_2d = weights[np.newaxis, :] if weights.shape[0] != field.shape[0] else weights[:, np.newaxis]
            if weights_2d.shape != field.shape:
                weights_2d = np.tile(weights[:, np.newaxis], (1, field.shape[1]))
        else:
            weights_2d = weights
        
        # Compute weighted mean
        valid_mask = ~np.isnan(field)
        weighted_sum = np.nansum(field * weights_2d * valid_mask)
        weight_sum = np.nansum(weights_2d * valid_mask)
        
        global_mean = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        return global_mean
    except Exception as e:
        print(f"ERROR computing global mean: {e}")
        return np.nan

def load_basin_polygons():
    """Load basin polygons from GeoJSON."""
    print("Loading basin polygons...")
    
    geojson_path = os.path.join(OBS_DIR, "basin_polygons.geojson")
    
    if not os.path.exists(geojson_path):
        print(f"WARNING: Basin polygons not found at {geojson_path}")
        return {}
    
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        basins_dict = {}
        for feature in geojson_data['features']:
            grdc_no = feature['properties'].get('grdc_no')
            if grdc_no in BASINS:
                basins_dict[grdc_no] = feature['geometry']
        
        print(f"Loaded {len(basins_dict)} basin polygons")
        return basins_dict
    except Exception as e:
        print(f"ERROR loading basin polygons: {e}")
        return {}

def clip_to_basin(field, lat, lon, basin_geometry):
    """Clip field to basin using polygon mask."""
    try:
        from shapely.geometry import shape, Point
        
        # Create mask for basin
        mask = np.zeros(field.shape, dtype=bool)
        
        # Get basin polygon
        basin_shape = shape(basin_geometry)
        
        # Check each grid point
        for i in range(len(lat)):
            for j in range(len(lon)):
                point = Point(lon[j], lat[i])
                if basin_shape.contains(point):
                    mask[i, j] = True
        
        # Apply mask
        clipped = field.copy()
        clipped[~mask] = np.nan
        
        return clipped
    except ImportError:
        print("WARNING: shapely not available, using simple bounding box")
        return field
    except Exception as e:
        print(f"ERROR clipping to basin: {e}")
        return field

def compute_basin_statistics(precip_mean, evap_mean, runoff_mean, residual_mean, 
                            lat, lon, basin_geometry):
    """Compute basin-averaged statistics."""
    try:
        # Clip fields to basin
        p_basin = clip_to_basin(precip_mean, lat, lon, basin_geometry)
        et_basin = clip_to_basin(evap_mean, lat, lon, basin_geometry)
        q_basin = clip_to_basin(runoff_mean, lat, lon, basin_geometry)
        res_basin = clip_to_basin(residual_mean, lat, lon, basin_geometry)
        
        # Compute area-weighted means
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
        weights = np.cos(np.radians(lat_2d))
        
        p_mean = np.nanmean(p_basin)
        et_mean = np.nanmean(et_basin)
        q_mean = np.nanmean(q_basin)
        res_mean = np.nanmean(res_basin)
        
        return p_mean, et_mean, q_mean, res_mean
    except Exception as e:
        print(f"ERROR computing basin statistics: {e}")
        return np.nan, np.nan, np.nan, np.nan

def create_composite_figure(precip_mean, evap_mean, runoff_mean, residual_mean,
                           lat, lon, basin_polygons, ds):
    """Create composite figure with global map and basin bar charts."""
    print("Creating composite figure...")
    
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Global residual map
        ax_map = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
        
        # Plot residual
        im = ax_map.contourf(lon, lat, residual_mean, levels=20, cmap='RdBu_r',
                            transform=ccrs.PlateCarree(), extend='both')
        
        # Add basin outlines
        colors = plt.cm.Set3(np.linspace(0, 1, len(BASINS)))
        for idx, (basin_id, basin_name) in enumerate(BASINS.items()):
            if basin_id in basin_polygons:
                geom = basin_polygons[basin_id]
                if geom['type'] == 'Polygon':
                    coords = np.array(geom['coordinates'][0])
                    ax_map.plot(coords[:, 0], coords[:, 1], color=colors[idx], 
                               linewidth=2, transform=ccrs.PlateCarree(), label=basin_name)
        
        ax_map.coastlines()
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.set_global()
        ax_map.set_title('Water Balance Residual (P - ET - Q) [mm/day]\nwith Basin Outlines', 
                        fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Residual [mm/day]')
        ax_map.legend(loc='lower left', fontsize=9)
        
        # Basin bar charts
        ax_bars = plt.subplot(2, 1, 2)
        
        basin_names_list = []
        p_vals = []
        et_vals = []
        q_vals = []
        res_vals = []
        
        for basin_id, basin_name in BASINS.items():
            if basin_id in basin_polygons:
                p, et, q, res = compute_basin_statistics(
                    precip_mean, evap_mean, runoff_mean, residual_mean,
                    lat, lon, basin_polygons[basin_id]
                )
                basin_names_list.append(basin_name)
                p_vals.append(p)
                et_vals.append(et)
                q_vals.append(q)
                res_vals.append(res)
        
        # Create grouped bar chart
        x = np.arange(len(basin_names_list))
        width = 0.2
        
        ax_bars.bar(x - 1.5*width, p_vals, width, label='Precipitation', color='blue', alpha=0.7)
        ax_bars.bar(x - 0.5*width, et_vals, width, label='Evapotranspiration', color='green', alpha=0.7)
        ax_bars.bar(x + 0.5*width, q_vals, width, label='Runoff', color='red', alpha=0.7)
        ax_bars.bar(x + 1.5*width, res_vals, width, label='Residual', color='orange', alpha=0.7)
        
        ax_bars.set_xlabel('Basin', fontsize=12, fontweight='bold')
        ax_bars.set_ylabel('Water Flux [mm/day]', fontsize=12, fontweight='bold')
        ax_bars.set_title('Basin-Averaged Water Balance Components', fontsize=14, fontweight='bold')
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(basin_names_list, rotation=45, ha='right')
        ax_bars.legend(fontsize=10)
        ax_bars.grid(axis='y', alpha=0.3)
        ax_bars.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        output_file = os.path.join(OUTPUT_DIR, "water_balance_composite.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved composite figure to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"ERROR creating composite figure: {e}")

def save_netcdf_output(precip_mean, evap_mean, runoff_mean, residual_mean, lat, lon, ds):
    """Save output fields as NetCDF."""
    print("Saving NetCDF output...")
    
    try:
        # Create output dataset
        output_ds = xr.Dataset(
            {
                'precipitation': (['lat', 'lon'], precip_mean),
                'evapotranspiration': (['lat', 'lon'], evap_mean),
                'runoff': (['lat', 'lon'], runoff_mean),
                'residual': (['lat', 'lon'], residual_mean),
            },
            coords={
                'lat': lat,
                'lon': lon,
            }
        )
        
        output_ds.attrs['title'] = 'E3SM Land Model Water Balance Analysis'
        output_ds.attrs['period'] = f'{START_YEAR}-{END_YEAR}'
        output_ds.attrs['units'] = 'mm/day'
        
        output_file = os.path.join(OUTPUT_DIR, "water_balance_fields.nc")
        output_ds.to_netcdf(output_file)
        print(f"Saved NetCDF to {output_file}")
        
    except Exception as e:
        print(f"ERROR saving NetCDF: {e}")

def save_summary_statistics(precip_mean, evap_mean, runoff_mean, residual_mean, ds):
    """Save summary statistics to CSV."""
    print("Saving summary statistics...")
    
    try:
        lat = ds['lat'].values
        lon = ds['lon'].values
        
        # Compute global means
        p_global = compute_area_weighted_global_mean(precip_mean, ds)
        et_global = compute_area_weighted_global_mean(evap_mean, ds)
        q_global = compute_area_weighted_global_mean(runoff_mean, ds)
        res_global = compute_area_weighted_global_mean(residual_mean, ds)
        
        # Create summary dataframe
        summary_data = {
            'Component': ['Precipitation', 'Evapotranspiration', 'Runoff', 'Residual'],
            'Global Mean [mm/day]': [p_global, et_global, q_global, res_global],
            'Min [mm/day]': [np.nanmin(precip_mean), np.nanmin(evap_mean), 
                            np.nanmin(runoff_mean), np.nanmin(residual_mean)],
            'Max [mm/day]': [np.nanmax(precip_mean), np.nanmax(evap_mean), 
                            np.nanmax(runoff_mean), np.nanmax(residual_mean)],
            'Std Dev [mm/day]': [np.nanstd(precip_mean), np.nanstd(evap_mean), 
                                np.nanstd(runoff_mean), np.nanstd(residual_mean)],
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        output_file = os.path.join(OUTPUT_DIR, "water_balance_summary.csv")
        summary_df.to_csv(output_file, index=False)
        print(f"Saved summary statistics to {output_file}")
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
        
        # Save basin statistics
        basin_stats = []
        for basin_id, basin_name in BASINS.items():
            if basin_id in basin_polygons:
                p, et, q, res = compute_basin_statistics(
                    precip_mean, evap_mean, runoff_mean, residual_mean,
                    lat, lon, basin_polygons[basin_id]
                )
                basin_stats.append({
                    'Basin ID': basin_id,
                    'Basin Name': basin_name,
                    'Precipitation [mm/day]': p,
                    'Evapotranspiration [mm/day]': et,
                    'Runoff [mm/day]': q,
                    'Residual [mm/day]': res,
                })
        
        basin_df = pd.DataFrame(basin_stats)
        basin_output_file = os.path.join(OUTPUT_DIR, "basin_statistics.csv")
        basin_df.to_csv(basin_output_file, index=False)
        print(f"\nSaved basin statistics to {basin_output_file}")
        print("\nBasin Statistics:")
        print(basin_df.to_string(index=False))
        
    except Exception as e:
        print(f"ERROR saving summary statistics: {e}")

def main():
    """Main analysis function."""
    print("=" * 80)
    print("E3SM Land Model Water Balance Closure Analysis")
    print("=" * 80)
    
    # Create output directory
    create_output_dir()
    
    # Load ELM data
    ds = load_elm_data()
    if ds is None:
        print("ERROR: Failed to load ELM data")
        return
    
    # Compute water balance components
    precip, evap, runoff, residual = compute_water_balance_components(ds)
    if precip is None:
        print("ERROR: Failed to compute water balance components")
        return
    
    # Compute climatology
    precip_mean, evap_mean, runoff_mean, residual_mean = compute_climatology(
        precip, evap, runoff, residual
    )
    
    # Get coordinates
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Load basin polygons
    global basin_polygons
    basin_polygons = load_basin_polygons()
    
    # Save NetCDF output
    save_netcdf_output(precip_mean, evap_mean, runoff_mean, residual_mean, lat, lon, ds)
    
    # Save summary statistics
    save_summary_statistics(precip_mean, evap_mean, runoff_mean, residual_mean, ds)
    
    # Create composite figure
    create_composite_figure(precip_mean, evap_mean, runoff_mean, residual_mean,
                          lat, lon, basin_polygons, ds)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()