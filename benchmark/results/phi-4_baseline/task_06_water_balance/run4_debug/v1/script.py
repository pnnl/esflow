import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import shape, Point
import geopandas as gpd

# Define constants and paths
data_dir = "./data/sample/e3sm/"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_06_water_balance/run4_debug/v1/output"
os.makedirs(output_dir, exist_ok=True)

case_name = "sample.v3.LR.historical"
basin_polygons_path = "./data/sample/obs/basin_polygons.geojson"
gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"

# Define basins of interest
basins_of_interest = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]

# Load basin polygons
try:
    basin_polygons = gpd.read_file(basin_polygons_path)
except Exception as e:
    print(f"Error loading basin polygons: {e}")
    raise

# Filter basins of interest
basin_polygons = basin_polygons[basin_polygons['grdc_no'].isin(basins_of_interest)]

# Load ELM data for the period 1985-1989
try:
    files = [os.path.join(data_dir, f"lnd/{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
             for year in range(1985, 1990) for month in range(1, 13)]
    ds = xr.open_mfdataset(files)
except Exception as e:
    print(f"Error loading ELM data: {e}")
    raise

# Calculate climatological time-mean fields
try:
    P = (ds['RAIN'] + ds['SNOW']).groupby('time.month').mean(dim='time')
    ET = (ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']).groupby('time.month').mean(dim='time')
    Q = ds['QRUNOFF'].groupby('time.month').mean(dim='time')
except Exception as e:
    print(f"Error calculating climatological time-mean fields: {e}")
    raise

# Compute area-weighted global mean
try:
    weights = np.cos(np.deg2rad(ds.lat))
    P_global_mean = (P * weights).sum(dim=('lat', 'lon')) / weights.sum()
    ET_global_mean = (ET * weights).sum(dim=('lat', 'lon')) / weights.sum()
    Q_global_mean = (Q * weights).sum(dim=('lat', 'lon')) / weights.sum()
except Exception as e:
    print(f"Error computing area-weighted global mean: {e}")
    raise

# Compute water balance residual
try:
    residual = P - ET - Q
except Exception as e:
    print(f"Error computing water balance residual: {e}")
    raise

# Clip fields to basins of interest
def clip_to_basin(da, basin):
    mask = gpd.GeoDataFrame(geometry=[shape(basin['geometry'])])
    points = [Point(lon, lat) for lon, lat in zip(da.lon.values.flatten(), da.lat.values.flatten())]
    geoms = gpd.GeoDataFrame({'geometry': points})
    geoms.crs = {'init': 'epsg:4326'}
    mask.crs = {'init': 'epsg:4326'}
    clipped = geoms[mask.contains(geoms.geometry)]
    return da.sel(lon=clipped.geometry.x, lat=clipped.geometry.y)

try:
    P_basins = {basin['grdc_no']: clip_to_basin(P, basin) for _, basin in basin_polygons.iterrows()}
    ET_basins = {basin['grdc_no']: clip_to_basin(ET, basin) for _, basin in basin_polygons.iterrows()}
    Q_basins = {basin['grdc_no']: clip_to_basin(Q, basin) for _, basin in basin_polygons.iterrows()}
    residual_basins = {basin['grdc_no']: clip_to_basin(residual, basin) for _, basin in basin_polygons.iterrows()}
except Exception as e:
    print(f"Error clipping fields to basins: {e}")
    raise

# Compute area-weighted means for each basin
def compute_area_weighted_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    return (da * weights).sum(dim=('lat', 'lon')) / weights.sum()

try:
    P_means = {basin: compute_area_weighted_mean(P_basins[basin]) for basin in basins_of_interest}
    ET_means = {basin: compute_area_weighted_mean(ET_basins[basin]) for basin in basins_of_interest}
    Q_means = {basin: compute_area_weighted_mean(Q_basins[basin]) for basin in basins_of_interest}
    residual_means = {basin: compute_area_weighted_mean(residual_basins[basin]) for basin in basins_of_interest}
except Exception as e:
    print(f"Error computing area-weighted means for basins: {e}")
    raise

# Plot global residual map with basin outlines
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
residual.mean(dim='time').plot(ax=ax, cmap='coolwarm', add_colorbar=False)
basin_polygons.boundary.plot(ax=ax, color='black')
plt.title('Global Water Balance Residual (P - ET - Q)')
plt.savefig(os.path.join(output_dir, 'global_residual_map.png'))
plt.close()

# Plot per-basin bar charts
basins = list(basins_of_interest)
bar_width = 0.2
index = np.arange(len(basins))

fig, ax = plt.subplots(figsize=(10, 6))
p_bars = ax.bar(index - 1.5 * bar_width, [P_means[basin].values for basin in basins], bar_width, label='Precipitation')
et_bars = ax.bar(index - 0.5 * bar_width, [ET_means[basin].values for basin in basins], bar_width, label='Evapotranspiration')
q_bars = ax.bar(index + 0.5 * bar_width, [Q_means[basin].values for basin in basins], bar_width, label='Runoff')
residual_bars = ax.bar(index + 1.5 * bar_width, [residual_means[basin].values for basin in basins], bar_width, label='Residual')

ax.set_xlabel('Basin ID')
ax.set_ylabel('Area-weighted mean (mm/s)')
ax.set_title('Per-basin Water Balance Components')
ax.set_xticks(index)
ax.set_xticklabels(basins)
ax.legend()

plt.savefig(os.path.join(output_dir, 'per_basin_bar_charts.png'))
plt.close()