import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import shape, Point
import geopandas as gpd
from scipy.stats import pearsonr

# Define constants and paths
E3SM_DIR = "./data/sample/e3sm/"
OBS_DIR = "./data/sample/obs/"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_05_basin_streamflow/run1_debug/v2/output"
CASE_NAME = "sample.v3.LR.historical"
BASIN_IDS = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
START_YEAR = 1985
END_YEAR = 1989

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mosart_discharge():
    """Load MOSART river discharge data."""
    try:
        files = sorted([f for f in os.listdir(E3SM_DIR + 'rof/') 
                        if f.startswith(CASE_NAME) and f.endswith('.mosart.h0.*.nc')])
        
        # Filter files to include only those within the specified year range
        ds_list = []
        for file in files:
            try:
                ds = xr.open_dataset(os.path.join(E3SM_DIR, 'rof', file))
                time_var = pd.to_datetime(ds['time'].values)
                if (time_var.year >= START_YEAR) and (time_var.year <= END_YEAR):
                    ds_list.append(ds['RIVER_DISCHARGE_OVER_LAND_LIQ'])
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        if not ds_list:
            raise ValueError("No valid MOSART discharge data found for the specified year range.")
        
        ds = xr.concat(ds_list, dim='time')
        return ds
    except Exception as e:
        print(f"Error loading MOSART discharge data: {e}")
        raise

def load_gauge_metadata():
    """Load gauge metadata."""
    try:
        return pd.read_csv(os.path.join(OBS_DIR, 'gauge_metadata.csv'))
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        raise

def load_basin_polygons():
    """Load basin polygons."""
    try:
        return gpd.read_file(os.path.join(OBS_DIR, 'basin_polygons.geojson'))
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        raise

def match_gauges_to_grid(ds_mosart, gauge_metadata):
    """Match gauges to the MOSART grid."""
    lat = ds_mosart.lat.values
    lon = ds_mosart.lon.values
    grid_points = np.array([lon.flatten(), lat.flatten()]).T
    
    def find_nearest(grid_points, point):
        distances = np.sqrt(((grid_points - point) ** 2).sum(axis=1))
        return np.argmin(distances)
    
    gauge_metadata['nearest_grid_idx'] = gauge_metadata.apply(
        lambda row: find_nearest(grid_points, [row.lon, row.lat]), axis=1
    )
    return gauge_metadata

def load_observed_discharge(gauge_id):
    """Load observed discharge for a given gauge."""
    try:
        obs_file = os.path.join(OBS_DIR, 'streamflow', f'{gauge_id}.csv')
        df = pd.read_csv(obs_file)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['discharge_m3s']
    except Exception as e:
        print(f"Error loading observed discharge for gauge {gauge_id}: {e}")
        raise

def compute_metrics(sim, obs):
    """Compute validation metrics."""
    sim_mean = np.nanmean(sim)
    obs_mean = np.nanmean(obs)
    
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
    r, _ = pearsonr(sim, obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + ((sim_mean / obs_mean) - 1) ** 2 + ((np.std(sim) / np.std(obs)) - 1) ** 2)
    pbias = (np.sum(sim - obs) / np.sum(obs)) * 100
    
    return rmse, nse, kge, pbias

def plot_basin_streamflow(basin_id, basin_polygon, sim_discharge, obs_discharge):
    """Plot streamflow for a given basin."""
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([basin_polygon.total_bounds[0], basin_polygon.total_bounds[2],
                   basin_polygon.total_bounds[1], basin_polygon.total_bounds[3]], crs=ccrs.PlateCarree())
    
    # Plot the basin boundary
    basin_polygon.boundary.plot(ax=ax, edgecolor='blue', linewidth=2)
    
    # Create a twin axis for time series plot
    ax2 = ax.twinx()
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    ax2.grid(False)
    
    # Plot the simulated and observed discharge
    sim_discharge.plot(ax=ax2, label='Simulated', color='red')
    obs_discharge.plot(ax=ax2, label='Observed', color='blue')
    
    ax2.legend(loc='upper left')
    plt.title(f"Basin {basin_id} Streamflow")
    plt.savefig(os.path.join(OUTPUT_DIR, f'basin_{basin_id}_streamflow.png'))
    plt.close()

def main():
    # Load data
    try:
        ds_mosart = load_mosart_discharge()
    except ValueError as e:
        print(f"Skipping analysis due to error: {e}")
        return
    
    gauge_metadata = load_gauge_metadata()
    basin_polygons = load_basin_polygons()
    
    # Match gauges to the MOSART grid
    gauge_metadata = match_gauges_to_grid(ds_mosart, gauge_metadata)
    
    for basin_id in BASIN_IDS:
        try:
            # Filter basins and gauges
            basin_polygon = basin_polygons[basin_polygons['grdc_no'] == basin_id].geometry.iloc[0]
            gauges_in_basin = gauge_metadata[gauge_metadata['nearest_grid_idx'].isin(
                [i for i, point in enumerate(ds_mosart.lat.values.flatten()) 
                 if Point(point).within(basin_polygon)]
            )]
            
            # Initialize arrays to store discharge data
            sim_discharge_all = []
            obs_discharge_all = []
            
            for _, gauge in gauges_in_basin.iterrows():
                try:
                    # Extract simulated discharge
                    grid_idx = gauge['nearest_grid_idx']
                    lat_idx, lon_idx = np.unravel_index(grid_idx, ds_mosart.lat.shape)
                    sim_discharge = ds_mosart.isel(lat=lat_idx, lon=lon_idx).values * 86400  # Convert to m3/day
                    
                    # Load observed discharge
                    obs_discharge = load_observed_discharge(gauge['gauge_id'])
                    
                    # Align time indices
                    sim_time = pd.date_range(start=f'{START_YEAR}-01-01', end=f'{END_YEAR}-12-31', freq='D')
                    sim_discharge_all.append(pd.Series(sim_discharge, index=sim_time))
                    obs_discharge_all.append(obs_discharge.reindex(sim_time, method='nearest'))
                    
                except Exception as e:
                    print(f"Error processing gauge {gauge['gauge_id']}: {e}")
                    continue
            
            # Concatenate all gauges' data
            sim_discharge = pd.concat(sim_discharge_all).groupby(level=0).sum()
            obs_discharge = pd.concat(obs_discharge_all).groupby(level=0).sum()
            
            # Compute metrics
            rmse, nse, kge, pbias = compute_metrics(sim_discharge, obs_discharge)
            print(f"Basin {basin_id} - RMSE: {rmse}, NSE: {nse}, KGE: {kge}, PBIAS: {pbias}")
            
            # Plot streamflow
            plot_basin_streamflow(basin_id, basin_polygon, sim_discharge, obs_discharge)
        
        except Exception as e:
            print(f"Error processing basin {basin_id}: {e}")

if __name__ == "__main__":
    main()