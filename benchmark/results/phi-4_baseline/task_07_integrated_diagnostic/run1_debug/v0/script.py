import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Define constants and paths
DATA_DIR = "./data/sample/e3sm/"
OBS_DIR = "./data/sample/obs/"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_07_integrated_diagnostic/run1_output"
CASE_NAME = "sample.v3.LR.historical"
BASIN_POLYGONS_PATH = os.path.join(OBS_DIR, "basin_polygons.geojson")
GAUGE_METADATA_PATH = os.path.join(OBS_DIR, "gauge_metadata.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_data(variables, component, years):
    """Load and compute climatological mean for specified variables."""
    data = {}
    for var in variables:
        files = [f"{DATA_DIR}{component}/{CASE_NAME}.{var}.h0.{year:04d}-*.nc" for year in range(years[0], years[1] + 1)]
        ds_list = [xr.open_mfdataset(f, combine='by_coords') for f in files]
        combined_ds = xr.concat(ds_list, dim='time')
        data[var] = combined_ds[var].groupby('time.month').mean(dim='time')
    return data

def fetch_ilamb_data(variable, dataset):
    """Fetch ILAMB data and compute climatological mean."""
    url_template = f"https://www.ilamb.org/ILAMB-Data/DATA/{dataset}/{variable}.nc"
    try:
        ds = xr.open_dataset(url_template)
        monthly_mean = ds[variable].groupby('time.month').mean(dim='time')
        return monthly_mean
    except Exception as e:
        print(f"Error fetching ILAMB data for {variable}: {e}")
        return None

def clip_to_basin(data, basin_polygon):
    """Clip data to a given basin polygon."""
    try:
        # Convert lat/lon to points and check if they are within the basin
        lats = data['lat'].values.flatten()
        lons = data['lon'].values.flatten()
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        
        mask = np.array([basin_polygon.contains(point) for point in points])
        mask = mask.reshape(data['lat'].shape)
        
        clipped_data = {}
        for var in data:
            if 'lat' not in var and 'lon' not in var:  # Exclude coordinate variables
                clipped_data[var] = (data[var].where(mask, drop=True)).mean(dim=('lat', 'lon'))
        return clipped_data
    except Exception as e:
        print(f"Error clipping data to basin: {e}")
        return None

def compute_fdc_metrics(simulated, observed):
    """Compute flow duration curve metrics."""
    try:
        # Sort and normalize the discharge values
        sim_sorted = np.sort(simulated)
        obs_sorted = np.sort(observed)
        
        # Compute volume bias
        volume_bias = (sim_sorted.sum() - obs_sorted.sum()) / obs_sorted.sum()
        
        # Compute Wasserstein distance
        wasserstein_dist = wasserstein_distance(sim_sorted, obs_sorted)
        
        return volume_bias, wasserstein_dist
    except Exception as e:
        print(f"Error computing FDC metrics: {e}")
        return None, None

def main():
    # Part 1: Extract model fields
    elm_vars = ['RAIN', 'SNOW', 'QVEGE', 'QVEGT', 'QSOIL', 'QRUNOFF']
    elm_data = load_model_data(elm_vars, 'lnd', (1985, 1989))
    
    # Combine precipitation and evapotranspiration
    elm_data['P'] = elm_data['RAIN'] + elm_data['SNOW']
    elm_data['ET'] = elm_data['QVEGE'] + elm_data['QVEGT'] + elm_data['QSOIL']

    # Part 2: Fetch and extract observation fields
    obs_vars = {
        'pr': ('GPCCv2018', 'pr'),
        'et': ('MODIS', 'evspsbl'),  # Extract variable 'et'
        'mrro': ('LORA', 'mrro')
    }
    
    obs_data = {}
    for var, (dataset, ilamb_var) in obs_vars.items():
        obs_data[var] = fetch_ilamb_data(ilamb_var, dataset)
    
    # Part 3: Clip to basin means
    basins = gpd.read_file(BASIN_POLYGONS_PATH)
    basin_ids = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
    basin_results = {}
    
    for basin_id in basin_ids:
        basin_polygon = basins[basins['grdc_no'] == basin_id].geometry.iloc[0]
        
        # Clip model data
        elm_clipped = clip_to_basin(elm_data, basin_polygon)
        
        # Clip observation data
        obs_clipped = {}
        for var in ['pr', 'et', 'mrro']:
            if obs_data[var] is not None:
                obs_clipped[var] = clip_to_basin(obs_data[var], basin_polygon)
        
        # Store results
        basin_results[basin_id] = {
            'model': elm_clipped,
            'observation': obs_clipped
        }
    
    # Part 4: Streamflow FDC metrics
    gauge_metadata = pd.read_csv(GAUGE_METADATA_PATH)
    fdc_metrics = {}
    
    for _, row in gauge_metadata.iterrows():
        gauge_id = row['gauge_id']
        lat, lon = row['lat'], row['lon']
        
        # Find nearest MOSART grid cell
        mosart_files = [f"{DATA_DIR}rof/{CASE_NAME}.mosart.h0.*.nc"]
        ds_list = [xr.open_mfdataset(f, combine='by_coords') for f in mosart_files]
        combined_ds = xr.concat(ds_list, dim='time')
        
        # Find nearest grid point
        dists = np.sqrt((combined_ds['lat'] - lat)**2 + (combined_ds['lon'] - lon)**2)
        min_idx = np.unravel_index(dists.argmin(), dists.shape)
        
        # Extract simulated discharge
        sim_discharge = combined_ds['RIVER_DISCHARGE_OVER_LAND_LIQ'].isel(lat=min_idx[0], lon=min_idx[1])
        
        # Load observed discharge
        obs_file = os.path.join(OBS_DIR, f"streamflow/{gauge_id}.csv")
        obs_df = pd.read_csv(obs_file)
        obs_discharge = obs_df['discharge_m3s'].values
        
        # Compute FDC metrics
        volume_bias, wasserstein_dist = compute_fdc_metrics(sim_discharge.values, obs_discharge)
        
        fdc_metrics[gauge_id] = {
            'volume_bias': volume_bias,
            'wasserstein_distance': wasserstein_dist
        }
    
    # Part 5: Combine and visualize
    summary_table = pd.DataFrame()
    
    for basin_id in basin_ids:
        model_means = basin_results[basin_id]['model']
        obs_means = basin_results[basin_id]['observation']
        
        p_bias = (model_means['P'] - obs_means['pr']).values.item() if 'pr' in obs_means else np.nan
        et_bias = (model_means['ET'] - obs_means['et']).values.item() if 'et' in obs_means else np.nan
        runoff_bias = (model_means['QRUNOFF'] - obs_means['mrro']).values.item() if 'mrro' in obs_means else np.nan
        
        # Collect FDC metrics for gauges within the basin
        gauge_ids_in_basin = gauge_metadata[gauge_metadata['grdc_no'] == basin_id]['gauge_id']
        fdc_metrics_for_basin = [fdc_metrics[gid] for gid in gauge_ids_in_basin if gid in fdc_metrics]
        
        # Compute average FDC metrics
        avg_volume_bias = np.mean([m['volume_bias'] for m in fdc_metrics_for_basin])
        avg_wasserstein_dist = np.mean([m['wasserstein_distance'] for m in fdc_metrics_for_basin])
        
        summary_table.loc[basin_id, 'P_bias'] = p_bias
        summary_table.loc[basin_id, 'ET_bias'] = et_bias
        summary_table.loc[basin_id, 'Runoff_bias'] = runoff_bias
        summary_table.loc[basin_id, 'Volume_bias'] = avg_volume_bias
        summary_table.loc[basin_id, 'Wasserstein_distance'] = avg_wasserstein_dist
    
    # Save summary table
    summary_table.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"))
    
    # Plot model vs observation bar chart
    fig, ax = plt.subplots()
    for basin_id in basin_ids:
        model_means = basin_results[basin_id]['model']
        obs_means = basin_results[basin_id]['observation']
        
        p_model = model_means['P'].values.item() if 'P' in model_means else np.nan
        et_model = model_means['ET'].values.item() if 'ET' in model_means else np.nan
        q_model = model_means['QRUNOFF'].values.item() if 'QRUNOFF' in model_means else np.nan
        
        p_obs = obs_means['pr'].values.item() if 'pr' in obs_means else np.nan
        et_obs = obs_means['et'].values.item() if 'et' in obs_means else np.nan
        q_obs = obs_means['mrro'].values.item() if 'mrro' in obs_means else np.nan
        
        ax.bar(basin_id, p_model - p_obs, label='P', color='blue')
        ax.bar(basin_id, et_model - et_obs, bottom=p_model - p_obs, label='ET', color='green')
        ax.bar(basin_id, q_model - q_obs, bottom=p_model - p_obs + et_model - et_obs, label='Q', color='red')
    
    ax.set_xlabel('Basin ID')
    ax.set_ylabel('Bias (Model - Observation)')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_vs_observation_bar_chart.png"))
    
    # Plot radar chart
    from math import pi
    
    def radar_plot(data, labels, title):
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], labels)
        
        data += data[:1]
        ax.plot(angles, data, linewidth=2, linestyle='solid')
        ax.fill(angles, data, 'b', alpha=0.1)
        
        plt.title(title, size=20, color='blue', y=1.1)
    
    for basin_id in basin_ids:
        row = summary_table.loc[basin_id]
        radar_plot(
            [row['P_bias'], row['ET_bias'], row['Runoff_bias'], row['Volume_bias'], row['Wasserstein_distance']],
            ['P Bias', 'ET Bias', 'Runoff Bias', 'Vol. Bias', 'Wasserstein'],
            f'Basin {basin_id} Diagnostic'
        )
    
    plt.savefig(os.path.join(OUTPUT_DIR, "radar_chart.png"))

if __name__ == "__main__":
    main()