import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import glob

# Define constants and paths
DATA_DIR = "./data/sample/e3sm/"
OBS_DIR = "./data/sample/obs/"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_07_integrated_diagnostic/run2_debug/v3/output"
CASE_NAME = "sample.v3.LR.historical"
BASIN_POLYGONS_PATH = os.path.join(OBS_DIR, "basin_polygons.geojson")
GAUGE_METADATA_PATH = os.path.join(OBS_DIR, "gauge_metadata.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_data(variables, component, years):
    """Load and compute climatological mean for specified variables."""
    data = {}
    for var in variables:
        files = []
        for year in range(years[0], years[1] + 1):
            pattern = f"{DATA_DIR}{component}/{CASE_NAME}.{var}.h0.{year:04d}-*.nc"
            matched_files = glob.glob(pattern)
            if matched_files:
                files.extend(matched_files)
        
        if not files:
            print(f"No files found for variable {var} in years {years[0]}-{years[1]}.")
            continue
        
        ds_list = [xr.open_dataset(f) for f in files]
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
        # Sort and calculate cumulative distribution
        sim_sorted = np.sort(simulated)
        obs_sorted = np.sort(observed)
        
        sim_cdf = np.cumsum(sim_sorted) / np.sum(sim_sorted)
        obs_cdf = np.cumsum(obs_sorted) / np.sum(obs_sorted)
        
        volume_bias = np.mean(sim_sorted - obs_sorted)
        wasserstein_dist = wasserstein_distance(sim_sorted, obs_sorted)
        
        return volume_bias, wasserstein_dist
    except Exception as e:
        print(f"Error computing FDC metrics: {e}")
        return None, None

def main():
    # Part 1: Extract model fields
    elm_vars = ['RAIN', 'SNOW', 'QVEGE', 'QVEGT', 'QSOIL', 'QRUNOFF']
    model_data = load_model_data(elm_vars, 'lnd', (1985, 1989))
    
    # Combine precipitation and evapotranspiration
    if 'RAIN' in model_data and 'SNOW' in model_data:
        model_data['P'] = model_data['RAIN'] + model_data['SNOW']
    else:
        print("Precipitation data missing for some years.")
        
    if all(var in model_data for var in ['QVEGE', 'QVEGT', 'QSOIL']):
        model_data['ET'] = model_data['QVEGE'] + model_data['QVEGT'] + model_data['QSOIL']
    else:
        print("Evapotranspiration data missing for some years.")

    # Part 2: Fetch and extract observation fields
    obs_precip = fetch_ilamb_data('pr', 'GPCCv2018')
    obs_et = fetch_ilamb_data('et', 'MODIS')
    obs_runoff = fetch_ilamb_data('mrro', 'LORA')

    # Part 3: Clip to basin means
    basins = gpd.read_file(BASIN_POLYGONS_PATH)
    
    # Ensure the 'grdc_no' column exists in the GeoDataFrame
    if 'grdc_no' not in basins.columns:
        print("Error: 'grdc_no' column missing from basin polygons.")
        return
    
    basin_ids = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
    
    results = []
    for basin_id in basin_ids:
        try:
            basin_polygon = basins[basins['grdc_no'] == basin_id].geometry.iloc[0]
        except IndexError:
            print(f"Basin ID {basin_id} not found in the GeoJSON.")
            continue
        
        # Clip model data
        clipped_model_data = clip_to_basin(model_data, basin_polygon)
        
        # Clip observation data (assuming similar structure to model data for simplicity)
        if obs_precip is not None:
            clipped_obs_precip = clip_to_basin(obs_precip.to_dataset(), basin_polygon)
        else:
            clipped_obs_precip = None
        
        if obs_et is not None:
            clipped_obs_et = clip_to_basin(obs_et.to_dataset(), basin_polygon)
        else:
            clipped_obs_et = None
        
        if obs_runoff is not None:
            clipped_obs_runoff = clip_to_basin(obs_runoff.to_dataset(), basin_polygon)
        else:
            clipped_obs_runoff = None
        
        # Part 4: Streamflow FDC metrics
        gauge_metadata = pd.read_csv(GAUGE_METADATA_PATH)
        
        for _, row in gauge_metadata.iterrows():
            if 'grdc_no' not in row or row['grdc_no'] != basin_id:
                continue
            
            gauge_id = row['gauge_id']
            obs_streamflow_path = os.path.join(OBS_DIR, f"streamflow/{gauge_id}.csv")
            
            try:
                obs_df = pd.read_csv(obs_streamflow_path)
                obs_discharge = obs_df['discharge_m3s'].values
                
                # Simulated discharge (assuming MOSART data is available in a similar format)
                mosart_files = [f"{DATA_DIR}rof/{CASE_NAME}.mosart.h1.*.nc"]
                matched_files = []
                for pattern in mosart_files:
                    matched_files.extend(glob.glob(pattern))
                
                if not matched_files:
                    print(f"No MOSART files found for basin {basin_id}.")
                    continue
                
                ds_list = [xr.open_dataset(f) for f in matched_files]
                combined_ds = xr.concat(ds_list, dim='time')
                
                # Find nearest grid cell
                lat_diff = np.abs(combined_ds['lat'].values - row['lat'])
                lon_diff = np.abs(combined_ds['lon'].values - row['lon'])
                idx = np.unravel_index(np.argmin(lat_diff + lon_diff), combined_ds['lat'].shape)
                
                sim_discharge = combined_ds['RIVER_DISCHARGE_OVER_LAND_LIQ'].isel(lat=idx[0], lon=idx[1]).values
                
                # Compute FDC metrics
                volume_bias, wasserstein_dist = compute_fdc_metrics(sim_discharge, obs_discharge)
                
                results.append({
                    'basin_id': basin_id,
                    'P_bias': clipped_model_data['P'] - clipped_obs_precip['pr'] if clipped_obs_precip else None,
                    'ET_bias': clipped_model_data['ET'] - clipped_obs_et['et'] if clipped_obs_et else None,
                    'Q_bias': clipped_model_data['QRUNOFF'] - clipped_obs_runoff['mrro'] if clipped_obs_runoff else None,
                    'volume_bias': volume_bias,
                    'wasserstein_dist': wasserstein_dist
                })
            except Exception as e:
                print(f"Error processing gauge {gauge_id}: {e}")
    
    # Part 5: Combine and visualise
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_table.csv'), index=False)
    
    # Plotting
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for basin_id in basin_ids:
            subset = results_df[results_df['basin_id'] == basin_id]
            if not subset.empty:
                ax.bar(subset.index + 0.1 * np.arange(len(basin_ids)), subset['P_bias'], width=0.2, label=f'P Bias {basin_id}')
                ax.bar(subset.index + 0.3 * np.arange(len(basin_ids)), subset['ET_bias'], width=0.2, label=f'ET Bias {basin_id}')
                ax.bar(subset.index + 0.5 * np.arange(len(basin_ids)), subset['Q_bias'], width=0.2, label=f'Q Bias {basin_id}')
        
        ax.set_xticks(np.arange(len(basin_ids)) + 0.2)
        ax.set_xticklabels([f'{id}' for id in basin_ids])
        ax.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_vs_obs_bar_chart.png'))
    except Exception as e:
        print(f"Error plotting bar chart: {e}")

if __name__ == "__main__":
    main()