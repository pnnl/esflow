import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import cartopy.crs as ccrs

# Define paths and case name
data_dir = "./data/sample/e3sm/"
obs_dir = "./data/sample/obs/streamflow/"
gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_04_streamflow_fdc/run2_debug/v2/output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load gauge metadata and ensure 'grdc_no' column exists
gauge_metadata = pd.read_csv(gauge_metadata_path)
if 'grdc_no' not in gauge_metadata.columns:
    print("Column 'grdc_no' is missing from the gauge metadata. Please check your data.")
else:
    selected_gauges = gauge_metadata[gauge_metadata['grdc_no'].isin([3629000, 4121801, 4115200, 6742900, 2969100, 1159100])]

def load_mosart_data(case_name, start_year=1985, end_year=1989):
    """Load MOSART river discharge data for the specified period."""
    files = [f"{data_dir}rof/{case_name}.mosart.h1.{year:04d}-{month:02d}-01-00000.nc" 
             for year in range(start_year, end_year + 1) 
             for month in range(1, 13)]
    
    ds_list = []
    for file in files:
        try:
            ds = xr.open_dataset(file)
            ds_list.append(ds['RIVER_DISCHARGE_OVER_LAND_LIQ'])
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not ds_list:
        raise ValueError("No MOSART data loaded.")
    
    return xr.concat(ds_list, dim='time')

def load_observation_data(gauge_id):
    """Load observation streamflow data for a given gauge."""
    try:
        obs_df = pd.read_csv(f"{obs_dir}{gauge_id}.csv", parse_dates=['date'], index_col='date')
        return obs_df['discharge_m3s']
    except Exception as e:
        print(f"Error loading observations for gauge {gauge_id}: {e}")
        return None

def compute_fdc_metrics(simulated, observed):
    """Compute FDC metrics: volume bias, Wasserstein distance, and quantile ratios."""
    if simulated is None or observed is None:
        return None
    
    # Compute flow duration curve (FDC)
    sim_sorted = np.sort(simulated)[::-1]
    obs_sorted = np.sort(observed)[::-1]
    
    # Volume Bias
    volume_bias = np.sum(sim_sorted) / np.sum(obs_sorted) - 1
    
    # Wasserstein Distance
    wasserstein_dist = wasserstein_distance(sim_sorted, obs_sorted)
    
    # Quantile Ratios (Q10, Q50, Q90)
    q10_ratio = sim_sorted[int(0.1 * len(sim_sorted))] / obs_sorted[int(0.1 * len(obs_sorted))]
    q50_ratio = sim_sorted[int(0.5 * len(sim_sorted))] / obs_sorted[int(0.5 * len(obs_sorted))]
    q90_ratio = sim_sorted[int(0.9 * len(sim_sorted))] / obs_sorted[int(0.9 * len(obs_sorted))]
    
    return {
        'volume_bias': volume_bias,
        'wasserstein_distance': wasserstein_dist,
        'q10_ratio': q10_ratio,
        'q50_ratio': q50_ratio,
        'q90_ratio': q90_ratio
    }

def plot_wasserstein_map(gauge_metadata, metrics):
    """Plot a map showing the Wasserstein distance for each gauge."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    
    # Plot gauges with color-coded Wasserstein distances
    sc = ax.scatter(gauge_metadata['lon'], gauge_metadata['lat'],
                    c=[metrics[g] for g in gauge_metadata['grdc_no']],
                    cmap='viridis', transform=ccrs.PlateCarree(), s=100)
    plt.colorbar(sc, label='Wasserstein Distance')
    
    plt.title('Wasserstein Distance by Gauge Location')
    plt.show()

def main():
    if 'selected_gauges' in locals():
        case_name = "sample.v3.LR.historical"
        mosart_data = load_mosart_data(case_name)

        metrics_df = pd.DataFrame(columns=['gauge_id', 'volume_bias', 'wasserstein_distance', 
                                           'q10_ratio', 'q50_ratio', 'q90_ratio'])
        
        for _, gauge in selected_gauges.iterrows():
            gauge_id = gauge['grdc_no']
            
            # Find nearest grid cell
            lat, lon = gauge['lat'], gauge['lon']
            dist = np.sqrt((mosart_data.lat - lat)**2 + (mosart_data.lon - lon)**2)
            nearest_idx = dist.argmin()
            
            # Extract simulated discharge for the gauge location
            sim_discharge = mosart_data.isel(lat=nearest_idx, lon=nearest_idx).values
            
            # Load observed data
            obs_discharge = load_observation_data(gauge_id)
            
            # Filter to 1985-1989 period
            sim_discharge = pd.Series(sim_discharge, index=pd.date_range(start='1985-01-01', periods=len(sim_discharge), freq='D'))
            obs_discharge = obs_discharge['1985':'1990']
            
            # Compute metrics
            metrics = compute_fdc_metrics(sim_discharge, obs_discharge)
            
            if metrics:
                metrics_df = metrics_df.append({
                    'gauge_id': gauge_id,
                    **metrics
                }, ignore_index=True)
        
        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(output_dir, "fdc_metrics.csv"), index=False)
        
        # Plot Wasserstein map
        plot_wasserstein_map(selected_gauges, {row['gauge_id']: row['wasserstein_distance'] for _, row in metrics_df.iterrows()})

if __name__ == "__main__":
    main()