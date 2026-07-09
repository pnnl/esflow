import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import pearsonr

# Define constants and paths
e3sm_path = "./data/sample/e3sm/"
modis_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_03_et_benchmark/run1_debug/v1/output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def fetch_modis_data(url):
    try:
        modis_ds = xr.open_dataset(url)
        return modis_ds['et']
    except Exception as e:
        print(f"Error fetching MODIS data: {e}")
        # Attempt to download the file locally if not found
        local_file_path = os.path.join(output_dir, 'et_0.5x0.5.nc')
        try:
            import requests
            response = requests.get(url)
            response.raise_for_status()
            with open(local_file_path, 'wb') as f:
                f.write(response.content)
            modis_ds = xr.open_dataset(local_file_path)
            return modis_ds['et']
        except Exception as download_error:
            print(f"Error downloading MODIS data: {download_error}")
            return None

def load_e3sm_et(case_name, years):
    et_vars = ['QVEGE', 'QVEGT', 'QSOIL']
    try:
        et_data = []
        for year in range(years[0], years[1] + 1):
            file_path = os.path.join(e3sm_path, f"lnd/{case_name}.elm.h0.{year:04d}-01.nc")
            ds = xr.open_dataset(file_path)
            et_yearly = sum(ds[var].sum(dim='time') for var in et_vars) * 86400  # Convert mm/s to mm/day
            et_data.append(et_yearly)
        return xr.concat(et_data, dim='time').mean(dim='time')
    except Exception as e:
        print(f"Error loading E3SM ET data: {e}")
        return None

def compute_bias_and_correlation(model_et, obs_et):
    try:
        bias = model_et - obs_et
        global_mean_bias = bias.mean().item()
        correlation, _ = pearsonr(bias.values.flatten(), obs_et.values.flatten())
        return bias, global_mean_bias, correlation
    except Exception as e:
        print(f"Error computing bias and correlation: {e}")
        return None, None, None

def plot_bias_map(bias):
    try:
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        bias.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', cbar_kwargs={'label': 'Bias (mm/day)'})
        ax.coastlines()
        plt.title('Spatial Bias Map: E3SM ET vs MODIS ET')
        plt.savefig(os.path.join(output_dir, 'bias_map.png'))
    except Exception as e:
        print(f"Error plotting bias map: {e}")

def main():
    case_name = "sample.v3.LR.historical"
    years = (1985, 1989)

    # Fetch MODIS ET data
    modis_et = fetch_modis_data(modis_url)
    if modis_et is None:
        return

    # Load E3SM ET data
    e3sm_et = load_e3sm_et(case_name, years)
    if e3sm_et is None:
        return

    # Compute bias and correlation
    bias, global_mean_bias, correlation = compute_bias_and_correlation(e3sm_et, modis_et)
    if bias is None:
        return

    # Save results to CSV
    try:
        pd.DataFrame({
            'global_mean_bias': [global_mean_bias],
            'correlation': [correlation]
        }).to_csv(os.path.join(output_dir, 'bias_statistics.csv'), index=False)
    except Exception as e:
        print(f"Error saving bias statistics: {e}")

    # Plot and save the bias map
    plot_bias_map(bias)

if __name__ == "__main__":
    main()