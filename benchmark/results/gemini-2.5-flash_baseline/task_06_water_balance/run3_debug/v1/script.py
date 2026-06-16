import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os
import glob

def analyze_water_balance(output_dir):
    """
    Analyzes the E3SM land model water balance closure.

    Args:
        output_dir (str): Directory to save output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    data_root = "./data/sample/e3sm/lnd/"
    case_name = "sample.v3.LR.historical"
    start_year = 1985
    end_year = 1989
    years = range(start_year, end_year + 1)

    # E3SM variable names and conversion factors
    # ELM output is kg/m2/s for RAIN, SNOW, QVEGE, QVEGT, QSOIL, QRUNOFF
    # 1 kg/m2/s = 1 mm/s (density of water is 1000 kg/m3)
    # To convert to mm/day: mm/s * 60 s/min * 60 min/hr * 24 hr/day = mm/day * 86400
    SEC_TO_DAY = 86400

    print(f"Processing E3SM ELM data for {case_name} from {start_year}-{end_year}...")

    # 1. Load ELM monthly output for the specified period
    file_pattern = os.path.join(data_root, f"{case_name}.elm.h0.*.nc")
    all_files = sorted(glob.glob(file_pattern))

    # Filter files for the specified years
    files_to_open = []
    for year in years:
        files_to_open.extend([f for f in all_files if f"elm.h0.{year}-" in f])

    if not files_to_open:
        print(f"Error: No ELM files found for {case_name} in {data_root} for years {start_year}-{end_year}.")
        return

    try:
        ds = xr.open_mfdataset(files_to_open, combine='by_coords', decode_times=True)
        print(f"Successfully opened {len(files_to_open)} ELM files.")
    except Exception as e:
        print(f"Error opening ELM files: {e}")
        return

    # Ensure time dimension is present and valid
    if 'time' not in ds.dims:
        print("Error: 'time' dimension not found in ELM dataset.")
        return
    if ds['time'].size == 0:
        print("Error: ELM dataset has an empty 'time' dimension.")
        return

    # Extract climatological time-mean fields
    print("Calculating climatological time-mean fields...")
    try:
        # Calculate total precipitation (P)
        P = (ds['RAIN'] + ds['SNOW']).mean(dim='time') * SEC_TO_DAY
        P.name = 'P'
        P.attrs['units'] = 'mm/day'
        P.attrs['long_name'] = 'Total Precipitation (RAIN + SNOW)'

        # Calculate total evapotranspiration (ET)
        ET = (ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']).mean(dim='time') * SEC_TO_DAY
        ET.name = 'ET'
        ET.attrs['units'] = 'mm/day'
        ET.attrs['long_name'] = 'Total Evapotranspiration (QVEGE + QVEGT + QSOIL)'

        # Calculate total runoff (Q)
        Q = ds['QRUNOFF'].mean(dim='time') * SEC_TO_DAY
        Q.name = 'Q'
        Q.attrs['units'] = 'mm/day'
        Q.attrs['long_name'] = 'Total Runoff (QRUNOFF)'

        # Compute the water balance residual P - ET - Q
        Residual = P - ET - Q
        Residual.name = 'Residual'
        Residual.attrs['units'] = 'mm/day'
        Residual.attrs['long_name'] = 'Water Balance Residual (P - ET - Q)'

        print("Climatological means calculated.")
    except KeyError as e:
        print(f"Error: Missing variable in ELM dataset: {e}")
        return
    except Exception as e:
        print(f"Error during calculation of climatological means: {e}")
        return

    # Get grid cell areas (assuming 'area' variable is available or calculate from 'lat'/'lon')
    # ELM output typically has 'area' variable
    if 'area' in ds.data_vars:
        grid_area = ds['area']
        print("Using 'area' variable from dataset for area weighting.")
    else:
        print("Warning: 'area' variable not found. Attempting to calculate approximate grid cell areas.")
        # Approximate area calculation for a regular lat/lon grid (not ideal for unstructured grids)
        # This is a fallback and might not be accurate for E3SM's native grid.
        # E3SM ELM output is usually on a structured lat/lon grid for h0 files.
        if 'lat' in ds.dims and 'lon' in ds.dims:
            lat_rad = np.deg2rad(ds['lat'])
            lon_rad = np.deg2rad(ds['lon'])
            dlat = np.abs(lat_rad[1] - lat_rad[0]) if len(lat_rad) > 1 else 1
            dlon = np.abs(lon_rad[1] - lon_rad[0]) if len(lon_rad) > 1 else 1
            R = 6.371e6 # Earth radius in meters
            grid_area = (R**2 * np.cos(lat_rad) * dlat * dlon).broadcast_like(P)
            grid_area.attrs['units'] = 'm^2'
        else:
            print("Error: Cannot find 'area' variable or sufficient coordinates (lat/lon) to calculate grid areas.")
            return

    # Compute area-weighted global mean of each component
    print("Calculating area-weighted global means...")
    try:
        total_area = grid_area.sum()
        global_P = (P * grid_area).sum() / total_area
        global_ET = (ET * grid_area).sum() / total_area
        global_Q = (Q * grid_area).sum() / total_area
        global_Residual = (Residual * grid_area).sum() / total_area

        # Compute the Dask arrays to get scalar values before calling .item()
        global_P_val = global_P.compute().item()
        global_ET_val = global_ET.compute().item()
        global_Q_val = global_Q.compute().item()
        global_Residual_val = global_Residual.compute().item()

        print(f"Global Mean P: {global_P_val:.2f} mm/day")
        print(f"Global Mean ET: {global_ET_val:.2f} mm/day")
        print(f"Global Mean Q: {global_Q_val:.2f} mm/day")
        print(f"Global Mean Residual (P-ET-Q): {global_Residual_val:.2f} mm/day")

        # Save global means to a CSV
        global_means_df = pd.DataFrame({
            'Component': ['P', 'ET', 'Q', 'Residual'],
            'Global Mean (mm/day)': [global_P_val, global_ET_val, global_Q_val, global_Residual_val]
        })
        global_means_path = os.path.join(output_dir, 'global_water_balance_means.csv')
        global_means_df.to_csv(global_means_path, index=False)
        print(f"Global means saved to {global_means_path}")

    except Exception as e:
        print(f"Error calculating global means: {e}")
        return

    # 2. Clip fields to major basins
    basin_polygons_path = "./data/sample/obs/basin_polygons.geojson"
    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"

    try:
        basins_gdf = gpd.read_file(basin_polygons_path)
        gauge_metadata = pd.read_csv(gauge_metadata_path)
        print(f"Loaded basin polygons from {basin_polygons_path}")
        print(f"Loaded gauge metadata from {gauge_metadata_path}")
    except Exception as e:
        print(f"Error loading basin data: {e}")
        return

    # Select target basins by GRDC number (gauge_id)
    target_grdc_nos = {
        3629000: "Amazon",
        4121801: "Missouri",
        4115200: "Columbia",
        6742900: "Danube",
        2969100: "Mekong",
        1159100: "Orange"
    }

    selected_basins_gdf = basins_gdf[basins_gdf['grdc_no'].isin(target_grdc_nos.keys())].copy()
    selected_basins_gdf['basin_name'] = selected_basins_gdf['grdc_no'].map(target_grdc_nos)

    if selected_basins_gdf.empty:
        print("Error: No target basins found in the GeoJSON file.")
        return

    # Prepare data for clipping (create a Dataset for easier handling)
    ds_wb = xr.Dataset({
        'P': P,
        'ET': ET,
        'Q': Q,
        'Residual': Residual
    })

    # Ensure the E3SM data has 'lat' and 'lon' dimensions/coordinates
    if 'lat' not in ds_wb.dims or 'lon' not in ds_wb.dims:
        print("Error: E3SM data must have 'lat' and 'lon' dimensions for clipping.")
        return

    # Convert E3SM longitude from 0-360 to -180-180 for easier spatial operations if needed
    # (GeoPandas usually handles this, but explicit conversion can prevent issues)
    if ds_wb['lon'].max() > 180:
        ds_wb = ds_wb.assign_coords(lon=(((ds_wb['lon'] + 180) % 360) - 180))
        ds_wb = ds_wb.sortby('lon')

    basin_data = {}
    basin_means_df = pd.DataFrame(columns=['Basin', 'P (mm/day)', 'ET (mm/day)', 'Q (mm/day)', 'Residual (mm/day)'])

    print("Clipping fields to basins and calculating basin-averaged values...")
    for _, basin_row in selected_basins_gdf.iterrows():
        basin_name = basin_row['basin_name']
        basin_geometry = basin_row['geometry']

        try:
            # Create a mask for the current basin
            lon_grid, lat_grid = np.meshgrid(ds_wb['lon'].values, ds_wb['lat'].values)
            points = gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten())
            mask_series = points.within(basin_geometry)
            basin_mask = xr.DataArray(
                mask_series.values.reshape(lon_grid.shape),
                coords={'lat': ds_wb['lat'], 'lon': ds_wb['lon']},
                dims=['lat', 'lon']
            )

            # Apply the mask to the data variables
            masked_P = P.where(basin_mask)
            masked_ET = ET.where(basin_mask)
            masked_Q = Q.where(basin_mask)
            masked_Residual = Residual.where(basin_mask)

            # Calculate area-weighted mean for the basin
            basin_grid_area = grid_area.where(basin_mask)
            basin_total_area = basin_grid_area.sum().compute() # Compute here

            if basin_total_area.item() > 0:
                basin_P_mean = (masked_P * basin_grid_area).sum() / basin_total_area
                basin_ET_mean = (masked_ET * basin_grid_area).sum() / basin_total_area
                basin_Q_mean = (masked_Q * basin_grid_area).sum() / basin_total_area
                basin_Residual_mean = (masked_Residual * basin_grid_area).sum() / basin_total_area

                # Compute the Dask arrays to get scalar values before calling .item()
                basin_P_mean_val = basin_P_mean.compute().item()
                basin_ET_mean_val = basin_ET_mean.compute().item()
                basin_Q_mean_val = basin_Q_mean.compute().item()
                basin_Residual_mean_val = basin_Residual_mean.compute().item()

                basin_data[basin_name] = {
                    'P': basin_P_mean_val,
                    'ET': basin_ET_mean_val,
                    'Q': basin_Q_mean_val,
                    'Residual': basin_Residual_mean_val
                }
                basin_means_df.loc[len(basin_means_df)] = [
                    basin_name,
                    basin_P_mean_val,
                    basin_ET_mean_val,
                    basin_Q_mean_val,
                    basin_Residual_mean_val
                ]
                print(f"  - {basin_name}: P={basin_P_mean_val:.2f}, ET={basin_ET_mean_val:.2f}, Q={basin_Q_mean_val:.2f}, Residual={basin_Residual_mean_val:.2f} mm/day")
            else:
                print(f"  - Warning: No E3SM grid cells found within {basin_name} basin. Skipping.")
                basin_data[basin_name] = {'P': np.nan, 'ET': np.nan, 'Q': np.nan, 'Residual': np.nan}

        except Exception as e:
            print(f"Error processing basin {basin_name}: {e}")
            basin_data[basin_name] = {'P': np.nan, 'ET': np.nan, 'Q': np.nan, 'Residual': np.nan}

    # Save basin means to CSV
    basin_means_path = os.path.join(output_dir, 'basin_water_balance_means.csv')
    basin_means_df.to_csv(basin_means_path, index=False)
    print(f"Basin means saved to {basin_means_path}")

    # 3. Produce composite figure
    print("Generating composite figure...")
    try:
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1]) # Two rows, top for map, bottom for bar charts

        # Top panel: Global Residual Map with Basin Outlines
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, color='lightgray')
        ax_map.add_feature(cfeature.OCEAN, color='lightblue')

        # Plot Residual
        residual_plot = Residual.plot.contourf(
            ax=ax_map,
            transform=ccrs.PlateCarree(),
            levels=np.linspace(-1, 1, 21), # Adjust levels for residual
            cmap='RdBu',
            cbar_kwargs={'label': 'Residual (P - ET - Q) [mm/day]', 'shrink': 0.7},
            extend='both'
        )
        ax_map.set_title('Global Water Balance Residual (P - ET - Q) [mm/day]')

        # Plot basin outlines
        selected_basins_gdf.to_crs(ccrs.PlateCarree()).plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.7)
        for idx, row in selected_basins_gdf.iterrows():
            centroid = row.geometry.centroid
            ax_map.text(centroid.x, centroid.y, row['basin_name'],
                        transform=ccrs.PlateCarree(),
                        fontsize=9, color='black', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # Bottom panel: Per-basin bar charts
        ax_bar = fig.add_subplot(gs[1, 0])

        if not basin_means_df.empty:
            basin_means_df_plot = basin_means_df.set_index('Basin')
            basin_means_df_plot[['P (mm/day)', 'ET (mm/day)', 'Q (mm/day)', 'Residual (mm/day)']].plot(
                kind='bar', ax=ax_bar, width=0.8, figsize=(12, 6)
            )
            ax_bar.set_title('Basin-averaged Water Balance Components')
            ax_bar.set_ylabel('Flux (mm/day)')
            ax_bar.tick_params(axis='x', rotation=45)
            ax_bar.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax_bar.text(0.5, 0.5, "No basin data to plot.", horizontalalignment='center', verticalalignment='center', transform=ax_bar.transAxes)


        plt.tight_layout()
        figure_path = os.path.join(output_dir, 'water_balance_composite_figure.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Composite figure saved to {figure_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating composite figure: {e}")

    print("Water balance analysis complete.")

if __name__ == "__main__":
    # Define the output directory as specified in the task
    output_directory = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_06_water_balance/run3_debug/v1/output"
    analyze_water_balance(output_directory)