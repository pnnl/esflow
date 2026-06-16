import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import geopandas as gpd
from shapely.geometry import mapping

# --- Configuration ---
E3SM_DATA_DIR = "./data/sample/e3sm/"
OBS_DATA_DIR = "./data/sample/obs/"
CASE_NAME = "sample.v3.LR.historical"
START_YEAR = 1985
END_YEAR = 1989
OUTPUT_ROOT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_06_water_balance/run4_output"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "task06_water_balance")

# E3SM variable definitions
ELM_PRECIP_VARS = ["RAIN", "SNOW"]  # kg/m2/s
ELM_ET_VARS = ["QVEGE", "QVEGT", "QSOIL"]  # mm/s
ELM_RUNOFF_VAR = "QRUNOFF"  # mm/s

# Target basins (GRDC_NO from gauge_metadata.csv)
TARGET_BASINS = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}

# --- Helper Functions ---
def create_output_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Output directory created: {path}")
    except Exception as e:
        print(f"Error creating output directory {path}: {e}")

def calculate_grid_area(ds):
    """
    Calculate grid cell area for a dataset with 'lat' and 'lon' coordinates.
    Assumes a regular lat/lon grid.
    """
    if 'area' in ds:
        return ds['area']

    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        raise ValueError("Dataset must have 'lat' and 'lon' coordinates to calculate area.")

    # Approximate Earth radius in meters
    R = 6.371e6

    # Convert lat/lon to radians
    lat_rad = np.deg2rad(ds['lat'])
    lon_rad = np.deg2rad(ds['lon'])

    # Calculate delta_lat and delta_lon
    if len(ds['lat']) > 1:
        dlat = np.abs(np.diff(lat_rad)[0])
    else: # Handle single latitude point case, assume a reasonable default
        dlat = np.deg2rad(1.0) # 1 degree latitude width

    if len(ds['lon']) > 1:
        dlon = np.abs(np.diff(lon_rad)[0])
    else: # Handle single longitude point case, assume a reasonable default
        dlon = np.deg2rad(1.0) # 1 degree longitude width

    # Area formula for a spherical cap segment
    # Area = R^2 * dlon * (sin(lat2) - sin(lat1))
    # For a grid cell, it's R^2 * dlon * (sin(lat + dlat/2) - sin(lat - dlat/2))
    # Which simplifies to R^2 * dlon * 2 * cos(lat) * sin(dlat/2)
    # Or, more simply for small dlat, R^2 * dlon * dlat * cos(lat)

    # Using the more precise formula for area of a spherical quadrilateral
    # Area = R^2 * dlon * (sin(lat_upper) - sin(lat_lower))
    # For a grid cell, lat_upper = lat + dlat/2, lat_lower = lat - dlat/2
    # This is equivalent to R^2 * dlon * (sin(lat + dlat/2) - sin(lat - dlat/2))
    # For a regular grid, dlat and dlon are constant.
    # Let's assume the 'lat' coordinate represents the cell center.
    # The cell boundaries would be lat - dlat/2 and lat + dlat/2.

    # If the grid is not regular, or if dlat/dlon are not explicitly defined,
    # we can approximate using the difference between adjacent points.
    # For simplicity, let's assume a regular grid and calculate cell boundaries.
    # If 'lat_bnds' and 'lon_bnds' exist, use them. Otherwise, infer.

    if 'lat_bnds' in ds and 'lon_bnds' in ds:
        lat_bnds = ds['lat_bnds']
        lon_bnds = ds['lon_bnds']
        area = (R**2 * (lon_bnds[:, 1] - lon_bnds[:, 0]) *
                (np.sin(np.deg2rad(lat_bnds[:, 1])) - np.sin(np.deg2rad(lat_bnds[:, 0]))))
        # Expand to 2D if needed
        if 'lon' in ds.dims and 'lat' in ds.dims:
            area = area.expand_dims(lon=len(ds['lon']), axis=1) # This is not quite right for 2D
            # A better way is to create a 2D area array
            area_2d = xr.DataArray(
                data=np.zeros((len(ds['lat']), len(ds['lon']))),
                coords={'lat': ds['lat'], 'lon': ds['lon']},
                dims=['lat', 'lon']
            )
            for i, lat_val in enumerate(ds['lat']):
                lat_bnd_lower = lat_bnds[i, 0]
                lat_bnd_upper = lat_bnds[i, 1]
                for j, lon_val in enumerate(ds['lon']):
                    lon_bnd_lower = lon_bnds[j, 0]
                    lon_bnd_upper = lon_bnds[j, 1]
                    area_2d[i, j] = R**2 * (np.deg2rad(lon_bnd_upper) - np.deg2rad(lon_bnd_lower)) * \
                                    (np.sin(np.deg2rad(lat_bnd_upper)) - np.sin(np.deg2rad(lat_bnd_lower)))
            return area_2d
        else:
            return area # Return 1D area if only 1D lat/lon
    else:
        # Infer cell boundaries for a regular grid
        lat_edges = np.concatenate([
            [lat_rad[0] - dlat / 2],
            (lat_rad[1:] + lat_rad[:-1]) / 2,
            [lat_rad[-1] + dlat / 2]
        ])
        lon_edges = np.concatenate([
            [lon_rad[0] - dlon / 2],
            (lon_rad[1:] + lon_rad[:-1]) / 2,
            [lon_rad[-1] + dlon / 2]
        ])

        # Create 2D area array
        area = xr.DataArray(
            data=np.zeros((len(ds['lat']), len(ds['lon']))),
            coords={'lat': ds['lat'], 'lon': ds['lon']},
            dims=['lat', 'lon']
        )

        for i in range(len(ds['lat'])):
            for j in range(len(ds['lon'])):
                area[i, j] = R**2 * (lon_edges[j+1] - lon_edges[j]) * \
                             (np.sin(lat_edges[i+1]) - np.sin(lat_edges[i]))
        return area

def area_weighted_global_mean(da, area_weights):
    """
    Calculate the area-weighted global mean of a DataArray.
    """
    if 'lat' not in da.dims or 'lon' not in da.dims:
        raise ValueError("DataArray must have 'lat' and 'lon' dimensions.")

    # Ensure weights are aligned with the data
    weights = area_weights.sel(lat=da.lat, lon=da.lon)
    weighted_data = da * weights
    global_sum = weighted_data.sum(dim=['lat', 'lon'])
    total_area = weights.sum(dim=['lat', 'lon'])
    return global_sum / total_area

def clip_to_basin(da, basin_polygon):
    """
    Clip a DataArray to a given basin polygon.
    Returns a DataArray with values outside the basin set to NaN.
    """
    # Ensure the DataArray has 'lat' and 'lon' dimensions
    if 'lat' not in da.dims or 'lon' not in da.dims:
        raise ValueError("DataArray must have 'lat' and 'lon' dimensions.")

    # Create a mask from the polygon
    # This requires rasterizing the polygon onto the grid
    # Use xarray's `rasterize` or a manual approach
    # For simplicity, we'll create a mask array and then apply it.

    # Create a dummy DataArray for the mask
    mask_da = xr.DataArray(
        data=np.zeros((len(da['lat']), len(da['lon'])), dtype=bool),
        coords={'lat': da['lat'], 'lon': da['lon']},
        dims=['lat', 'lon']
    )

    # Convert the polygon to a shapely geometry
    if isinstance(basin_polygon, gpd.GeoSeries):
        basin_geometry = basin_polygon.geometry.iloc[0]
    else:
        basin_geometry = basin_polygon

    # Create a grid of points
    lon_grid, lat_grid = np.meshgrid(da['lon'].values, da['lat'].values)
    points = gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten())
    
    # Check which points are within the basin
    # This can be slow for large grids/complex polygons
    within_basin = points.within(basin_geometry)
    mask_array = within_basin.reshape(len(da['lat']), len(da['lon']))

    # Apply the mask
    clipped_da = da.where(mask_array)
    return clipped_da

# --- Main Script ---
if __name__ == "__main__":
    create_output_dir(OUTPUT_DIR)

    # 1. Load ELM monthly data for 1985-1989
    elm_files = []
    for year in range(START_YEAR, END_YEAR + 1):
        elm_files.extend(glob.glob(os.path.join(E3SM_DATA_DIR, "lnd",
                                                f"{CASE_NAME}.elm.h0.{year}-*.nc")))
    elm_files.sort()

    if not elm_files:
        print(f"No ELM files found for {CASE_NAME} in {E3SM_DATA_DIR} for {START_YEAR}-{END_YEAR}.")
        exit()

    print(f"Found {len(elm_files)} ELM files. Loading...")
    try:
        ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
        print("ELM data loaded successfully.")
    except Exception as e:
        print(f"Error loading ELM data: {e}")
        exit()

    # Ensure 'time' dimension is present and valid
    if 'time' not in ds_elm.dims:
        print("Error: 'time' dimension not found in ELM dataset.")
        exit()
    if not pd.api.types.is_datetime64_any_dtype(ds_elm['time']):
        print("Warning: 'time' coordinate is not a datetime type. Attempting to decode.")
        ds_elm = xr.decode_cf(ds_elm) # Try to decode again if needed

    # Select the time period
    ds_elm = ds_elm.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    if ds_elm.time.size == 0:
        print(f"No data found for the period {START_YEAR}-{END_YEAR}. Check file paths and time decoding.")
        exit()

    # Calculate grid cell areas
    try:
        # ELM grid uses 'lat' and 'lon'
        # Check if 'area' is already present (e.g., from ELM output)
        if 'area' in ds_elm.data_vars:
            elm_area_weights = ds_elm['area']
            print("Using 'area' variable from ELM dataset for weights.")
        else:
            print("Calculating grid cell areas for ELM grid...")
            elm_area_weights = calculate_grid_area(ds_elm)
            print("Grid cell areas calculated.")
    except Exception as e:
        print(f"Error calculating ELM grid area: {e}")
        exit()

    # 2. Extract climatological time-mean fields
    # Convert units to mm/day for consistency (ELM output is kg/m2/s or mm/s)
    # 1 kg/m2/s = 1 mm/s (for water density ~1000 kg/m3)
    # 1 mm/s * 86400 s/day = 86400 mm/day

    SECONDS_PER_DAY = 86400

    # Precipitation (P)
    try:
        P_components = [ds_elm[var] for var in ELM_PRECIP_VARS if var in ds_elm.data_vars]
        if not P_components:
            print(f"Warning: No precipitation variables ({ELM_PRECIP_VARS}) found in ELM dataset.")
            P_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder
        else:
            P_total = sum(P_components)
            P_clim = P_total.mean(dim='time') * SECONDS_PER_DAY # mm/day
            P_clim.name = 'P_clim'
            P_clim.attrs['units'] = 'mm/day'
            P_clim.attrs['long_name'] = 'Climatological Mean Precipitation'
            print("Climatological Precipitation (P) calculated.")
    except Exception as e:
        print(f"Error calculating P_clim: {e}")
        P_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder

    # Evapotranspiration (ET)
    try:
        ET_components = [ds_elm[var] for var in ELM_ET_VARS if var in ds_elm.data_vars]
        if not ET_components:
            print(f"Warning: No ET variables ({ELM_ET_VARS}) found in ELM dataset.")
            ET_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder
        else:
            ET_total = sum(ET_components)
            ET_clim = ET_total.mean(dim='time') * SECONDS_PER_DAY # mm/day
            ET_clim.name = 'ET_clim'
            ET_clim.attrs['units'] = 'mm/day'
            ET_clim.attrs['long_name'] = 'Climatological Mean Evapotranspiration'
            print("Climatological Evapotranspiration (ET) calculated.")
    except Exception as e:
        print(f"Error calculating ET_clim: {e}")
        ET_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder

    # Runoff (Q)
    try:
        if ELM_RUNOFF_VAR in ds_elm.data_vars:
            Q_clim = ds_elm[ELM_RUNOFF_VAR].mean(dim='time') * SECONDS_PER_DAY # mm/day
            Q_clim.name = 'Q_clim'
            Q_clim.attrs['units'] = 'mm/day'
            Q_clim.attrs['long_name'] = 'Climatological Mean Runoff'
            print("Climatological Runoff (Q) calculated.")
        else:
            print(f"Warning: Runoff variable '{ELM_RUNOFF_VAR}' not found in ELM dataset.")
            Q_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder
    except Exception as e:
        print(f"Error calculating Q_clim: {e}")
        Q_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder

    # 3. Compute area-weighted global mean of each component
    global_means = {}
    for var_da in [P_clim, ET_clim, Q_clim]:
        if not np.isnan(var_da.values).all(): # Only compute if not all NaNs
            try:
                global_mean_val = area_weighted_global_mean(var_da, elm_area_weights)
                global_means[var_da.name] = global_mean_val.item() # .item() to get scalar
                print(f"Global mean {var_da.name}: {global_means[var_da.name]:.2f} {var_da.units}")
            except Exception as e:
                print(f"Error calculating global mean for {var_da.name}: {e}")
                global_means[var_da.name] = np.nan
        else:
            global_means[var_da.name] = np.nan

    # 4. Compute the water balance residual P - ET - Q as a spatial field
    try:
        residual_clim = P_clim - ET_clim - Q_clim
        residual_clim.name = 'Residual_clim'
        residual_clim.attrs['units'] = 'mm/day'
        residual_clim.attrs['long_name'] = 'Climatological Mean Water Balance Residual (P - ET - Q)'
        print("Climatological Residual (P - ET - Q) calculated.")
    except Exception as e:
        print(f"Error calculating residual_clim: {e}")
        residual_clim = xr.DataArray(np.nan, coords=ds_elm.coords, dims=ds_elm.dims) # Placeholder

    # Calculate global mean residual
    if not np.isnan(residual_clim.values).all():
        try:
            global_means['Residual_clim'] = area_weighted_global_mean(residual_clim, elm_area_weights).item()
            print(f"Global mean Residual_clim: {global_means['Residual_clim']:.2f} {residual_clim.units}")
        except Exception as e:
            print(f"Error calculating global mean for Residual_clim: {e}")
            global_means['Residual_clim'] = np.nan
    else:
        global_means['Residual_clim'] = np.nan

    # Save global means to CSV
    try:
        global_means_df = pd.DataFrame.from_dict(global_means, orient='index', columns=['Value (mm/day)'])
        global_means_df.index.name = 'Variable'
        global_means_csv_path = os.path.join(OUTPUT_DIR, "global_water_balance_means.csv")
        global_means_df.to_csv(global_means_csv_path)
        print(f"Global water balance means saved to {global_means_csv_path}")
    except Exception as e:
        print(f"Error saving global means to CSV: {e}")

    # 5. Clip all four fields (P, ET, Q, and residual) to six major basins
    # Load basin polygons
    basin_polygons_path = os.path.join(OBS_DATA_DIR, "basin_polygons.geojson")
    try:
        gdf_basins = gpd.read_file(basin_polygons_path)
        print(f"Loaded {len(gdf_basins)} basin polygons.")
    except Exception as e:
        print(f"Error loading basin polygons from {basin_polygons_path}: {e}")
        exit()

    # Load gauge metadata to map GRDC_NO to basin names
    gauge_metadata_path = os.path.join(OBS_DATA_DIR, "gauge_metadata.csv")
    try:
        df_gauges = pd.read_csv(gauge_metadata_path)
        print(f"Loaded {len(df_gauges)} gauge metadata entries.")
    except Exception as e:
        print(f"Error loading gauge metadata from {gauge_metadata_path}: {e}")
        exit()

    # Map GRDC_NO to basin names for plotting
    grdc_to_name = df_gauges.set_index('gauge_id')['river_name'].to_dict()
    basin_names_map = {grdc_id: grdc_to_name.get(int(grdc_id), f"Basin {grdc_id}") for grdc_id in TARGET_BASINS.values()}

    clipped_data = {}
    basin_mean_values = {}

    for basin_name, grdc_no in TARGET_BASINS.items():
        print(f"Processing basin: {basin_name} (GRDC_NO: {grdc_no})")
        try:
            basin_gdf = gdf_basins[gdf_basins['grdc_no'] == int(grdc_no)]
            if basin_gdf.empty:
                print(f"Warning: Basin with GRDC_NO {grdc_no} not found in GeoJSON. Skipping.")
                continue

            basin_polygon = basin_gdf.geometry.iloc[0]

            basin_mean_values[basin_name] = {}
            for var_da in [P_clim, ET_clim, Q_clim, residual_clim]:
                if np.isnan(var_da.values).all():
                    clipped_data[f"{var_da.name}_{basin_name}"] = var_da.copy(data=np.full_like(var_da.values, np.nan))
                    basin_mean_values[basin_name][var_da.name] = np.nan
                    continue

                # Clip the data
                clipped_da = clip_to_basin(var_da, basin_polygon)
                clipped_data[f"{var_da.name}_{basin_name}"] = clipped_da

                # Calculate area-weighted mean for the basin
                # Need to mask area weights as well
                masked_area_weights = clip_to_basin(elm_area_weights, basin_polygon)
                if not np.isnan(clipped_da.values).all() and not np.isnan(masked_area_weights.values).all():
                    basin_mean = area_weighted_global_mean(clipped_da, masked_area_weights)
                    basin_mean_values[basin_name][var_da.name] = basin_mean.item()
                else:
                    basin_mean_values[basin_name][var_da.name] = np.nan
                print(f"  {var_da.name} mean for {basin_name}: {basin_mean_values[basin_name][var_da.name]:.2f} {var_da.units}")

        except Exception as e:
            print(f"Error processing basin {basin_name}: {e}")
            basin_mean_values[basin_name] = {v.name: np.nan for v in [P_clim, ET_clim, Q_clim, residual_clim]}

    # Save basin means to CSV
    try:
        basin_means_df = pd.DataFrame.from_dict(basin_mean_values, orient='index')
        basin_means_df.index.name = 'Basin'
        basin_means_df.columns.name = 'Variable (mm/day)'
        basin_means_csv_path = os.path.join(OUTPUT_DIR, "basin_water_balance_means.csv")
        basin_means_df.to_csv(basin_means_csv_path)
        print(f"Basin water balance means saved to {basin_means_csv_path}")
    except Exception as e:
        print(f"Error saving basin means to CSV: {e}")

    # 6. Produce a composite figure
    try:
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # Top: Global residual map with basin outlines
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_map.set_title(f'E3SM ELM Climatological Water Balance Residual (P - ET - Q) {START_YEAR}-{END_YEAR}')

        # Plot residual
        if not np.isnan(residual_clim.values).all():
            residual_clim.plot.contourf(ax=ax_map, transform=ccrs.PlateCarree(),
                                        levels=np.linspace(-5, 5, 21), cmap='RdBu',
                                        cbar_kwargs={'label': 'Residual (mm/day)'})
        else:
            ax_map.text(0.5, 0.5, "Residual data not available", transform=ax_map.transAxes,
                        ha='center', va='center', fontsize=12, color='red')

        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN)

        # Plot basin outlines
        for basin_name, grdc_no in TARGET_BASINS.items():
            basin_gdf = gdf_basins[gdf_basins['grdc_no'] == int(grdc_no)]
            if not basin_gdf.empty:
                ax_map.add_geometries(basin_gdf.geometry, crs=ccrs.PlateCarree(),
                                      edgecolor='blue', facecolor='none', linewidth=1.5,
                                      label=basin_name)
                # Add basin name label
                centroid = basin_gdf.geometry.centroid.iloc[0]
                ax_map.text(centroid.x, centroid.y, basin_name,
                            transform=ccrs.PlateCarree(),
                            fontsize=8, color='darkblue', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Bottom: Per-basin bar charts
        ax_bar = fig.add_subplot(gs[1])
        ax_bar.set_title('Area-Weighted Mean Water Balance Components per Basin')

        basin_names_ordered = list(TARGET_BASINS.keys())
        variables = ['P_clim', 'ET_clim', 'Q_clim', 'Residual_clim']
        colors = {'P_clim': 'skyblue', 'ET_clim': 'lightcoral', 'Q_clim': 'lightgreen', 'Residual_clim': 'gray'}
        labels = {'P_clim': 'P', 'ET_clim': 'ET', 'Q_clim': 'Q', 'Residual_clim': 'Residual'}

        bar_width = 0.2
        x = np.arange(len(basin_names_ordered))

        for i, var in enumerate(variables):
            values = [basin_mean_values[b].get(var, np.nan) for b in basin_names_ordered]
            ax_bar.bar(x + i * bar_width - (len(variables) - 1) * bar_width / 2,
                       values, bar_width, label=labels[var], color=colors[var])

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(basin_names_ordered, rotation=45, ha='right')
        ax_bar.set_ylabel('Mean Flux (mm/day)')
        ax_bar.legend(loc='upper right')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        composite_figure_path = os.path.join(OUTPUT_DIR, "water_balance_composite_figure.png")
        plt.savefig(composite_figure_path, dpi=300)
        print(f"Composite figure saved to {composite_figure_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error creating composite figure: {e}")

    print("Script finished.")