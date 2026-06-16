#!/usr/bin/env python3
import os
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def normalize_lon(lon_vals, target_range='grid'):
    lon_vals = np.array(lon_vals, dtype=float)
    return ((lon_vals + 180) % 360) - 180

def wrap_lon_for_grid(lon, grid_lons):
    # Determine grid longitude convention and wrap lon accordingly
    lon = float(lon)
    glon = np.asarray(grid_lons).flatten()
    if np.nanmax(glon) > 180:
        # grid is 0-360
        lon_wrapped = lon % 360
    else:
        # grid is -180..180
        lon_wrapped = ((lon + 180) % 360) - 180
    return lon_wrapped

def angular_distance_approx(lat1, lon1, lat2, lon2):
    # Approximate angular distance suitable for nearest-neighbor on lat/lon grids
    # Inputs can be arrays
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = (lon2r - lon1r + np.pi) % (2*np.pi) - np.pi
    a = dlat**2 + (np.cos(lat1r) * dlon)**2
    return a

def find_nearest_grid_indices(ds, pt_lat, pt_lon):
    if 'lat' not in ds.variables or 'lon' not in ds.variables:
        raise ValueError("Dataset missing 'lat' and 'lon' variables for grid mapping.")
    latv = ds['lat']
    lonv = ds['lon']

    # Wrap point longitude to grid convention
    pt_lon_wrapped = wrap_lon_for_grid(pt_lon, lonv.values)

    idx_map = {}
    grid_point = (np.nan, np.nan)

    if latv.ndim == 1 and lonv.ndim == 1:
        # Structured grid with 1D lat and lon
        lat_arr = latv.values
        lon_arr = lonv.values
        ilat = int(np.argmin(np.abs(lat_arr - pt_lat)))
        # For lon, account for wrap-around
        lon_wrapped_arr = np.array([wrap_lon_for_grid(l, lon_arr) for l in lon_arr])
        ilon = int(np.argmin(np.abs(lon_wrapped_arr - pt_lon_wrapped)))
        idx_map[latv.dims[0]] = ilat
        idx_map[lonv.dims[0]] = ilon
        grid_point = (float(lat_arr[ilat]), float(lon_wrapped_arr[ilon]))
    elif latv.ndim == 2 and lonv.ndim == 2:
        lat2d = latv.values
        lon2d = lonv.values
        # Wrap lon2d to -180..180 for distance calculation around pt_lon_wrapped
        lon2d_wrapped = np.array([[wrap_lon_for_grid(lv, lon2d) for lv in row] for row in lon2d])
        dist = angular_distance_approx(pt_lat, pt_lon_wrapped, lat2d, lon2d_wrapped)
        flat_index = np.argmin(dist)
        j, i = np.unravel_index(flat_index, lat2d.shape)
        dims = latv.dims  # two dims
        idx_map[dims[0]] = int(j)
        idx_map[dims[1]] = int(i)
        grid_point = (float(lat2d[j, i]), float(lon2d_wrapped[j, i]))
    else:
        raise ValueError("Unsupported grid: 'lat' and 'lon' must be 1D or 2D with matching shapes.")
    return idx_map, grid_point

def compute_metrics(sim, obs):
    # sim and obs are 1D numpy arrays with same length; may contain NaN
    mask = np.isfinite(sim) & np.isfinite(obs)
    simv = sim[mask]
    obsv = obs[mask]
    n = len(simv)
    out = {'RMSE': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'N': n}
    if n == 0:
        return out
    # RMSE
    out['RMSE'] = float(np.sqrt(np.mean((simv - obsv) ** 2)))
    # NSE
    denom = np.sum((obsv - np.mean(obsv)) ** 2)
    if denom > 0:
        out['NSE'] = float(1.0 - np.sum((simv - obsv) ** 2) / denom)
    else:
        out['NSE'] = np.nan
    # KGE
    if n >= 2 and np.std(obsv, ddof=1) > 0:
        r = np.corrcoef(simv, obsv)[0, 1] if n >= 2 else np.nan
        alpha = np.std(simv, ddof=1) / np.std(obsv, ddof=1) if np.std(obsv, ddof=1) != 0 else np.nan
        beta = np.mean(simv) / np.mean(obsv) if np.mean(obsv) != 0 else np.nan
        if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
            out['KGE'] = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        else:
            out['KGE'] = np.nan
    else:
        out['KGE'] = np.nan
    # PBIAS
    denom_sum = np.sum(obsv)
    if denom_sum != 0:
        out['PBIAS'] = float(100.0 * np.sum(simv - obsv) / denom_sum)
    else:
        out['PBIAS'] = np.nan
    return out

def read_geojson_polygons(geojson_path):
    features = {}
    try:
        with open(geojson_path, 'r') as f:
            gj = json.load(f)
        for feat in gj.get('features', []):
            props = feat.get('properties', {})
            gid = str(props.get('grdc_no'))
            geom = feat.get('geometry', {})
            gtype = geom.get('type')
            coords = geom.get('coordinates', [])
            polygon_list = []
            if gtype == 'Polygon':
                # coords: [ [ring1], [ring2], ... ]
                polygon_list.append(coords)
            elif gtype == 'MultiPolygon':
                # coords: [ [ [ring1], [ring2], ... ], [ ... ], ... ]
                for poly in coords:
                    polygon_list.append(poly)
            else:
                continue
            features[gid] = polygon_list
    except Exception as e:
        print(f"Error reading GeoJSON {geojson_path}: {e}")
    return features

def polygon_bounds(polygon_list):
    # polygon_list: list of polygons; each polygon is list of rings; ring is list of [lon,lat]
    lons = []
    lats = []
    for poly in polygon_list:
        for ring in poly:
            for pt in ring:
                if len(pt) >= 2:
                    lons.append(pt[0])
                    lats.append(pt[1])
    if len(lons) == 0:
        return None
    return (min(lons), max(lons), min(lats), max(lats))

def plot_basin_and_points(ax, polygon_list, gauge_lon, gauge_lat, model_lon, model_lat, basin_name):
    proj = ccrs.PlateCarree()
    # Plot polygons
    try:
        for poly in polygon_list:
            # Use only exterior ring (first ring) for plotting
            if len(poly) == 0:
                continue
            exterior = poly[0]
            lons = [pt[0] for pt in exterior]
            lats = [pt[1] for pt in exterior]
            ax.plot(lons, lats, transform=proj, color='k', linewidth=1)
            ax.fill(lons, lats, transform=proj, color='lightblue', alpha=0.3, zorder=1)
    except Exception as e:
        print(f"Error plotting polygon for {basin_name}: {e}")
    # Plot points
    try:
        ax.plot(gauge_lon, gauge_lat, marker='*', color='red', markersize=8, transform=proj, label='Gauge')
        ax.plot(model_lon, model_lat, marker='o', color='blue', markersize=6, transform=proj, label='Model grid')
    except Exception as e:
        print(f"Error plotting points for {basin_name}: {e}")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray')

def main():
    # Configuration
    case_name = "sample.v3.LR.historical"
    mosart_dir = "./data/sample/e3sm/rof"
    obs_stream_dir = "./data/sample/obs/streamflow"
    gauge_metadata_file = "./data/sample/obs/gauge_metadata.csv"
    basin_geojson_file = "./data/sample/obs/basin_polygons.geojson"
    target_variable = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    start_year, end_year = 1985, 1989

    # Output directory (user-specified absolute path)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_05_basin_streamflow/run2_output"
    ensure_dir(output_dir)

    # Basins of interest: name and gauge_id
    basins = [
        ("Amazon", "3629000"),
        ("Missouri", "4121801"),
        ("Columbia", "4115200"),
        ("Danube", "6742900"),
        ("Mekong", "2969100"),
        ("Orange", "1159100"),
    ]
    basin_ids = [gid for (_, gid) in basins]
    basin_names = {gid: name for (name, gid) in basins}

    # Gather files for MOSART monthly output over the period
    files = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            fpath = os.path.join(mosart_dir, f"{case_name}.mosart.h0.{y:04d}-{m:02d}.nc")
            if os.path.exists(fpath):
                files.append(fpath)
            else:
                print(f"Warning: missing MOSART file {fpath}")
    if len(files) == 0:
        print("No MOSART monthly files found for the specified period. Exiting.")
        return

    # Open dataset
    try:
        ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True)
    except Exception as e:
        print(f"Error opening MOSART dataset: {e}")
        return

    # Identify variable
    varname = None
    if target_variable in ds.data_vars:
        varname = target_variable
    else:
        candidates = [v for v in ds.data_vars if 'RIVER' in v.upper() and 'DISCH' in v.upper()]
        if len(candidates) > 0:
            varname = candidates[0]
            print(f"Using detected variable: {varname}")
        else:
            print("Could not find river discharge variable in dataset.")
            print("Available variables:", list(ds.data_vars))
            return

    # Time slice to ensure range is precise
    try:
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    except Exception as e:
        print(f"Warning: could not slice dataset by time: {e}")

    # Load gauge metadata
    try:
        gmeta = pd.read_csv(gauge_metadata_file)
    except Exception as e:
        print(f"Error reading gauge metadata {gauge_metadata_file}: {e}")
        return
    # Ensure gauge_id is string for matching
    gmeta['gauge_id'] = gmeta['gauge_id'].astype(str)
    meta_sel = gmeta[gmeta['gauge_id'].isin(basin_ids)].copy()
    if meta_sel.empty:
        print("No matching gauges found in metadata for the specified basins.")
        return
    meta_sel = meta_sel.set_index('gauge_id')

    # Read basin polygons
    basin_polys = read_geojson_polygons(basin_geojson_file)

    # Define monthly time axis
    months = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq='MS')

    # Prepare containers for outputs
    metrics_rows = []
    sim_data = pd.DataFrame(index=months)
    obs_data = pd.DataFrame(index=months)

    # Iterate basins
    for gid in basin_ids:
        basin_name = basin_names.get(gid, gid)
        print(f"Processing basin {basin_name} (Gauge {gid})")

        # Get metadata
        if gid not in meta_sel.index:
            print(f"Gauge {gid} not found in metadata; skipping.")
            continue
        row = meta_sel.loc[gid]
        gauge_lat = float(row['lat'])
        gauge_lon = float(row['lon'])

        # Find nearest grid cell
        try:
            idx_map, (grid_lat, grid_lon) = find_nearest_grid_indices(ds, gauge_lat, gauge_lon)
        except Exception as e:
            print(f"Error finding nearest grid cell for {gid}: {e}")
            continue

        # Extract model time series at nearest grid
        try:
            var = ds[varname]
            # Determine spatial dims present in variable
            spatial_dims = [d for d in var.dims if d != 'time']
            sel_dict = {}
            # Map indices based on dataset lat/lon dims
            for dname, idx in idx_map.items():
                if dname in spatial_dims:
                    sel_dict[dname] = idx
            # If variable has different spatial dims order, we still use isel with the dims present
            var_ts = var.isel(**sel_dict)
            # Ensure only time dimension remains
            if set(var_ts.dims) != {'time'}:
                # If there are leftover dims, attempt to squeeze
                var_ts = var_ts.squeeze()
                if set(var_ts.dims) != {'time'}:
                    # As a fallback, take first index along any remaining dims
                    for d in list(var_ts.dims):
                        if d != 'time':
                            var_ts = var_ts.isel({d: 0})
            sim_series = var_ts.to_series()
            # Convert to monthly by period and then to month-start timestamps
            sim_series = sim_series.groupby(sim_series.index.to_period('M')).mean()
            sim_series = sim_series.to_timestamp('MS')
            # Align to target months
            sim_series = sim_series.reindex(months)
        except Exception as e:
            print(f"Error extracting model series for {gid}: {e}")
            sim_series = pd.Series(index=months, dtype=float)

        # Read observed daily discharge and aggregate to monthly mean (m3/s)
        obs_series = pd.Series(index=months, dtype=float)
        obs_file = os.path.join(obs_stream_dir, f"{gid}.csv")
        try:
            odf = pd.read_csv(obs_file, parse_dates=['date'])
            odf = odf.set_index('date').sort_index()
            odf = odf.loc[(odf.index >= months[0]) & (odf.index <= (months[-1] + pd.offsets.MonthEnd(1) - pd.Timedelta(days=1)))]
            obs_monthly = odf['discharge_m3s'].resample('MS').mean()
            obs_series = obs_monthly.reindex(months)
        except Exception as e:
            print(f"Error reading observations for {gid} from {obs_file}: {e}")

        # Compute metrics
        sim_vals = sim_series.values.astype(float)
        obs_vals = obs_series.values.astype(float)
        met = compute_metrics(sim_vals, obs_vals)
        met_row = {
            'gauge_id': gid,
            'basin_name': basin_name,
            'RMSE_m3s': met['RMSE'],
            'NSE': met['NSE'],
            'KGE': met['KGE'],
            'PBIAS_percent': met['PBIAS'],
            'N_points': met['N'],
            'gauge_lat': gauge_lat,
            'gauge_lon': gauge_lon,
            'model_grid_lat': grid_lat,
            'model_grid_lon': grid_lon
        }
        metrics_rows.append(met_row)

        # Save per-basin time series CSV
        try:
            ts_df = pd.DataFrame({
                'sim_m3s': sim_series,
                'obs_m3s': obs_series
            }, index=months)
            ts_df.index.name = 'time'
            csv_path = os.path.join(output_dir, f"{gid}_{basin_name}_discharge_timeseries.csv")
            ts_df.to_csv(csv_path, float_format="%.6f")
        except Exception as e:
            print(f"Error saving time series CSV for {gid}: {e}")

        # Store for combined output
        sim_data[gid] = sim_series
        obs_data[gid] = obs_series

        # Plot figure for basin: map + time series
        try:
            fig = plt.figure(figsize=(12, 6))
            # Map subplot
            ax_map = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
            ax_map.set_title(f"{basin_name} Basin and Gauge")
            poly_list = basin_polys.get(str(gid), None)
            if poly_list is not None:
                bounds = polygon_bounds(poly_list)
                if bounds:
                    min_lon, max_lon, min_lat, max_lat = bounds
                    # Add margins
                    lon_margin = max(5, 0.1 * (max_lon - min_lon))
                    lat_margin = max(5, 0.1 * (max_lat - min_lat))
                    ax_map.set_extent([min_lon - lon_margin, max_lon + lon_margin,
                                       min_lat - lat_margin, max_lat + lat_margin], crs=ccrs.PlateCarree())
            plot_basin_and_points(ax_map, poly_list if poly_list is not None else [], gauge_lon, gauge_lat,
                                  grid_lon, grid_lat, basin_name)
            ax_map.legend(loc='lower left')

            # Time series subplot
            ax_ts = plt.subplot(1, 2, 2)
            ax_ts.plot(months, obs_series.values, label='Observed', color='black', linewidth=1.5)
            ax_ts.plot(months, sim_series.values, label='Simulated', color='tab:blue', linewidth=1.5)
            ax_ts.set_title(f"Monthly Discharge (1985-1989)")
            ax_ts.set_xlabel("Time")
            ax_ts.set_ylabel("Discharge (m3/s)")
            ax_ts.grid(True, linestyle='--', alpha=0.5)
            ax_ts.legend()
            # Metrics annotation
            textstr = (f"RMSE: {met_row['RMSE_m3s']:.2f}\n"
                       f"NSE: {met_row['NSE']:.3f}\n"
                       f"KGE: {met_row['KGE']:.3f}\n"
                       f"PBIAS: {met_row['PBIAS_percent']:.2f}%\n"
                       f"N: {met_row['N_points']}")
            ax_ts.text(0.02, 0.98, textstr, transform=ax_ts.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            fig.suptitle(f"{basin_name} (Gauge {gid})", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig_path = os.path.join(output_dir, f"{gid}_{basin_name}_streamflow.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting/saving figure for {gid}: {e}")

    # Save metrics summary CSV
    try:
        met_df = pd.DataFrame(metrics_rows)
        met_csv = os.path.join(output_dir, "validation_metrics_summary.csv")
        met_df.to_csv(met_csv, index=False, float_format="%.6f")
    except Exception as e:
        print(f"Error saving metrics summary CSV: {e}")

    # Save combined NetCDF with sim and obs time series
    try:
        # Ensure consistent column order for gauges
        gauges = [gid for (_, gid) in basins if gid in sim_data.columns]
        sim_aligned = sim_data[gauges]
        obs_aligned = obs_data[gauges]
        ds_out = xr.Dataset(
            data_vars=dict(
                sim_discharge=(['time', 'gauge'], sim_aligned.values),
                obs_discharge=(['time', 'gauge'], obs_aligned.values),
            ),
            coords=dict(
                time=months.to_pydatetime(),
                gauge=np.array(gauges, dtype=str),
            ),
            attrs=dict(
                title="Basin streamflow time series (monthly mean) for gauges",
                case_name=case_name,
                variable=varname,
                period=f"{start_year}-{end_year}",
                created=datetime.now().isoformat()
            )
        )
        ds_out['sim_discharge'].attrs.update(units='m3 s-1', description='Simulated river discharge at nearest MOSART grid cell')
        ds_out['obs_discharge'].attrs.update(units='m3 s-1', description='Observed river discharge (monthly mean of daily)')
        # Add gauge metadata as auxiliary variables
        gauge_meta_aux = []
        for gid in gauges:
            if gid in meta_sel.index:
                gauge_meta_aux.append((gid, float(meta_sel.loc[gid, 'lat']), float(meta_sel.loc[gid, 'lon']), basin_names.get(gid, gid)))
            else:
                gauge_meta_aux.append((gid, np.nan, np.nan, basin_names.get(gid, gid)))
        lat_arr = np.array([v[1] for v in gauge_meta_aux], dtype=float)
        lon_arr = np.array([v[2] for v in gauge_meta_aux], dtype=float)
        name_arr = np.array([v[3] for v in gauge_meta_aux], dtype=str)

        ds_out['gauge_lat'] = xr.DataArray(lat_arr, dims=['gauge'])
        ds_out['gauge_lon'] = xr.DataArray(lon_arr, dims=['gauge'])
        ds_out['basin_name'] = xr.DataArray(name_arr, dims=['gauge'])
        nc_path = os.path.join(output_dir, "gauge_streamflow_timeseries.nc")
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_out.data_vars}
        ds_out.to_netcdf(nc_path, encoding=encoding)
    except Exception as e:
        print(f"Error saving NetCDF time series: {e}")

    print("Analysis completed.")

if __name__ == "__main__":
    main()