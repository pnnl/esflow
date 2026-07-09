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

def wrap_lon(lon):
    return ((lon + 180) % 360) - 180

def generate_month_list(start_year, start_month, end_year, end_month):
    dates = pd.date_range(f"{start_year:04d}-{start_month:02d}-01", f"{end_year:04d}-{end_month:02d}-01", freq="MS")
    return [(d.year, d.month) for d in dates]

def load_geojson_polygons(path):
    polygons_by_id = {}
    try:
        with open(path, 'r') as f:
            gj = json.load(f)
    except Exception as e:
        print(f"Error reading GeoJSON {path}: {e}")
        return polygons_by_id
    features = gj.get('features', [])
    for feat in features:
        props = feat.get('properties', {})
        gid = props.get('grdc_no')
        if gid is None:
            continue
        gid = str(gid)
        geom = feat.get('geometry', {})
        gtype = geom.get('type')
        coords = geom.get('coordinates', [])
        rings_list = []
        if gtype == 'Polygon':
            # coords is list of linear rings
            for ring in coords:
                rings_list.append(ring)
        elif gtype == 'MultiPolygon':
            for poly in coords:
                for ring in poly:
                    rings_list.append(ring)
        else:
            continue
        # Store as list of rings (each ring is list of [lon,lat] points)
        polygons_by_id[gid] = rings_list
    return polygons_by_id

def haversine_distance(lat1, lon1, lat2, lon2):
    # lat/lon in degrees, returns distance in meters
    R = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def find_nearest_grid_index(lat_da, lon_da, target_lat, target_lon):
    # Ensure arrays are numpy arrays
    lat_arr = np.array(lat_da)
    lon_arr = np.array(lon_da)
    # Wrap longitudes to [-180,180]
    lon_arr_wrapped = wrap_lon(lon_arr)
    target_lon_wrapped = wrap_lon(target_lon)
    # Compute simple great-circle distance approximation using haversine on flattened arrays
    flat_lat = lat_arr.ravel()
    flat_lon = lon_arr_wrapped.ravel()
    # Use approximate metric to avoid heavy computation; but we can vectorize haversine
    # Vectorized haversine
    R = 6371000.0
    phi1 = np.deg2rad(target_lat)
    phi2 = np.deg2rad(flat_lat)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(flat_lon - target_lon_wrapped)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist = R * c
    idx_flat = np.nanargmin(dist)
    return np.unravel_index(idx_flat, lat_arr.shape)

def compute_metrics(sim, obs):
    # Expect pandas Series with same datetime index; drop NaNs
    df = pd.DataFrame({'sim': sim, 'obs': obs}).dropna()
    if df.empty:
        return {'RMSE': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'N': 0}
    s = df['sim'].values
    o = df['obs'].values
    n = len(df)
    rmse = np.sqrt(np.mean((s - o) ** 2))
    # NSE
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = np.nan
    if denom > 0:
        nse = 1 - np.sum((s - o) ** 2) / denom
    # KGE
    r = np.nan
    try:
        if np.std(s, ddof=1) > 0 and np.std(o, ddof=1) > 0:
            r = np.corrcoef(s, o)[0, 1]
        else:
            r = np.nan
    except Exception:
        r = np.nan
    alpha = np.std(s, ddof=1) / np.std(o, ddof=1) if np.std(o, ddof=1) != 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge = np.nan
    else:
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    # PBIAS
    if np.sum(o) != 0:
        pbias = 100.0 * np.sum(s - o) / np.sum(o)
    else:
        pbias = np.nan
    return {'RMSE': rmse, 'NSE': nse, 'KGE': kge, 'PBIAS': pbias, 'N': n}

def build_mosart_dataset(rof_dir, case_name, start_year, start_month, end_year, end_month):
    months = generate_month_list(start_year, start_month, end_year, end_month)
    file_list = []
    for y, m in months:
        fpath = os.path.join(rof_dir, f"{case_name}.mosart.h0.{y:04d}-{m:02d}.nc")
        if os.path.exists(fpath):
            file_list.append(fpath)
        else:
            print(f"Warning: Missing MOSART monthly file {fpath}")
    if len(file_list) == 0:
        print("Error: No MOSART monthly files found for the specified period.")
        return None, None
    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords', decode_times=True)
    except Exception as e:
        print(f"Error opening MOSART files: {e}")
        return None, None
    # Ensure time is sorted
    if 'time' in ds:
        ds = ds.sortby('time')
    # Get lat/lon
    lat = ds.get('lat', None)
    lon = ds.get('lon', None)
    if lat is None or lon is None:
        print("Error: lat/lon not found in MOSART dataset.")
        return None, None
    return ds, (lat, lon)

def extract_sim_timeseries(ds, lat_da, lon_da, target_lat, target_lon, var_candidates=None):
    if var_candidates is None:
        var_candidates = [
            'RIVER_DISCHARGE_OVER_LAND_LIQ',
            'RIVER_DISCHARGE_OVER_LAND',
            'RIVER_DISCHARGE',
            'discharge',
            'RIVER_DISCHARGE_LIQ'
        ]
    var_name = None
    for vn in var_candidates:
        if vn in ds.variables:
            var_name = vn
            break
    if var_name is None:
        print(f"Error: None of the candidate variables found in dataset: {var_candidates}")
        return None, None, None
    grid_index = find_nearest_grid_index(lat_da, lon_da, target_lat, target_lon)
    # Find time dimension name
    var = ds[var_name]
    dims = list(var.dims)
    time_dim = None
    for d in dims:
        if 'time' in d:
            time_dim = d
            break
    if time_dim is None and 'time' in ds.dims:
        time_dim = 'time'
    # Determine spatial dims
    space_dims = [d for d in dims if d != time_dim]
    sel_dict = {}
    if len(space_dims) == 2:
        sel_dict[space_dims[0]] = grid_index[0]
        sel_dict[space_dims[1]] = grid_index[1]
    elif len(space_dims) == 1:
        # Unstructured grid case
        # grid_index will be tuple with one element if lat/lon 1D
        if isinstance(grid_index, tuple) and len(grid_index) == 1:
            sel_dict[space_dims[0]] = grid_index[0]
        else:
            # Flattened index can be used directly if needed
            sel_dict[space_dims[0]] = np.ravel_multi_index(grid_index, np.array(lat_da).shape)
    else:
        print(f"Unexpected number of spatial dims for variable {var_name}: {space_dims}")
        return None, None, None
    try:
        ts_da = var.isel(**sel_dict)
    except Exception as e:
        print(f"Error selecting data at grid index {grid_index}: {e}")
        return None, None, None
    # Extract grid cell center lat/lon for reporting
    try:
        grid_lat = float(np.array(lat_da)[grid_index])
        grid_lon = float(np.array(lon_da)[grid_index])
    except Exception:
        grid_lat = np.nan
        grid_lon = np.nan
    return ts_da, grid_index, (grid_lat, grid_lon)

def resample_to_monthly(series):
    # series is pandas Series with datetime index
    # Convert to monthly mean at month start
    s = series.copy()
    s = s.sort_index()
    monthly = s.resample('MS').mean()
    return monthly

def ensure_output_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {path}: {e}")

def plot_basin_map_and_timeseries(basin_name, gauge_id, basin_rings, gauge_lat, gauge_lon, grid_lat, grid_lon, ts_obs, ts_sim, out_png):
    try:
        fig = plt.figure(figsize=(12, 6))
        # Map subplot
        ax_map = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax_map.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=0)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        # Plot basin boundary
        if basin_rings:
            # Compute bounds
            lons_all = []
            lats_all = []
            for ring in basin_rings:
                xs = [wrap_lon(pt[0]) for pt in ring]
                ys = [pt[1] for pt in ring]
                lons_all.extend(xs)
                lats_all.extend(ys)
                ax_map.plot(xs, ys, color='tab:green', linewidth=1.0, transform=ccrs.PlateCarree(), zorder=2)
            if len(lons_all) > 0 and len(lats_all) > 0:
                lon_min, lon_max = min(lons_all), max(lons_all)
                lat_min, lat_max = min(lats_all), max(lats_all)
                # Ensure reasonable extent padding
                pad_lon = max(2.0, (lon_max - lon_min) * 0.1)
                pad_lat = max(2.0, (lat_max - lat_min) * 0.1)
                ax_map.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        # Plot gauge and grid cell
        ax_map.plot(wrap_lon(gauge_lon), gauge_lat, 'o', color='red', markersize=6, transform=ccrs.PlateCarree(), label='Gauge', zorder=3)
        if np.isfinite(grid_lat) and np.isfinite(grid_lon):
            ax_map.plot(wrap_lon(grid_lon), grid_lat, 'x', color='blue', markersize=6, transform=ccrs.PlateCarree(), label='Nearest Grid', zorder=3)
        ax_map.legend(loc='lower left')
        ax_map.set_title(f"{basin_name} Basin (Gauge {gauge_id})")

        # Time series subplot
        ax_ts = plt.subplot(1, 2, 2)
        # Align time series
        df = pd.DataFrame({'Sim': ts_sim, 'Obs': ts_obs})
        df = df.sort_index()
        ax_ts.plot(df.index, df['Obs'], label='Observed', color='black', linewidth=1.5)
        ax_ts.plot(df.index, df['Sim'], label='Simulated', color='tab:blue', linewidth=1.5)
        ax_ts.set_title("Monthly Mean Discharge (m3/s)")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Discharge (m3/s)")
        ax_ts.grid(True, linestyle='--', alpha=0.4)
        ax_ts.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {out_png}")
    except Exception as e:
        print(f"Error plotting/saving figure {out_png}: {e}")

def main():
    # Paths and configuration
    case_name = "sample.v3.LR.historical"
    data_base = "./data/sample"
    rof_dir = os.path.join(data_base, "e3sm", "rof")
    obs_gauge_meta = os.path.join(data_base, "obs", "gauge_metadata.csv")
    obs_streamflow_dir = os.path.join(data_base, "obs", "streamflow")
    basin_geojson = os.path.join(data_base, "obs", "basin_polygons.geojson")

    # Requested output directory
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_05_basin_streamflow/run1_output"
    ensure_output_dir(output_dir)

    # Analysis period
    start_year, start_month = 1985, 1
    end_year, end_month = 1989, 12

    # Target basins with gauge IDs
    basins = [
        {"name": "Amazon", "gauge_id": "3629000"},
        {"name": "Missouri", "gauge_id": "4121801"},
        {"name": "Columbia", "gauge_id": "4115200"},
        {"name": "Danube", "gauge_id": "6742900"},
        {"name": "Mekong", "gauge_id": "2969100"},
        {"name": "Orange", "gauge_id": "1159100"},
    ]

    # Load gauge metadata
    try:
        gauge_meta = pd.read_csv(obs_gauge_meta)
        # Ensure gauge_id is string
        if 'gauge_id' in gauge_meta.columns:
            gauge_meta['gauge_id'] = gauge_meta['gauge_id'].astype(str)
        else:
            print(f"Error: gauge_metadata.csv missing 'gauge_id' column.")
            gauge_meta = pd.DataFrame()
    except Exception as e:
        print(f"Error reading gauge metadata {obs_gauge_meta}: {e}")
        return

    # Load basin polygons
    polygons_by_id = load_geojson_polygons(basin_geojson)
    if not polygons_by_id:
        print("Warning: No polygons loaded or error reading GeoJSON. Map boundaries may be missing.")

    # Load MOSART dataset once
    ds, (lat_da, lon_da) = build_mosart_dataset(rof_dir, case_name, start_year, start_month, end_year, end_month)
    if ds is None:
        print("Aborting: Unable to load MOSART dataset.")
        return

    # Prepare outputs
    metrics_records = []
    grid_match_rows = []

    for basin in basins:
        basin_name = basin['name']
        gauge_id = str(basin['gauge_id'])
        # Get gauge metadata row
        meta_row = None
        if not gauge_meta.empty:
            # Some gauge_id in metadata may have leading zeros; try match by casting to int then back
            meta_row = gauge_meta[gauge_meta['gauge_id'] == gauge_id]
            if meta_row.empty:
                # Try removing leading zeros
                try:
                    gid_int = int(gauge_id)
                    meta_row = gauge_meta[gauge_meta['gauge_id'].astype(str).str.lstrip('0') == str(gid_int)]
                except Exception:
                    pass
        if meta_row is None or meta_row.empty:
            print(f"Warning: Gauge metadata not found for gauge_id {gauge_id}")
            continue
        meta_row = meta_row.iloc[0]
        gauge_lat = float(meta_row['lat']) if 'lat' in meta_row else np.nan
        gauge_lon = float(meta_row['lon']) if 'lon' in meta_row else np.nan
        river_name = meta_row['river_name'] if 'river_name' in meta_row and isinstance(meta_row['river_name'], str) else basin_name

        # Load obs for gauge
        obs_file = os.path.join(obs_streamflow_dir, f"{gauge_id}.csv")
        try:
            obs_df = pd.read_csv(obs_file, parse_dates=['date'])
        except Exception as e:
            print(f"Error reading obs file {obs_file}: {e}")
            continue
        if 'discharge_m3s' not in obs_df.columns or 'date' not in obs_df.columns:
            print(f"Observation file {obs_file} missing required columns.")
            continue
        obs_df = obs_df.set_index('date').sort_index()
        # Filter period
        period_start = pd.Timestamp(f"{start_year:04d}-{start_month:02d}-01")
        period_end = pd.Timestamp(f"{end_year:04d}-{end_month:02d}-01") + pd.offsets.MonthEnd(1)
        obs_df = obs_df[(obs_df.index >= period_start) & (obs_df.index <= period_end)]
        # Resample to monthly mean
        obs_monthly = resample_to_monthly(obs_df['discharge_m3s'])

        # Extract sim timeseries
        ts_da, grid_index, (grid_lat, grid_lon) = extract_sim_timeseries(ds, lat_da, lon_da, gauge_lat, gauge_lon)
        if ts_da is None:
            print(f"Skipping basin {basin_name} ({gauge_id}) due to extraction error.")
            continue
        # Convert to pandas series
        try:
            sim_series = ts_da.to_pandas()
        except Exception:
            # Alternative conversion
            times = pd.to_datetime(ts_da[ts_da.dims[0]].values)
            sim_series = pd.Series(ts_da.values, index=times)
        # Subset to period
        sim_series = sim_series[(sim_series.index >= period_start) & (sim_series.index <= period_end)]
        # Ensure monthly and align index to month start
        sim_series_monthly = sim_series.resample('MS').mean()

        # Align both time series
        combined = pd.DataFrame({
            'obs_m3s': obs_monthly,
            'sim_m3s': sim_series_monthly
        }).dropna()

        # Compute metrics
        metrics = compute_metrics(combined['sim_m3s'], combined['obs_m3s'])
        metrics_record = {
            'basin': basin_name,
            'gauge_id': gauge_id,
            'river_name': river_name,
            'grid_i': grid_index[0] if isinstance(grid_index, tuple) and len(grid_index) > 0 else grid_index,
            'grid_j': grid_index[1] if isinstance(grid_index, tuple) and len(grid_index) > 1 else np.nan,
            'grid_lat': grid_lat,
            'grid_lon': grid_lon,
            'gauge_lat': gauge_lat,
            'gauge_lon': gauge_lon,
            'RMSE': metrics['RMSE'],
            'NSE': metrics['NSE'],
            'KGE': metrics['KGE'],
            'PBIAS': metrics['PBIAS'],
            'N_months': metrics['N']
        }
        metrics_records.append(metrics_record)
        grid_match_rows.append({
            'basin': basin_name,
            'gauge_id': gauge_id,
            'grid_i': metrics_record['grid_i'],
            'grid_j': metrics_record['grid_j'],
            'grid_lat': grid_lat,
            'grid_lon': grid_lon
        })

        # Save per-basin time series CSV
        ts_out = combined.copy()
        ts_out.index.name = 'date'
        ts_out_file = os.path.join(output_dir, f"{gauge_id}_{basin_name.replace(' ','_')}_timeseries_1985-1989.csv")
        try:
            ts_out.to_csv(ts_out_file, float_format="%.6f")
            print(f"Saved time series CSV: {ts_out_file}")
        except Exception as e:
            print(f"Error saving time series CSV {ts_out_file}: {e}")

        # Plot figure
        rings = polygons_by_id.get(gauge_id)
        if rings is None:
            # Try matching with stripped leading zeros
            rings = None
            try:
                gid_int = int(gauge_id)
                for key in polygons_by_id.keys():
                    if key.lstrip('0') == str(gid_int):
                        rings = polygons_by_id[key]
                        break
            except Exception:
                pass
        fig_file = os.path.join(output_dir, f"{gauge_id}_{basin_name.replace(' ','_')}_map_timeseries.png")
        plot_basin_map_and_timeseries(basin_name, gauge_id, rings, gauge_lat, gauge_lon, grid_lat, grid_lon, obs_monthly, sim_series_monthly, fig_file)

    # Save summary metrics CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_file = os.path.join(output_dir, "basin_streamflow_metrics_1985-1989.csv")
    try:
        metrics_df.to_csv(metrics_file, index=False, float_format="%.6f")
        print(f"Saved metrics CSV: {metrics_file}")
    except Exception as e:
        print(f"Error saving metrics CSV {metrics_file}: {e}")

    # Save grid match info CSV
    grid_match_df = pd.DataFrame(grid_match_rows)
    grid_match_file = os.path.join(output_dir, "gauge_to_grid_match.csv")
    try:
        grid_match_df.to_csv(grid_match_file, index=False)
        print(f"Saved grid match CSV: {grid_match_file}")
    except Exception as e:
        print(f"Error saving grid match CSV {grid_match_file}: {e}")

    # Optionally save a NetCDF with all simulated time series for the basins
    try:
        ds_out_list = []
        for rec in metrics_records:
            gauge_id = rec['gauge_id']
            basin_name = rec['basin']
            ts_csv = os.path.join(output_dir, f"{gauge_id}_{basin_name.replace(' ','_')}_timeseries_1985-1989.csv")
            if not os.path.exists(ts_csv):
                continue
            df = pd.read_csv(ts_csv, parse_dates=['date'])
            df = df.set_index('date')
            da_sim = xr.DataArray(
                data=df['sim_m3s'].values,
                coords={'time': df.index.values, 'gauge_id': gauge_id},
                dims=['time'],
                name=f"sim_{gauge_id}"
            )
            da_obs = xr.DataArray(
                data=df['obs_m3s'].values,
                coords={'time': df.index.values, 'gauge_id': gauge_id},
                dims=['time'],
                name=f"obs_{gauge_id}"
            )
            ds_tmp = xr.Dataset({f"sim_{gauge_id}": da_sim, f"obs_{gauge_id}": da_obs})
            # add attrs
            ds_tmp.attrs['basin_name'] = basin_name
            ds_tmp.attrs['gauge_id'] = gauge_id
            ds_out_list.append(ds_tmp)
        if ds_out_list:
            ds_all = xr.merge(ds_out_list, compat='no_conflicts')
            nc_file = os.path.join(output_dir, "basin_streamflow_timeseries_1985-1989.nc")
            ds_all.to_netcdf(nc_file)
            print(f"Saved NetCDF: {nc_file}")
    except Exception as e:
        print(f"Error saving NetCDF of time series: {e}")

if __name__ == "__main__":
    main()