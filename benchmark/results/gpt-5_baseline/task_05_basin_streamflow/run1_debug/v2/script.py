#!/usr/bin/env python3
import os
import json
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
            for ring in coords:
                rings_list.append(ring)
        elif gtype == 'MultiPolygon':
            for poly in coords:
                for ring in poly:
                    rings_list.append(ring)
        else:
            continue
        polygons_by_id[gid] = rings_list
    return polygons_by_id

def compute_metrics(sim, obs):
    df = pd.DataFrame({'sim': sim, 'obs': obs}).dropna()
    if df.empty:
        return {'RMSE': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'N': 0}
    s = df['sim'].values
    o = df['obs'].values
    rmse = np.sqrt(np.mean((s - o) ** 2))
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = np.nan
    if denom > 0:
        nse = 1 - np.sum((s - o) ** 2) / denom
    try:
        r = np.corrcoef(s, o)[0, 1] if np.std(s) > 0 and np.std(o) > 0 else np.nan
    except Exception:
        r = np.nan
    alpha = np.std(s) / np.std(o) if np.std(o) != 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge = np.nan
    else:
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    if np.sum(o) != 0:
        pbias = 100.0 * np.sum(s - o) / np.sum(o)
    else:
        pbias = np.nan
    return {'RMSE': rmse, 'NSE': nse, 'KGE': kge, 'PBIAS': pbias, 'N': len(df)}

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
        return None, (None, None)
    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords', decode_times=True)
    except Exception as e:
        print(f"Error opening MOSART files: {e}")
        return None, (None, None)
    if 'time' in ds:
        ds = ds.sortby('time')
    lat = ds.get('lat', None)
    lon = ds.get('lon', None)
    if lat is None or lon is None:
        print("Error: lat/lon not found in MOSART dataset.")
        return None, (None, None)
    return ds, (lat, lon)

def find_nearest_grid_selection(lat_da, lon_da, target_lat, target_lon):
    try:
        lat2d, lon2d = xr.broadcast(lat_da, lon_da)
    except Exception as e:
        try:
            lat_arr = np.array(lat_da)
            lon_arr = np.array(lon_da)
            if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                lon2d_vals, lat2d_vals = np.meshgrid(lon_arr, lat_arr)
                lat2d = xr.DataArray(lat2d_vals, dims=('lat_dim', 'lon_dim'))
                lon2d = xr.DataArray(lon2d_vals, dims=('lat_dim', 'lon_dim'))
            else:
                raise RuntimeError("Unable to broadcast lat/lon arrays")
        except Exception as e2:
            print(f"Error broadcasting lat/lon arrays: {e2}")
            return None, None, None, np.nan, np.nan
    lat_vals = lat2d.values
    lon_vals = wrap_lon(lon2d.values)
    target_lon_wrapped = wrap_lon(target_lon)
    R = 6371000.0
    phi1 = np.deg2rad(target_lat)
    phi2 = np.deg2rad(lat_vals)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon_vals - target_lon_wrapped)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c
    try:
        idx_flat = np.nanargmin(dist)
    except Exception as e:
        print(f"Error computing nearest grid cell: {e}")
        return None, None, None, np.nan, np.nan
    idx_multi = np.unravel_index(idx_flat, dist.shape)
    sel_dict = {dim: idx_multi[i] for i, dim in enumerate(lat2d.dims)}
    grid_lat = float(lat_vals[idx_multi])
    grid_lon = float(lon2d.values[idx_multi])
    return sel_dict, idx_multi, lat2d.dims, grid_lat, grid_lon

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
        return None, None, (np.nan, np.nan)
    sel_dict_latlon, idx_multi, grid_dims, grid_lat, grid_lon = find_nearest_grid_selection(lat_da, lon_da, target_lat, target_lon)
    if sel_dict_latlon is None:
        return None, None, (np.nan, np.nan)
    var = ds[var_name]
    sel_dict_var = {dim: sel_dict_latlon[dim] for dim in sel_dict_latlon if dim in var.dims}
    try:
        ts_da = var.isel(**sel_dict_var)
    except Exception as e:
        print(f"Error selecting data using selection {sel_dict_var}: {e}")
        return None, None, (np.nan, np.nan)
    return ts_da, (idx_multi if isinstance(idx_multi, tuple) else (idx_multi,)), (grid_lat, grid_lon)

def ts_da_to_monthly_period_series(ts_da, start_year, end_year):
    # Subset by years using string slicing to handle cftime calendars
    try:
        ts_sel = ts_da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    except Exception:
        ts_sel = ts_da
    # Build PeriodIndex from year and month, then average duplicates (e.g., daily -> monthly)
    try:
        years = ts_sel['time'].dt.year.values.astype(int)
        months = ts_sel['time'].dt.month.values.astype(int)
    except Exception as e:
        # Fallback: try to convert to pandas datetime, then extract
        try:
            times = ts_sel.indexes['time'].to_datetimeindex()
            years = times.year
            months = times.month
        except Exception as e2:
            print(f"Error extracting year/month from time coordinate: {e2}")
            return pd.Series(dtype=float)
    periods = pd.PeriodIndex(year=years, month=months, freq='M')
    values = np.array(ts_sel.values).astype(float)
    ser = pd.Series(values, index=periods)
    ser = ser.groupby(level=0).mean().sort_index()
    # Keep only within desired year range
    ser = ser[(ser.index.year >= start_year) & (ser.index.year <= end_year)]
    return ser

def resample_obs_to_monthly_period(obs_df, start_year, end_year):
    obs_monthly = obs_df.resample('MS').mean()
    obs_period = obs_monthly.copy()
    try:
        obs_period.index = obs_period.index.to_period('M')
    except Exception:
        # Fallback: construct PeriodIndex manually
        years = obs_monthly.index.year.values
        months = obs_monthly.index.month.values
        obs_period.index = pd.PeriodIndex(year=years, month=months, freq='M')
    obs_period = obs_period[(obs_period.index.year >= start_year) & (obs_period.index.year <= end_year)]
    return obs_period

def ensure_output_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {path}: {e}")

def plot_basin_map_and_timeseries(basin_name, gauge_id, basin_rings, gauge_lat, gauge_lon, grid_lat, grid_lon, ts_obs_period, ts_sim_period, out_png):
    try:
        # Convert PeriodIndex to Timestamp for plotting
        obs_plot = ts_obs_period.copy()
        sim_plot = ts_sim_period.copy()
        if isinstance(obs_plot.index, pd.PeriodIndex):
            obs_plot.index = obs_plot.index.to_timestamp(how='start')
        if isinstance(sim_plot.index, pd.PeriodIndex):
            sim_plot.index = sim_plot.index.to_timestamp(how='start')

        fig = plt.figure(figsize=(12, 6))
        ax_map = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax_map.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=0)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        if basin_rings:
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
                pad_lon = max(2.0, (lon_max - lon_min) * 0.1)
                pad_lat = max(2.0, (lat_max - lat_min) * 0.1)
                ax_map.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        ax_map.plot(wrap_lon(gauge_lon), gauge_lat, 'o', color='red', markersize=6, transform=ccrs.PlateCarree(), label='Gauge', zorder=3)
        if np.isfinite(grid_lat) and np.isfinite(grid_lon):
            ax_map.plot(wrap_lon(grid_lon), grid_lat, 'x', color='blue', markersize=6, transform=ccrs.PlateCarree(), label='Nearest Grid', zorder=3)
        ax_map.legend(loc='lower left')
        ax_map.set_title(f"{basin_name} Basin (Gauge {gauge_id})")

        ax_ts = plt.subplot(1, 2, 2)
        df = pd.DataFrame({'Observed': obs_plot, 'Simulated': sim_plot}).sort_index()
        ax_ts.plot(df.index, df['Observed'], label='Observed', color='black', linewidth=1.5)
        ax_ts.plot(df.index, df['Simulated'], label='Simulated', color='tab:blue', linewidth=1.5)
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
    case_name = "sample.v3.LR.historical"
    data_base = "./data/sample"
    rof_dir = os.path.join(data_base, "e3sm", "rof")
    obs_gauge_meta = os.path.join(data_base, "obs", "gauge_metadata.csv")
    obs_streamflow_dir = os.path.join(data_base, "obs", "streamflow")
    basin_geojson = os.path.join(data_base, "obs", "basin_polygons.geojson")

    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_05_basin_streamflow/run1_debug/v2/output"
    ensure_output_dir(output_dir)

    start_year, start_month = 1985, 1
    end_year, end_month = 1989, 12

    basins = [
        {"name": "Amazon", "gauge_id": "3629000"},
        {"name": "Missouri", "gauge_id": "4121801"},
        {"name": "Columbia", "gauge_id": "4115200"},
        {"name": "Danube", "gauge_id": "6742900"},
        {"name": "Mekong", "gauge_id": "2969100"},
        {"name": "Orange", "gauge_id": "1159100"},
    ]

    try:
        gauge_meta = pd.read_csv(obs_gauge_meta)
        if 'gauge_id' in gauge_meta.columns:
            gauge_meta['gauge_id'] = gauge_meta['gauge_id'].astype(str)
        else:
            print(f"Error: gauge_metadata.csv missing 'gauge_id' column.")
            gauge_meta = pd.DataFrame()
    except Exception as e:
        print(f"Error reading gauge metadata {obs_gauge_meta}: {e}")
        return

    polygons_by_id = load_geojson_polygons(basin_geojson)
    if not polygons_by_id:
        print("Warning: No polygons loaded or error reading GeoJSON. Map boundaries may be missing.")

    ds, (lat_da, lon_da) = build_mosart_dataset(rof_dir, case_name, start_year, start_month, end_year, end_month)
    if ds is None or lat_da is None or lon_da is None:
        print("Aborting: Unable to load MOSART dataset.")
        return

    metrics_records = []
    grid_match_rows = []

    for basin in basins:
        basin_name = basin['name']
        gauge_id = str(basin['gauge_id'])
        meta_row = None
        if not gauge_meta.empty:
            meta_row = gauge_meta[gauge_meta['gauge_id'] == gauge_id]
            if meta_row.empty:
                try:
                    gid_int = int(gauge_id)
                    meta_row = gauge_meta[gauge_meta['gauge_id'].astype(str).str.lstrip('0') == str(gid_int)]
                except Exception:
                    pass
        if meta_row is None or meta_row.empty:
            print(f"Warning: Gauge metadata not found for gauge_id {gauge_id}")
            continue
        meta_row = meta_row.iloc[0]
        try:
            gauge_lat = float(meta_row['lat'])
            gauge_lon = float(meta_row['lon'])
        except Exception:
            print(f"Warning: Invalid lat/lon for gauge_id {gauge_id}")
            continue
        river_name = meta_row['river_name'] if 'river_name' in meta_row and isinstance(meta_row['river_name'], str) else basin_name

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
        # Filter to the overall time span first
        obs_df = obs_df[(obs_df.index >= pd.Timestamp(f"{start_year}-01-01")) & (obs_df.index <= pd.Timestamp(f"{end_year}-12-31"))]
        obs_monthly_period = resample_obs_to_monthly_period(obs_df['discharge_m3s'], start_year, end_year)

        ts_da, grid_index, (grid_lat, grid_lon) = extract_sim_timeseries(ds, lat_da, lon_da, gauge_lat, gauge_lon)
        if ts_da is None:
            print(f"Skipping basin {basin_name} ({gauge_id}) due to extraction error.")
            continue

        # Convert model time series to monthly Period series
        sim_series_period = ts_da_to_monthly_period_series(ts_da, start_year, end_year)

        # Align both by PeriodIndex
        common_idx = obs_monthly_period.index.intersection(sim_series_period.index)
        obs_aligned = obs_monthly_period.reindex(common_idx)
        sim_aligned = sim_series_period.reindex(common_idx)

        combined = pd.DataFrame({
            'obs_m3s': obs_aligned,
            'sim_m3s': sim_aligned
        }).dropna()

        metrics = compute_metrics(combined['sim_m3s'], combined['obs_m3s'])
        metrics_record = {
            'basin': basin_name,
            'gauge_id': gauge_id,
            'river_name': river_name,
            'grid_i': grid_index[0] if isinstance(grid_index, (tuple, list)) and len(grid_index) > 0 else np.nan,
            'grid_j': grid_index[1] if isinstance(grid_index, (tuple, list)) and len(grid_index) > 1 else np.nan,
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

        # Save per-basin time series CSV with timestamp index for readability
        ts_out = combined.copy()
        ts_out.index.name = 'period'
        try:
            ts_out_ts = ts_out.copy()
            if isinstance(ts_out_ts.index, pd.PeriodIndex):
                ts_out_ts = ts_out_ts.copy()
                ts_out_ts['date'] = ts_out_ts.index.to_timestamp(how='start')
                ts_out_ts = ts_out_ts.set_index('date')
        except Exception:
            pass
        ts_out_file = os.path.join(output_dir, f"{gauge_id}_{basin_name.replace(' ','_')}_timeseries_1985-1989.csv")
        try:
            ts_out_ts.to_csv(ts_out_file, float_format="%.6f")
            print(f"Saved time series CSV: {ts_out_file}")
        except Exception as e:
            print(f"Error saving time series CSV {ts_out_file}: {e}")

        rings = polygons_by_id.get(gauge_id)
        if rings is None:
            try:
                gid_int = int(gauge_id)
                for key in polygons_by_id.keys():
                    if key.lstrip('0') == str(gid_int):
                        rings = polygons_by_id[key]
                        break
            except Exception:
                pass
        fig_file = os.path.join(output_dir, f"{gauge_id}_{basin_name.replace(' ','_')}_map_timeseries.png")
        plot_basin_map_and_timeseries(basin_name, gauge_id, rings, gauge_lat, gauge_lon, grid_lat, grid_lon, obs_aligned, sim_aligned, fig_file)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_file = os.path.join(output_dir, "basin_streamflow_metrics_1985-1989.csv")
    try:
        metrics_df.to_csv(metrics_file, index=False, float_format="%.6f")
        print(f"Saved metrics CSV: {metrics_file}")
    except Exception as e:
        print(f"Error saving metrics CSV {metrics_file}: {e}")

    grid_match_df = pd.DataFrame(grid_match_rows)
    grid_match_file = os.path.join(output_dir, "gauge_to_grid_match.csv")
    try:
        grid_match_df.to_csv(grid_match_file, index=False)
        print(f"Saved grid match CSV: {grid_match_file}")
    except Exception as e:
        print(f"Error saving grid match CSV {grid_match_file}: {e}")

    # Save NetCDF of time series with a unified time coordinate (timestamp)
    try:
        ds_out_list = []
        for rec in metrics_records:
            gid = rec['gauge_id']
            basin_name = rec['basin']
            ts_csv = os.path.join(output_dir, f"{gid}_{basin_name.replace(' ','_')}_timeseries_1985-1989.csv")
            if not os.path.exists(ts_csv):
                continue
            df = pd.read_csv(ts_csv, parse_dates=['date']).set_index('date')
            if df.empty:
                continue
            da_sim = xr.DataArray(
                data=df['sim_m3s'].values,
                coords={'time': df.index.values},
                dims=['time'],
                name=f"sim_{gid}"
            )
            da_obs = xr.DataArray(
                data=df['obs_m3s'].values,
                coords={'time': df.index.values},
                dims=['time'],
                name=f"obs_{gid}"
            )
            ds_tmp = xr.Dataset({f"sim_{gid}": da_sim, f"obs_{gid}": da_obs})
            ds_tmp.attrs['basin_name'] = basin_name
            ds_tmp.attrs['gauge_id'] = gid
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