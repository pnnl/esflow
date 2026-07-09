import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import wasserstein_distance

def ensure_dir(d):
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {d}: {e}")

def haversine_dist(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return c

def wrap_longitudes(lon_vals, target_is_0360=True):
    if target_is_0360:
        return np.mod(lon_vals, 360.0)
    else:
        l = (np.array(lon_vals) + 180.0) % 360.0 - 180.0
        if np.isscalar(lon_vals):
            return float(l)
        return l

def find_latlon_vars(ds):
    lat_name = None
    lon_name = None
    for cand in ['lat', 'latitude', 'LATIXY', 'LAT']:
        if cand in ds.variables:
            lat_name = cand
            break
    for cand in ['lon', 'longitude', 'LONGXY', 'LON']:
        if cand in ds.variables:
            lon_name = cand
            break
    if lat_name is None or lon_name is None:
        raise ValueError("Could not find latitude/longitude variables in dataset.")
    return lat_name, lon_name

def infer_grid_dims(lat_da, lon_da):
    if lat_da.ndim == 1 and lon_da.ndim == 1:
        return ('1d', lat_da.dims[0], lon_da.dims[0])
    elif lat_da.ndim == 2 and lon_da.ndim == 2:
        return ('2d', lat_da.dims[0], lat_da.dims[1])
    else:
        if lat_da.ndim == 2 and lon_da.ndim == 2:
            return ('2d', lat_da.dims[0], lat_da.dims[1])
        else:
            raise ValueError("Unsupported lat/lon dimensionality.")

def nearest_cell_indices(gauge_lat, gauge_lon, lat_da, lon_da, grid_type, ydim, xdim, model_lon_is_0360):
    g_lon_adj = wrap_longitudes(gauge_lon, target_is_0360=model_lon_is_0360)
    if grid_type == '1d':
        lat_vals = lat_da.values
        lon_vals = lon_da.values
        lat_idx = int(np.argmin(np.abs(lat_vals - gauge_lat)))
        diffs = np.abs(lon_vals - g_lon_adj)
        lon_idx = int(np.argmin(diffs))
        return {ydim: lat_idx, xdim: lon_idx}
    else:
        lat2d = lat_da.values
        lon2d = lon_da.values
        if model_lon_is_0360:
            lon2d_wrapped = np.mod(lon2d, 360.0)
        else:
            lon2d_wrapped = (lon2d + 180.0) % 360.0 - 180.0
        dist = haversine_dist(g_lon_adj, gauge_lat, lon2d_wrapped, lat2d)
        flat_index = np.argmin(dist)
        j_idx, i_idx = np.unravel_index(flat_index, lat2d.shape)
        return {ydim: int(j_idx), xdim: int(i_idx)}

def safe_ratio(a, b):
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b

def compute_fdc(values, exceed_probs):
    return np.percentile(values, 100.0 - np.array(exceed_probs))

def read_observation_timeseries(obs_path, start_date, end_date):
    try:
        df = pd.read_csv(obs_path)
    except Exception as e:
        print(f"Failed to read observation file {obs_path}: {e}")
        return None
    if 'date' not in df.columns or 'discharge_m3s' not in df.columns:
        print(f"Observation file {obs_path} missing required columns.")
        return None
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Failed to parse dates in observation file {obs_path}: {e}")
        return None
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    df = df.sort_values('date')
    df = df[['date', 'discharge_m3s']].rename(columns={'discharge_m3s': 'obs'})
    df['obs'] = df['obs'].clip(lower=0)
    df = df.dropna()
    return df

def times_to_pandas(time_values):
    # Convert numpy datetime64 or cftime objects to pandas datetime64[ns]
    try:
        # numpy datetime64 path
        if np.issubdtype(time_values.dtype, np.datetime64):
            return pd.to_datetime(time_values)
    except Exception:
        pass
    # Fallback: format as strings and parse
    out = []
    for t in time_values:
        try:
            s = t.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            s = str(t)
        out.append(s)
    return pd.to_datetime(out, errors='coerce')

def main():
    case_name = "sample.v3.LR.historical"
    base_e3sm_dir = "./data/sample/e3sm"
    rof_dir = os.path.join(base_e3sm_dir, "rof")
    obs_streamflow_dir = "./data/sample/obs/streamflow"
    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_04_streamflow_fdc/run1_debug/v1/output"
    ensure_dir(output_dir)

    years = list(range(1985, 1990))
    start_date = pd.Timestamp("1985-01-01")
    end_date = pd.Timestamp("1989-12-31")
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    varname_target = "RIVER_DISCHARGE_OVER_LAND_LIQ"

    try:
        gmeta = pd.read_csv(gauge_metadata_path)
    except Exception as e:
        print(f"Error reading gauge metadata: {e}")
        return
    if 'gauge_id' not in gmeta.columns or 'lat' not in gmeta.columns or 'lon' not in gmeta.columns:
        print("Gauge metadata missing required columns: gauge_id, lat, lon")
        return
    gmeta['gauge_id_str'] = gmeta['gauge_id'].astype(str)

    ds_grid = None
    try:
        h0_example = os.path.join(rof_dir, f"{case_name}.mosart.h0.1985-01.nc")
        if os.path.exists(h0_example):
            ds_grid = xr.open_dataset(h0_example)
        else:
            h0_files = glob.glob(os.path.join(rof_dir, f"{case_name}.mosart.h0.*.nc"))
            if len(h0_files) > 0:
                ds_grid = xr.open_dataset(h0_files[0])
            else:
                h1_files = glob.glob(os.path.join(rof_dir, f"{case_name}.mosart.h1.1985-01-*.nc"))
                if len(h1_files) > 0:
                    ds_grid = xr.open_dataset(h1_files[0])
    except Exception as e:
        print(f"Error opening MOSART grid file: {e}")
        ds_grid = None

    if ds_grid is None:
        print("Could not open any MOSART file to read grid. Exiting.")
        return

    try:
        lat_name, lon_name = find_latlon_vars(ds_grid)
        lat_da = ds_grid[lat_name]
        lon_da = ds_grid[lon_name]
        grid_type, ydim, xdim = infer_grid_dims(lat_da, lon_da)
        lon_vals = lon_da.values
        lon_max = float(np.nanmax(lon_vals))
        model_lon_is_0360 = lon_max > 180.0
    except Exception as e:
        print(f"Error processing grid lat/lon: {e}")
        return

    gauge_to_index = {}
    model_cell_coords = {}
    for idx, row in gmeta.iterrows():
        gid = row['gauge_id_str']
        glat = float(row['lat'])
        glon = float(row['lon'])
        try:
            sel = nearest_cell_indices(glat, glon, lat_da, lon_da, grid_type, ydim, xdim, model_lon_is_0360)
            gauge_to_index[gid] = sel
            if grid_type == '1d':
                j = sel[ydim]
                i = sel[xdim]
                lat_m = float(lat_da.values[j])
                lon_m = float(lon_da.values[i])
                if model_lon_is_0360:
                    lon_m = np.mod(lon_m, 360.0)
                else:
                    lon_m = (lon_m + 180.0) % 360.0 - 180.0
            else:
                j = sel[ydim]
                i = sel[xdim]
                lat_m = float(lat_da.values[j, i])
                lon_m = float(lon_da.values[j, i])
                if model_lon_is_0360:
                    lon_m = np.mod(lon_m, 360.0)
                else:
                    lon_m = (lon_m + 180.0) % 360.0 - 180.0
            model_cell_coords[gid] = {'lat_model': lat_m, 'lon_model': lon_m}
        except Exception as e:
            print(f"Failed to map gauge {gid} to grid: {e}")

    daily_files = []
    for y in years:
        pattern = os.path.join(rof_dir, f"{case_name}.mosart.h1.{y}-*.nc")
        files = sorted(glob.glob(pattern))
        daily_files.extend(files)
    if len(daily_files) == 0:
        print("No MOSART daily files found for requested years.")
        return
    print(f"Found {len(daily_files)} daily MOSART files.")

    try:
        ds_all = xr.open_mfdataset(daily_files, combine='by_coords', decode_times=True, parallel=False)
    except Exception as e:
        print(f"Error opening daily MOSART dataset: {e}")
        return

    var_candidates = list(ds_all.data_vars)
    if varname_target in ds_all.data_vars:
        varname = varname_target
    else:
        v_upper = [v.upper() for v in var_candidates]
        if varname_target.upper() in v_upper:
            varname = var_candidates[v_upper.index(varname_target.upper())]
        else:
            v_match = [v for v in var_candidates if ('RIVER' in v.upper() and 'DISCHARGE' in v.upper())]
            if len(v_match) > 0:
                varname = v_match[0]
            else:
                print(f"Could not find variable {varname_target} in dataset. Available: {var_candidates[:10]} ...")
                try:
                    ds_all.close()
                except Exception:
                    pass
                return

    # Subset time using string-based slice to handle cftime calendars
    try:
        ds_all = ds_all.sel(time=slice(start_str, end_str))
    except Exception as e:
        print(f"Error subsetting dataset by time (using string slicing): {e}")

    metrics_records = []
    fdc_records = []
    exceed_probs = np.arange(1, 100, 1)  # 1..99

    for idx, row in gmeta.iterrows():
        gid = row['gauge_id_str']
        river_name = row['river_name'] if 'river_name' in row and isinstance(row['river_name'], str) else ''
        glat = float(row['lat'])
        glon = float(row['lon'])
        obs_path = os.path.join(obs_streamflow_dir, f"{gid}.csv")
        if not os.path.exists(obs_path):
            print(f"Observation file not found for gauge {gid}")
            continue
        if gid not in gauge_to_index:
            print(f"No mapped model grid index for gauge {gid}")
            continue

        obs_df = read_observation_timeseries(obs_path, start_date, end_date)
        if obs_df is None or obs_df.empty:
            print(f"No valid observations for gauge {gid}")
            continue

        try:
            sel = gauge_to_index[gid]
            var_dims = ds_all[varname].dims
            if 'time' not in var_dims:
                print(f"Variable {varname} missing time dimension.")
                continue
            indexer = {}
            for d in var_dims:
                if d == 'time':
                    continue
                if d in sel:
                    indexer[d] = sel[d]
                else:
                    # attempt to match by y/x dims if possible
                    if grid_type == '1d':
                        if d == ydim and ydim in sel:
                            indexer[d] = sel[ydim]
                        elif d == xdim and xdim in sel:
                            indexer[d] = sel[xdim]
            da_point = ds_all[varname].isel(indexer).squeeze()
            # Ensure only time dimension remains
            remaining_dims = [d for d in da_point.dims if d != 'time']
            if len(remaining_dims) > 0:
                print(f"Remaining non-time dims for gauge {gid}: {remaining_dims}. Skipping.")
                continue
            times = da_point['time'].values
            dates = times_to_pandas(times)
            model_vals = da_point.values.astype(float)
            model_df = pd.DataFrame({'date': dates, 'model': model_vals})
            model_df = model_df.dropna(subset=['date'])
            model_df = model_df[(model_df['date'] >= start_date) & (model_df['date'] <= end_date)]
            model_df['model'] = model_df['model'].clip(lower=0)
        except Exception as e:
            print(f"Failed extracting model series for gauge {gid}: {e}")
            continue

        try:
            df = pd.merge(obs_df, model_df, on='date', how='inner')
            df = df.dropna()
        except Exception as e:
            print(f"Failed merging obs and model for gauge {gid}: {e}")
            continue

        if df.empty or len(df) < 10:
            print(f"Insufficient overlapping data for gauge {gid} ({len(df)} records).")
            continue

        obs_vals = df['obs'].values.astype(float)
        mod_vals = df['model'].values.astype(float)

        obs_sum = float(np.sum(obs_vals))
        mod_sum = float(np.sum(mod_vals))
        vol_ratio = safe_ratio(mod_sum, obs_sum)
        vol_bias = np.nan
        if not np.isnan(vol_ratio):
            vol_bias = vol_ratio - 1.0

        try:
            wdist = float(wasserstein_distance(obs_vals, mod_vals))
        except Exception:
            wdist = np.nan

        obs_q10 = float(np.percentile(obs_vals, 90.0))
        obs_q50 = float(np.percentile(obs_vals, 50.0))
        obs_q90 = float(np.percentile(obs_vals, 10.0))
        mod_q10 = float(np.percentile(mod_vals, 90.0))
        mod_q50 = float(np.percentile(mod_vals, 50.0))
        mod_q90 = float(np.percentile(mod_vals, 10.0))
        ratio_q10 = safe_ratio(mod_q10, obs_q10)
        ratio_q50 = safe_ratio(mod_q50, obs_q50)
        ratio_q90 = safe_ratio(mod_q90, obs_q90)

        rec = {
            'gauge_id': row['gauge_id'],
            'gauge_id_str': gid,
            'river_name': river_name if isinstance(river_name, str) else '',
            'lat': glat,
            'lon': glon,
            'model_cell_lat': model_cell_coords.get(gid, {}).get('lat_model', np.nan),
            'model_cell_lon': model_cell_coords.get(gid, {}).get('lon_model', np.nan),
            'n_common_days': len(df),
            'obs_mean_m3s': float(np.mean(obs_vals)),
            'mod_mean_m3s': float(np.mean(mod_vals)),
            'volume_bias_ratio': vol_ratio,
            'volume_bias_fraction': vol_bias,
            'wasserstein_m3s': wdist,
            'Q10_obs_m3s': obs_q10,
            'Q10_mod_m3s': mod_q10,
            'Q10_ratio': ratio_q10,
            'Q50_obs_m3s': obs_q50,
            'Q50_mod_m3s': mod_q50,
            'Q50_ratio': ratio_q50,
            'Q90_obs_m3s': obs_q90,
            'Q90_mod_m3s': mod_q90,
            'Q90_ratio': ratio_q90
        }
        metrics_records.append(rec)

        obs_fdc = compute_fdc(obs_vals, exceed_probs)
        mod_fdc = compute_fdc(mod_vals, exceed_probs)
        for p, of, mf in zip(exceed_probs, obs_fdc, mod_fdc):
            fdc_records.append({
                'gauge_id': row['gauge_id'],
                'gauge_id_str': gid,
                'p_exceed_percent': p,
                'obs_flow_m3s': float(of),
                'model_flow_m3s': float(mf)
            })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(output_dir, "streamflow_fdc_metrics_1985-1989.csv")
    try:
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics to {metrics_csv_path}")
    except Exception as e:
        print(f"Failed to save metrics CSV: {e}")

    fdc_df = pd.DataFrame(fdc_records)
    fdc_csv_path = os.path.join(output_dir, "streamflow_fdc_percentiles_1985-1989.csv")
    try:
        fdc_df.to_csv(fdc_csv_path, index=False)
        print(f"Saved FDC percentiles to {fdc_csv_path}")
    except Exception as e:
        print(f"Failed to save FDC CSV: {e}")

    # Plotting
    try:
        fig = plt.figure(figsize=(18, 9))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(nrows=2, ncols=6, width_ratios=[1,1,1,1,1,1], height_ratios=[1,1], wspace=0.4, hspace=0.35)

        ax_map = plt.subplot(gs[:, 0:3], projection=ccrs.Robinson())
        ax_map.set_global()
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='white')

        if not metrics_df.empty and 'gauge_id' in metrics_df.columns and 'wasserstein_m3s' in metrics_df.columns:
            plot_df = pd.merge(gmeta[['gauge_id', 'lat', 'lon']], metrics_df[['gauge_id', 'wasserstein_m3s']], on='gauge_id', how='inner')
            sc = ax_map.scatter(plot_df['lon'].values, plot_df['lat'].values, c=plot_df['wasserstein_m3s'].values,
                                cmap='viridis', s=30, transform=ccrs.PlateCarree(), edgecolor='k', linewidth=0.2)
            cb = plt.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.05, fraction=0.05)
            cb.set_label('Wasserstein distance (m3/s)')
        else:
            ax_map.scatter(gmeta['lon'].values, gmeta['lat'].values, color='gray', s=10, transform=ccrs.PlateCarree(), edgecolor='none')
        ax_map.set_title('E3SM vs Observed Streamflow (1985-1989): Wasserstein Distance and FDCs')

        special_gauges = [
            ('Amazon', '3629000'),
            ('Missouri', '4121801'),
            ('Columbia', '4115200'),
            ('Danube', '6742900'),
            ('Mekong', '2969100'),
            ('Orange', '1159100'),
        ]
        panel_axes = []
        for r in range(2):
            for c in range(3):
                ax = plt.subplot(gs[r, 3+c])
                panel_axes.append((ax, r, c))

        def set_logy_safe(ax, y1, y2):
            arr = np.concatenate([y1, y2]).astype(float)
            pos = arr[arr > 0]
            if pos.size > 0:
                ymin = max(pos.min() * 0.5, 1e-6)
                ymax = max(arr.max() * 1.1, ymin * 10)
            else:
                ymin = 1e-6
                ymax = 1.0
            ax.set_yscale('log')
            ax.set_ylim(ymin, ymax)

        for (ax, r, c), (river_label, gid_str) in zip(panel_axes, special_gauges):
            fdc_g = fdc_df[fdc_df['gauge_id_str'] == gid_str]
            if fdc_g.empty:
                ax.text(0.5, 0.5, f"No data\n{river_label} ({gid_str})", ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            p = fdc_g['p_exceed_percent'].values
            of = fdc_g['obs_flow_m3s'].values
            mf = fdc_g['model_flow_m3s'].values
            ax.plot(p, of, label='Obs', color='k', lw=1.5)
            ax.plot(p, mf, label='E3SM', color='tab:blue', lw=1.5)
            ax.set_xlim(0, 100)
            ax.grid(True, which='both', linestyle=':', alpha=0.4)
            set_logy_safe(ax, of, mf)
            mrow = metrics_df[metrics_df['gauge_id_str'] == gid_str]
            if not mrow.empty:
                try:
                    w = float(mrow['wasserstein_m3s'].iloc[0])
                except Exception:
                    w = np.nan
                vb = mrow['volume_bias_fraction'].iloc[0] if 'volume_bias_fraction' in mrow.columns else np.nan
                try:
                    vb_float = float(vb)
                except Exception:
                    vb_float = np.nan
                if not np.isnan(vb_float):
                    ax.set_title(f"{river_label} ({gid_str})\nWD={w:.0f} m3/s  VB={vb_float:+.1%}")
                else:
                    ax.set_title(f"{river_label} ({gid_str})\nWD={w:.0f} m3/s")
            else:
                ax.set_title(f"{river_label} ({gid_str})")
            if r == 1:
                ax.set_xlabel('Exceedance probability (%)')
            if c == 0:
                ax.set_ylabel('Discharge (m3/s)')
        if len(panel_axes) > 0:
            panel_axes[0][0].legend(loc='best', fontsize=8)

        fig_path = os.path.join(output_dir, "streamflow_fdc_map_panels.png")
        try:
            fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved figure to {fig_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
    except Exception as e:
        print(f"Failed to create/save figure: {e}")

    try:
        ds_all.close()
    except Exception:
        pass
    try:
        ds_grid.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()