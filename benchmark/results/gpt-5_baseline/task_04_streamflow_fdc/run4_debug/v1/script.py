import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def normalize_lon(lon_vals, target_convention):
    lon_vals = np.asarray(lon_vals).astype(float)
    if target_convention == 'zero360':
        lon_vals = np.mod(lon_vals, 360.0)
    else:
        lon_vals = ((lon_vals + 180) % 360) - 180
    return lon_vals


def infer_grid_lon_convention(lon_array):
    lon = np.asarray(lon_array)
    if np.nanmin(lon) >= 0 and np.nanmax(lon) <= 360:
        return 'zero360'
    return 'negpos'


def get_grid_info(ds):
    if 'lat' not in ds or 'lon' not in ds:
        raise ValueError("Dataset does not contain 'lat' and 'lon' coordinates.")
    lat = ds['lat']
    lon = ds['lon']
    info = {}
    info['lat'] = lat
    info['lon'] = lon
    info['lat_dims'] = lat.dims
    info['lon_dims'] = lon.dims
    info['is_2d'] = (lat.ndim == 2 and lon.ndim == 2)
    info['lon_convention'] = infer_grid_lon_convention(lon.values)
    if info['is_2d']:
        lat_vals = lat.values
        lon_vals = lon.values
        info['shape'] = lat_vals.shape
        info['flat_lat'] = lat_vals.ravel()
        info['flat_lon'] = lon_vals.ravel()
    else:
        lat_vals = lat.values
        lon_vals = lon.values
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        info['shape'] = lat2d.shape
        info['flat_lat'] = lat2d.ravel()
        info['flat_lon'] = lon2d.ravel()
    return info


def nearest_grid_index(lat_pt, lon_pt, grid_info):
    lon_pt_adj = lon_pt
    if grid_info['lon_convention'] == 'zero360':
        lon_pt_adj = normalize_lon(lon_pt, 'zero360')
    else:
        lon_pt_adj = normalize_lon(lon_pt, 'negpos')
    dists = haversine_km(lon_pt_adj, lat_pt, grid_info['flat_lon'], grid_info['flat_lat'])
    k = int(np.nanargmin(dists))
    if grid_info['is_2d']:
        iy, ix = np.unravel_index(k, grid_info['shape'])
        dimy, dimx = grid_info['lat_dims']
        return {dimy: iy, dimx: ix}, (grid_info['flat_lat'][k], grid_info['flat_lon'][k])
    else:
        iy, ix = np.unravel_index(k, grid_info['shape'])
        dimy = grid_info['lat_dims'][0]
        dimx = grid_info['lon_dims'][0]
        return {dimy: iy, dimx: ix}, (grid_info['flat_lat'][k], grid_info['flat_lon'][k])


def to_pandas_datetime_index(time_vals):
    # Convert various time coordinate representations to pandas.DatetimeIndex
    try:
        # If it's already a pandas-like datetime index
        if isinstance(time_vals, pd.DatetimeIndex):
            return time_vals
    except Exception:
        pass
    try:
        arr = np.asarray(time_vals)
        if np.issubdtype(arr.dtype, np.datetime64):
            return pd.to_datetime(arr)
    except Exception:
        pass
    # xarray CFTimeIndex may have .to_datetimeindex()
    try:
        if hasattr(time_vals, "to_datetimeindex"):
            return time_vals.to_datetimeindex()
    except Exception:
        pass
    # Fall back to element-wise conversion
    converted = []
    for t in list(time_vals):
        try:
            # numpy datetime64
            if isinstance(t, (np.datetime64,)):
                converted.append(pd.Timestamp(t))
                continue
            # cftime object with year/month/day
            y = getattr(t, 'year', None)
            m = getattr(t, 'month', None)
            d = getattr(t, 'day', None)
            hh = getattr(t, 'hour', 0)
            mm = getattr(t, 'minute', 0)
            ss = int(getattr(t, 'second', 0))
            if y is not None and m is not None and d is not None:
                converted.append(pd.Timestamp(y, m, d, hh, mm, ss))
            else:
                converted.append(pd.to_datetime(str(t)))
        except Exception:
            converted.append(pd.to_datetime(str(t)))
    return pd.DatetimeIndex(converted)


def extract_sim_series(ds, varname, idx_dict, t0, t1):
    var = ds[varname]
    da = var.isel(idx_dict).sel(time=slice(t0, t1))
    # Build pandas Series with pandas datetime index to avoid CFTimeIndex issues
    time_vals = da['time'].values
    dt_index = to_pandas_datetime_index(time_vals)
    series = pd.Series(da.values, index=dt_index).sort_index()
    series = series[(series.index >= pd.to_datetime(t0)) & (series.index <= pd.to_datetime(t1))]
    series = pd.to_numeric(series, errors='coerce').dropna()
    return series


def load_observed_series(obs_dir, gauge_id, t0, t1):
    csv_path = os.path.join(obs_dir, f"{gauge_id}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading obs file {csv_path}: {e}")
        return None
    if 'date' not in df.columns or 'discharge_m3s' not in df.columns:
        print(f"Obs file {csv_path} missing required columns.")
        return None
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Error parsing dates in {csv_path}: {e}")
        return None
    df = df.set_index('date').sort_index()
    df = df[(df.index >= pd.to_datetime(t0)) & (df.index <= pd.to_datetime(t1))]
    s = pd.to_numeric(df['discharge_m3s'], errors='coerce').dropna()
    return s


def quantile_linear(arr, q):
    try:
        return np.quantile(arr, q, method='linear')
    except TypeError:
        return np.quantile(arr, q, interpolation='linear')


def compute_fdc_quantiles(values, p_exceed):
    vals = np.asarray(values).astype(float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.full_like(p_exceed, np.nan, dtype=float)
    q = quantile_linear(vals, 1.0 - p_exceed)
    return q


def main():
    case_name = "sample.v3.LR.historical"
    mosart_dir = "./data/sample/e3sm/rof"
    obs_dir = "./data/sample/obs/streamflow"
    gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
    varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    years = list(range(1985, 1990))
    t0 = "1985-01-01"
    t1 = "1989-12-31"
    out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_04_streamflow_fdc/run4_debug/v1/output"
    ensure_dir(out_dir)

    try:
        gmeta = pd.read_csv(gauge_meta_path)
    except Exception as e:
        print(f"Failed to read gauge metadata: {e}")
        return
    if 'gauge_id' not in gmeta.columns or 'lat' not in gmeta.columns or 'lon' not in gmeta.columns:
        print("Gauge metadata missing required columns (gauge_id, lat, lon).")
        return
    gmeta['gauge_id'] = gmeta['gauge_id'].astype(str)
    if 'river_name' not in gmeta.columns:
        gmeta['river_name'] = ""

    file_list = []
    for y in years:
        pattern = os.path.join(mosart_dir, f"{case_name}.mosart.h1.{y}-*.nc")
        files = sorted(glob.glob(pattern))
        file_list.extend(files)
    if len(file_list) == 0:
        print("No MOSART daily files found for 1985-1989. Please check data path.")
        return

    try:
        ds_sample = xr.open_dataset(file_list[0])
    except Exception as e:
        print(f"Failed to open sample MOSART file: {e}")
        return

    try:
        grid_info = get_grid_info(ds_sample)
    except Exception as e:
        print(f"Error obtaining grid info: {e}")
        return

    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords', parallel=False)
    except Exception as e:
        print(f"Failed to open MOSART multi-file dataset: {e}")
        return

    if varname not in ds.variables:
        print(f"Variable {varname} not found in MOSART dataset.")
        return

    gauge_indices = {}
    gauge_grid_locs = {}
    for _, row in gmeta.iterrows():
        gid = str(row['gauge_id'])
        glat = float(row['lat'])
        glon = float(row['lon'])
        try:
            idx_dict, (glib_lat, glib_lon) = nearest_grid_index(glat, glon, grid_info)
            gauge_indices[gid] = idx_dict
            glon_plot = normalize_lon(glib_lon, 'negpos')
            gauge_grid_locs[gid] = {'grid_lat': float(glib_lat), 'grid_lon': float(glon_plot)}
        except Exception as e:
            print(f"Failed to match gauge {gid} to grid: {e}")

    metrics_records = []
    fdc_records = []
    p_exceed_vec = np.linspace(0.01, 0.99, 99)

    for _, row in gmeta.iterrows():
        gid = str(row['gauge_id'])
        river_name = row.get('river_name', "")
        lat = float(row['lat'])
        lon = float(row['lon'])
        area_km2 = row.get('area_km2', np.nan)

        obs_series = load_observed_series(obs_dir, gid, t0, t1)
        if obs_series is None or obs_series.empty:
            print(f"No observation data for gauge {gid}, skipping.")
            continue

        if gid not in gauge_indices:
            print(f"No grid mapping for gauge {gid}, skipping.")
            continue
        idx_dict = gauge_indices[gid]
        try:
            sim_series = extract_sim_series(ds, varname, idx_dict, t0, t1)
        except Exception as e:
            print(f"Failed to extract simulation for gauge {gid}: {e}")
            continue

        if sim_series is None or sim_series.empty:
            print(f"No simulation data for gauge {gid}, skipping.")
            continue

        try:
            df_pair = pd.concat(
                [sim_series.rename('sim'), obs_series.rename('obs')],
                axis=1, join='inner'
            ).dropna()
        except Exception as e:
            print(f"Failed to align series for gauge {gid}: {e}")
            continue

        if df_pair.empty:
            print(f"No overlapping dates for gauge {gid}, skipping.")
            continue

        sim_vals = df_pair['sim'].values.astype(float)
        obs_vals = df_pair['obs'].values.astype(float)
        n_days = df_pair.shape[0]

        sum_obs = np.sum(obs_vals)
        sum_sim = np.sum(sim_vals)
        volume_bias_ratio = np.nan if sum_obs == 0 else (sum_sim - sum_obs) / sum_obs

        try:
            wd = wasserstein_distance(obs_vals, sim_vals)
        except Exception as e:
            print(f"Failed to compute Wasserstein distance for gauge {gid}: {e}")
            wd = np.nan

        key_p = np.array([0.10, 0.50, 0.90])
        sim_q = compute_fdc_quantiles(sim_vals, key_p)
        obs_q = compute_fdc_quantiles(obs_vals, key_p)
        q10_sim, q50_sim, q90_sim = sim_q
        q10_obs, q50_obs, q90_obs = obs_q
        ratio_q10 = (q10_sim / q10_obs) if (q10_obs is not None and q10_obs != 0) else np.nan
        ratio_q50 = (q50_sim / q50_obs) if (q50_obs is not None and q50_obs != 0) else np.nan
        ratio_q90 = (q90_sim / q90_obs) if (q90_obs is not None and q90_obs != 0) else np.nan

        metrics_records.append({
            'gauge_id': gid,
            'river_name': river_name,
            'lat': lat,
            'lon': lon,
            'area_km2': area_km2,
            'n_days_overlap': n_days,
            'volume_bias_ratio': volume_bias_ratio,
            'wasserstein_m3s': wd,
            'sim_Q10_m3s': q10_sim,
            'obs_Q10_m3s': q10_obs,
            'ratio_Q10': ratio_q10,
            'sim_Q50_m3s': q50_sim,
            'obs_Q50_m3s': q50_obs,
            'ratio_Q50': ratio_q50,
            'sim_Q90_m3s': q90_sim,
            'obs_Q90_m3s': q90_obs,
            'ratio_Q90': ratio_q90
        })

        sim_fdc = compute_fdc_quantiles(sim_vals, p_exceed_vec)
        obs_fdc = compute_fdc_quantiles(obs_vals, p_exceed_vec)
        for p, sq, oq in zip(p_exceed_vec, sim_fdc, obs_fdc):
            fdc_records.append({
                'gauge_id': gid,
                'river_name': river_name,
                'lat': lat,
                'lon': lon,
                'p_exceed': p,
                'sim_flow_m3s': sq,
                'obs_flow_m3s': oq
            })

    metrics_df = pd.DataFrame(metrics_records)
    fdc_df = pd.DataFrame(fdc_records)

    try:
        metrics_csv = os.path.join(out_dir, "fdc_metrics_1985-1989.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics to {metrics_csv}")
    except Exception as e:
        print(f"Failed to save metrics CSV: {e}")

    try:
        fdc_csv = os.path.join(out_dir, "fdc_percentiles_1985-1989.csv")
        fdc_df.to_csv(fdc_csv, index=False)
        print(f"Saved FDC percentiles to {fdc_csv}")
    except Exception as e:
        print(f"Failed to save FDC CSV: {e}")

    highlight_gauges = {
        "3629000": "Amazon",
        "4121801": "Missouri",
        "4115200": "Columbia",
        "6742900": "Danube",
        "2969100": "Mekong",
        "1159100": "Orange"
        }

    try:
        if metrics_df.empty:
            print("No metrics to plot.")
            return
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.1, 1.0, 1.0])

        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='white')
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax_map.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.5)
        lons = normalize_lon(metrics_df['lon'].values.astype(float), 'negpos')
        lats = metrics_df['lat'].values.astype(float)
        wd_vals = metrics_df['wasserstein_m3s'].values.astype(float)
        sc = ax_map.scatter(lons, lats, c=wd_vals, s=30, cmap='viridis', transform=ccrs.PlateCarree(), edgecolor='k', linewidth=0.2)
        cb = fig.colorbar(sc, ax=ax_map, orientation='horizontal', fraction=0.05, pad=0.05)
        cb.set_label('Wasserstein distance (m3/s)')
        ax_map.set_title('Wasserstein Distance between Simulated and Observed Daily Discharge (1985-1989)')

        for gid, name in highlight_gauges.items():
            row = metrics_df[metrics_df['gauge_id'] == gid]
            if row.empty:
                continue
            ax_map.scatter(normalize_lon(row['lon'].values[0], 'negpos'), row['lat'].values[0],
                           s=80, facecolors='none', edgecolors='red', linewidths=1.5,
                           transform=ccrs.PlateCarree())
            ax_map.text(normalize_lon(row['lon'].values[0], 'negpos'), row['lat'].values[0],
                        f"{name}", transform=ccrs.PlateCarree(), fontsize=7, color='red')

        fdc_grouped = {gid: df for gid, df in fdc_df.groupby('gauge_id')}
        panel_positions = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
        for (gid, basin_name), (r, c) in zip(highlight_gauges.items(), panel_positions):
            ax = fig.add_subplot(gs[r, c])
            if gid not in fdc_grouped:
                ax.text(0.5, 0.5, f"No data for gauge {gid}", ha='center', va='center')
                ax.set_axis_off()
                continue
            df_g = fdc_grouped[gid].sort_values('p_exceed')
            ax.plot(df_g['p_exceed'], df_g['obs_flow_m3s'], label='Obs', color='black', lw=1.5)
            ax.plot(df_g['p_exceed'], df_g['sim_flow_m3s'], label='Sim', color='tab:blue', lw=1.5)
            ax.set_yscale('log')
            ax.set_xlim(0, 1)
            ymin_candidates = np.concatenate([df_g['obs_flow_m3s'].values, df_g['sim_flow_m3s'].values])
            pos = ymin_candidates[ymin_candidates > 0]
            ymin = np.nanmin(pos) if pos.size > 0 else 1e-3
            ymax = np.nanmax(ymin_candidates) if np.nanmax(ymin_candidates) > 0 else 1.0
            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > 0:
                ax.set_ylim(ymin, ymax * 1.1)
            ax.grid(True, which='both', ls=':', lw=0.5)
            mrow = metrics_df[metrics_df['gauge_id'] == gid]
            if not mrow.empty:
                wd_val = mrow['wasserstein_m3s'].values[0]
                vb = mrow['volume_bias_ratio'].values[0]
                try:
                    title = f"{basin_name} ({gid})\nWD={wd_val:.1f} m3/s, VolBias={vb*100:.1f}%"
                except Exception:
                    title = f"{basin_name} ({gid})"
            else:
                title = f"{basin_name} ({gid})"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Exceedance probability')
            ax.set_ylabel('Discharge (m3/s)')
            ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "streamflow_fdc_map_and_panels_1985-1989.png")
        try:
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"Saved figure to {fig_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
    except Exception as e:
        print(f"Failed to create figure: {e}")


if __name__ == "__main__":
    main()