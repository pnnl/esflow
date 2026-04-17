import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def haversine_km(lon1, lat1, lon2, lat2):
    # all args in degrees, returns distance in km
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
    # target_convention: 'negpos' for [-180,180], 'zero360' for [0,360)
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
    # Expect 'lat' and 'lon' in dataset
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
    # Flattened arrays for nearest search
    if info['is_2d']:
        lat_vals = lat.values
        lon_vals = lon.values
        info['shape'] = lat_vals.shape
        info['flat_lat'] = lat_vals.ravel()
        info['flat_lon'] = lon_vals.ravel()
    else:
        lat_vals = lat.values
        lon_vals = lon.values
        # Build 2D mesh for distance computation
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        info['shape'] = lat2d.shape
        info['flat_lat'] = lat2d.ravel()
        info['flat_lon'] = lon2d.ravel()
    return info

def nearest_grid_index(lat_pt, lon_pt, grid_info):
    # Adjust gauge longitude to grid convention
    lon_pt_adj = lon_pt
    if grid_info['lon_convention'] == 'zero360':
        lon_pt_adj = normalize_lon(lon_pt, 'zero360')
    else:
        lon_pt_adj = normalize_lon(lon_pt, 'negpos')
    # Compute distances to all grid points
    dists = haversine_km(lon_pt_adj, lat_pt, grid_info['flat_lon'], grid_info['flat_lat'])
    k = int(np.nanargmin(dists))
    if grid_info['is_2d']:
        iy, ix = np.unravel_index(k, grid_info['shape'])
        dimy, dimx = grid_info['lat_dims']
        return {dimy: iy, dimx: ix}, (grid_info['flat_lat'][k], grid_info['flat_lon'][k])
    else:
        iy, ix = np.unravel_index(k, grid_info['shape'])
        # lat dims: (lat,), lon dims: (lon,)
        dimy = grid_info['lat_dims'][0]
        dimx = grid_info['lon_dims'][0]
        return {dimy: iy, dimx: ix}, (grid_info['flat_lat'][k], grid_info['flat_lon'][k])

def extract_sim_series(ds, varname, idx_dict, t0, t1):
    var = ds[varname]
    # Construct selection dict with time slice
    try:
        da = var.isel(idx_dict).sel(time=slice(t0, t1))
    except Exception as e:
        # Try to reconcile dim names if needed
        raise e
    # Return as pandas Series with datetime index
    try:
        series = da.to_series()
    except Exception:
        series = da.to_pandas()
    series = series.dropna()
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
    s = df['discharge_m3s'].astype(float).dropna()
    return s

def compute_fdc_quantiles(values, p_exceed):
    # values: 1D array-like, positive flows can include zeros
    vals = np.asarray(values).astype(float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.full_like(p_exceed, np.nan, dtype=float)
    # For exceedance probability p, flow is quantile at (1 - p)
    q = np.quantile(vals, 1.0 - p_exceed, interpolation='linear' if 'interpolation' in xr.__version__ else 'linear')
    return q

def main():
    # Configuration
    case_name = "sample.v3.LR.historical"
    mosart_dir = "./data/sample/e3sm/rof"
    obs_dir = "./data/sample/obs/streamflow"
    gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
    varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    years = list(range(1985, 1990))
    t0 = "1985-01-01"
    t1 = "1989-12-31"
    # Output directory (as specified)
    out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_04_streamflow_fdc/run4_output"
    ensure_dir(out_dir)

    # Load gauge metadata
    try:
        gmeta = pd.read_csv(gauge_meta_path)
    except Exception as e:
        print(f"Failed to read gauge metadata: {e}")
        return
    # Standardize types
    if 'gauge_id' not in gmeta.columns or 'lat' not in gmeta.columns or 'lon' not in gmeta.columns:
        print("Gauge metadata missing required columns (gauge_id, lat, lon).")
        return
    gmeta['gauge_id'] = gmeta['gauge_id'].astype(str)
    if 'river_name' not in gmeta.columns:
        gmeta['river_name'] = ""

    # Find MOSART daily files
    file_list = []
    for y in years:
        pattern = os.path.join(mosart_dir, f"{case_name}.mosart.h1.{y}-*.nc")
        files = sorted(glob.glob(pattern))
        file_list.extend(files)
    if len(file_list) == 0:
        print("No MOSART daily files found for 1985-1989. Please check data path.")
        return

    # Open one file to get grid info
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

    # Open all daily files as mf dataset
    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords', parallel=False)
    except Exception as e:
        print(f"Failed to open MOSART multi-file dataset: {e}")
        return

    if varname not in ds.variables:
        print(f"Variable {varname} not found in MOSART dataset.")
        return

    # Precompute nearest indices for all gauges
    gauge_indices = {}
    gauge_grid_locs = {}
    for _, row in gmeta.iterrows():
        gid = str(row['gauge_id'])
        glat = float(row['lat'])
        glon = float(row['lon'])
        try:
            idx_dict, (glib_lat, glib_lon) = nearest_grid_index(glat, glon, grid_info)
            gauge_indices[gid] = idx_dict
            # Convert grid lon to -180..180 for output
            glon_plot = normalize_lon(glib_lon, 'negpos')
            gauge_grid_locs[gid] = {'grid_lat': float(glib_lat), 'grid_lon': float(glon_plot)}
        except Exception as e:
            print(f"Failed to match gauge {gid} to grid: {e}")

    # Prepare outputs
    metrics_records = []
    fdc_records = []

    # Exceedance probability vector for FDC CSV
    p_exceed_vec = np.linspace(0.01, 0.99, 99)

    # Loop over gauges
    for _, row in gmeta.iterrows():
        gid = str(row['gauge_id'])
        river_name = row.get('river_name', "")
        lat = float(row['lat'])
        lon = float(row['lon'])
        area_km2 = row.get('area_km2', np.nan)

        # Observation series
        obs_series = load_observed_series(obs_dir, gid, t0, t1)
        if obs_series is None or obs_series.empty:
            print(f"No observation data for gauge {gid}, skipping.")
            continue

        # Simulation series
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

        # Align time series on overlapping dates
        df_pair = pd.DataFrame({
            'sim': sim_series,
            'obs': obs_series
        }).dropna()
        if df_pair.empty:
            print(f"No overlapping dates for gauge {gid}, skipping.")
            continue

        sim_vals = df_pair['sim'].values.astype(float)
        obs_vals = df_pair['obs'].values.astype(float)
        n_days = df_pair.shape[0]

        # Metrics
        sum_obs = np.sum(obs_vals)
        sum_sim = np.sum(sim_vals)
        if sum_obs == 0:
            volume_bias_ratio = np.nan
        else:
            volume_bias_ratio = (sum_sim - sum_obs) / sum_obs

        try:
            wd = wasserstein_distance(obs_vals, sim_vals)
        except Exception as e:
            print(f"Failed to compute Wasserstein distance for gauge {gid}: {e}")
            wd = np.nan

        # FDC quantiles for key exceedances
        key_p = np.array([0.10, 0.50, 0.90])
        sim_q = compute_fdc_quantiles(sim_vals, key_p)
        obs_q = compute_fdc_quantiles(obs_vals, key_p)
        q10_sim, q50_sim, q90_sim = sim_q
        q10_obs, q50_obs, q90_obs = obs_q
        ratio_q10 = (q10_sim / q10_obs) if q10_obs not in [0, None] else np.nan
        ratio_q50 = (q50_sim / q50_obs) if q50_obs not in [0, None] else np.nan
        ratio_q90 = (q90_sim / q90_obs) if q90_obs not in [0, None] else np.nan

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

        # FDC percentiles for CSV
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

    # Save CSV outputs
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

    # Plot map of Wasserstein distance and FDC panels for six gauges
    highlight_gauges = {
        "3629000": "Amazon",
        "4121801": "Missouri",
        "4115200": "Columbia",
        "6742900": "Danube",
        "2969100": "Mekong",
        "1159100": "Orange"
    }

    try:
        # Prepare map data
        if metrics_df.empty:
            print("No metrics to plot.")
            return
        # Map figure with gridspec: row0 map, rows1-2 are 2x3 panels
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.1, 1.0, 1.0])

        # Map subplot spanning top row
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='white')
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax_map.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.5)
        # Scatter all gauges colored by Wasserstein distance
        lons = normalize_lon(metrics_df['lon'].values.astype(float), 'negpos')
        lats = metrics_df['lat'].values.astype(float)
        wd_vals = metrics_df['wasserstein_m3s'].values.astype(float)
        sc = ax_map.scatter(lons, lats, c=wd_vals, s=30, cmap='viridis', transform=ccrs.PlateCarree(), edgecolor='k', linewidth=0.2)
        cb = fig.colorbar(sc, ax=ax_map, orientation='horizontal', fraction=0.05, pad=0.05)
        cb.set_label('Wasserstein distance (m3/s)')
        ax_map.set_title('Wasserstein Distance between Simulated and Observed Daily Discharge (1985-1989)')

        # Highlight specific gauges on the map
        for gid, name in highlight_gauges.items():
            row = metrics_df[metrics_df['gauge_id'] == gid]
            if row.empty:
                continue
            ax_map.scatter(normalize_lon(row['lon'].values[0], 'negpos'), row['lat'].values[0],
                           s=80, facecolors='none', edgecolors='red', linewidths=1.5,
                           transform=ccrs.PlateCarree())
            ax_map.text(normalize_lon(row['lon'].values[0], 'negpos'), row['lat'].values[0],
                        f"{name}", transform=ccrs.PlateCarree(), fontsize=7, color='red')

        # FDC panels for six gauges
        # Prepare combined FDC data for quick access
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
            # Avoid log scale issues if zeros present
            ymin_candidates = np.concatenate([df_g['obs_flow_m3s'].values, df_g['sim_flow_m3s'].values])
            ymin = np.nanmin(ymin_candidates[ymin_candidates > 0]) if np.any(ymin_candidates > 0) else 1e-3
            ymax = np.nanmax(ymin_candidates) if np.nanmax(ymin_candidates) > 0 else 1.0
            if np.isfinite(ymin) and np.isfinite(ymax):
                ax.set_ylim(ymin, ymax * 1.1)
            ax.grid(True, which='both', ls=':', lw=0.5)
            mrow = metrics_df[metrics_df['gauge_id'] == gid]
            if not mrow.empty:
                wd_val = mrow['wasserstein_m3s'].values[0]
                vb = mrow['volume_bias_ratio'].values[0]
                title = f"{basin_name} ({gid})\nWD={wd_val:.1f} m3/s, VolBias={vb*100:.1f}%"
            else:
                title = f"{basin_name} ({gid})"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Exceedance probability')
            ax.set_ylabel('Discharge (m3/s)')
            ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "streamflow_fdc_map_and_panels_1985-1989.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure to {fig_path}")
    except Exception as e:
        print(f"Failed to create figure: {e}")

if __name__ == "__main__":
    main()