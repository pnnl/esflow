import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def lon_to_180(lon):
    return ((lon + 180) % 360) - 180

def latlon_to_xyz(lat, lon):
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.vstack((x, y, z)).T

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def build_kdtree_from_grid(lat_da, lon_da):
    # Handle 1D lat/lon (rectilinear) or 2D lat/lon grids
    lat_vals = np.array(lat_da)
    lon_vals = np.array(lon_da)

    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        # Create 2D meshgrid (nlat, nlon)
        lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2 and lat_vals.shape == lon_vals.shape:
        lat2d, lon2d = lat_vals, lon_vals
    else:
        # Try to broadcast if possible
        try:
            lat2d, lon2d = np.broadcast_arrays(lat_vals, lon_vals)
        except Exception:
            raise ValueError(f"Unsupported lat/lon shapes: lat {lat_vals.shape}, lon {lon_vals.shape}")

    lon2d = lon_to_180(lon2d)
    flat_lat = lat2d.ravel()
    flat_lon = lon2d.ravel()
    xyz = latlon_to_xyz(flat_lat, flat_lon)
    tree = cKDTree(xyz)
    return tree, lat2d.shape, flat_lat, flat_lon

def nearest_grid_index(tree, flat_lat, flat_lon, shape, glat, glon):
    glon = lon_to_180(glon)
    xyz = latlon_to_xyz(np.array([glat]), np.array([glon]))
    dist, idx = tree.query(xyz)
    flat_idx = int(idx[0])
    i, j = np.unravel_index(flat_idx, shape)
    nlat = float(flat_lat[flat_idx])
    nlon = float(flat_lon[flat_idx])
    d_km = float(haversine_km(glat, glon, nlat, nlon))
    return i, j, nlat, nlon, d_km, flat_idx

def compute_fdc(flow_values, exceedance_perc):
    arr = np.array(flow_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full_like(exceedance_perc, np.nan, dtype=float)
    q = 1.0 - (exceedance_perc / 100.0)
    q = np.clip(q, 0.0, 1.0)
    perc = q * 100.0
    vals = np.nanpercentile(arr, perc)
    return vals

def compute_quantiles(flow_values, probs):
    arr = np.array(flow_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [np.nan for _ in probs]
    percs = [p * 100 for p in probs]
    return list(np.nanpercentile(arr, percs))

def load_observation_series(obs_path, start_date, end_date):
    try:
        df = pd.read_csv(obs_path)
        if 'date' not in df.columns or 'discharge_m3s' not in df.columns:
            print(f"Warning: Observation file {obs_path} missing required columns.")
            return pd.Series(dtype=float)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        s = pd.to_numeric(df['discharge_m3s'], errors='coerce')
        s = s.dropna()
        s.name = 'obs'
        return s
    except Exception as e:
        print(f"Error reading observation file {obs_path}: {e}")
        return pd.Series(dtype=float)

def extract_model_timeseries(ds, varname, idx_i, idx_j=None):
    try:
        var = ds[varname]
        if 'time' not in var.dims:
            print(f"Model variable {varname} missing time dimension.")
            return pd.Series(dtype=float)
        var_dims = list(var.dims)
        non_time_dims = [d for d in var_dims if d != 'time']
        if idx_j is None:
            if len(non_time_dims) != 1:
                print(f"Unexpected spatial dims for {varname}: {non_time_dims}")
                return pd.Series(dtype=float)
            ts_da = var.isel({non_time_dims[0]: idx_i})
        else:
            if len(non_time_dims) != 2:
                print(f"Unexpected spatial dims for {varname}: {non_time_dims}")
                return pd.Series(dtype=float)
            ts_da = var.isel({non_time_dims[0]: idx_i, non_time_dims[1]: idx_j})
        # Convert to pandas Series with time index
        ts = ts_da.to_series()
        ts = ts.dropna()
        ts.name = 'mod'
        return ts
    except Exception as e:
        print(f"Error extracting model timeseries: {e}")
        return pd.Series(dtype=float)

def convert_time_to_pandas(ds):
    try:
        time_index = ds.indexes.get('time', None)
    except Exception:
        time_index = None
    if time_index is None:
        return ds
    # Try to convert CFTimeIndex to pandas DatetimeIndex
    try:
        from xarray.coding.cftimeindex import CFTimeIndex
        if isinstance(time_index, CFTimeIndex):
            pd_index = time_index.to_datetimeindex()
            ds = ds.assign_coords(time=pd_index)
            return ds
    except Exception:
        pass
    # Fallback: attempt string formatting
    try:
        times = ds['time'].values
        times_str = [str(t) for t in times]
        pd_index = pd.to_datetime(times_str, errors='coerce')
        ds = ds.assign_coords(time=pd_index)
        return ds
    except Exception:
        return ds

def main():
    # Configuration
    case_name = "sample.v3.LR.historical"
    data_base = "./data/sample"
    rof_dir = os.path.join(data_base, "e3sm", "rof")
    obs_metadata_path = os.path.join(data_base, "obs", "gauge_metadata.csv")
    obs_streamflow_dir = os.path.join(data_base, "obs", "streamflow")
    varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    start_date = pd.Timestamp("1985-01-01")
    end_date = pd.Timestamp("1989-12-31")

    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_04_streamflow_fdc/run2_debug/v2/output"
    ensure_dir(output_dir)

    # Open a monthly file to get grid coordinates
    monthly_grid_file = os.path.join(rof_dir, f"{case_name}.mosart.h0.1985-01.nc")
    try:
        with xr.open_dataset(monthly_grid_file) as ds_grid:
            if 'lat' not in ds_grid.variables or 'lon' not in ds_grid.variables:
                print(f"lat/lon not found in grid file {monthly_grid_file}")
                return
            lat_da = ds_grid['lat']
            lon_da = ds_grid['lon']
            tree, grid_shape, flat_lat, flat_lon = build_kdtree_from_grid(lat_da, lon_da)
    except Exception as e:
        print(f"Error opening grid file {monthly_grid_file}: {e}")
        return

    # Gather daily model files
    daily_files = []
    for year in range(1985, 1990):
        pattern = os.path.join(rof_dir, f"{case_name}.mosart.h1.{year}-*.nc")
        files_year = sorted(glob.glob(pattern))
        daily_files.extend(files_year)
    if len(daily_files) == 0:
        print("No daily MOSART files found for 1985-1989.")
        return

    try:
        ds_daily = xr.open_mfdataset(daily_files, combine='by_coords', parallel=False)
        if varname not in ds_daily.variables:
            print(f"Variable {varname} not found in daily dataset.")
            try:
                ds_daily.close()
            except Exception:
                pass
            return
        # Convert cftime to pandas datetime for safe slicing
        ds_daily = convert_time_to_pandas(ds_daily)
        # Subset time if possible
        if 'time' in ds_daily.coords and np.issubdtype(ds_daily['time'].dtype, np.datetime64):
            ds_daily = ds_daily.sel(time=slice(start_date, end_date))
    except Exception as e:
        print(f"Error opening daily dataset: {e}")
        return

    # Load gauge metadata
    try:
        gauges_df = pd.read_csv(obs_metadata_path, dtype={'gauge_id': str})
    except Exception as e:
        print(f"Error reading gauge metadata {obs_metadata_path}: {e}")
        try:
            ds_daily.close()
        except Exception:
            pass
        return

    # Results containers
    metrics_list = []
    fdc_records = []

    exceedance_perc = np.arange(1, 100, 1)  # 1% to 99%

    # Iterate gauges
    for idx, row in gauges_df.iterrows():
        gauge_id = str(row['gauge_id'])
        try:
            glat = float(row['lat'])
            glon = float(row['lon'])
        except Exception:
            print(f"Skipping gauge {gauge_id}: invalid lat/lon.")
            continue
        river_name = str(row.get('river_name', ''))
        obs_file = os.path.join(obs_streamflow_dir, f"{gauge_id}.csv")

        # Load observation series
        obs_ts = load_observation_series(obs_file, start_date, end_date)
        if obs_ts.empty:
            print(f"Skipping gauge {gauge_id}: no observation data in period.")
            continue

        # Nearest grid cell
        try:
            i, j, nlat, nlon, d_km, flat_idx = nearest_grid_index(tree, flat_lat, flat_lon, grid_shape, glat, glon)
        except Exception as e:
            print(f"Error finding nearest grid cell for gauge {gauge_id}: {e}")
            continue

        # Extract model timeseries
        try:
            var = ds_daily[varname]
            var_dims = list(var.dims)
            non_time_dims = [d for d in var_dims if d != 'time']
            if len(non_time_dims) == 2:
                mod_ts = extract_model_timeseries(ds_daily, varname, i, j)
            elif len(non_time_dims) == 1:
                mod_ts = extract_model_timeseries(ds_daily, varname, i)
            else:
                print(f"Unexpected dims for {varname}: {var_dims}")
                continue

            # Ensure time filtering and proper type
            if not mod_ts.index.inferred_type == 'datetime64':
                try:
                    mod_ts.index = pd.to_datetime(mod_ts.index)
                except Exception:
                    pass
            mod_ts = mod_ts[(mod_ts.index >= start_date) & (mod_ts.index <= end_date)]
        except Exception as e:
            print(f"Error extracting model series for gauge {gauge_id}: {e}")
            continue

        if mod_ts.empty:
            print(f"Skipping gauge {gauge_id}: no model data in period.")
            continue

        # Align for volume comparison
        combined = pd.concat([obs_ts.rename('obs'), mod_ts.rename('mod')], axis=1).dropna()
        n_overlap = combined.shape[0]
        n_obs = obs_ts.shape[0]
        n_mod = mod_ts.shape[0]

        if n_overlap == 0:
            print(f"Skipping gauge {gauge_id}: no overlapping days between obs and model.")
            continue

        # Volume bias using overlapping days, percent
        sum_obs = combined['obs'].sum()
        sum_mod = combined['mod'].sum()
        if sum_obs == 0 or not np.isfinite(sum_obs):
            volume_bias_pct = np.nan
        else:
            volume_bias_pct = (sum_mod / sum_obs - 1.0) * 100.0

        # Wasserstein distance on distributions (m3/s)
        try:
            wdist = float(wasserstein_distance(obs_ts.values, mod_ts.values))
        except Exception as e:
            print(f"Error computing Wasserstein distance for gauge {gauge_id}: {e}")
            wdist = np.nan

        # Quantiles for Q10, Q50, Q90 (exceedance) => percentiles 90, 50, 10
        probs = [0.90, 0.50, 0.10]
        q_obs = compute_quantiles(obs_ts.values, probs)
        q_mod = compute_quantiles(mod_ts.values, probs)

        def safe_ratio(a, b):
            if b is None:
                return np.nan
            if not (np.isfinite(a) and np.isfinite(b)):
                return np.nan
            if b == 0:
                return np.nan
            return a / b

        ratio_q10 = safe_ratio(q_mod[0], q_obs[0])
        ratio_q50 = safe_ratio(q_mod[1], q_obs[1])
        ratio_q90 = safe_ratio(q_mod[2], q_obs[2])

        # FDC values
        fdc_obs = compute_fdc(obs_ts.values, exceedance_perc)
        fdc_mod = compute_fdc(mod_ts.values, exceedance_perc)
        for p, fo, fm in zip(exceedance_perc, fdc_obs, fdc_mod):
            fdc_records.append({
                'gauge_id': gauge_id,
                'river_name': river_name,
                'exceedance_percent': p,
                'obs_flow_m3s': fo,
                'mod_flow_m3s': fm
            })

        # Store metrics
        metrics_list.append({
            'gauge_id': gauge_id,
            'river_name': river_name,
            'gauge_lat': glat,
            'gauge_lon': lon_to_180(glon),
            'grid_lat': nlat,
            'grid_lon': nlon,
            'grid_i': i,
            'grid_j': j if j is not None else -1,
            'grid_flat_index': flat_idx,
            'distance_km': d_km,
            'n_days_obs': n_obs,
            'n_days_mod': n_mod,
            'n_days_overlap': n_overlap,
            'volume_bias_pct': volume_bias_pct,
            'wasserstein_m3s': wdist,
            'Q10_exceed_obs_m3s': q_obs[0],
            'Q10_exceed_mod_m3s': q_mod[0],
            'Q10_ratio_mod_over_obs': ratio_q10,
            'Q50_exceed_obs_m3s': q_obs[1],
            'Q50_exceed_mod_m3s': q_mod[1],
            'Q50_ratio_mod_over_obs': ratio_q50,
            'Q90_exceed_obs_m3s': q_obs[2],
            'Q90_exceed_mod_m3s': q_mod[2],
            'Q90_ratio_mod_over_obs': ratio_q90
        })

    # Close dataset
    try:
        ds_daily.close()
    except Exception:
        pass

    # Save CSVs
    metrics_df = pd.DataFrame(metrics_list)
    fdc_df = pd.DataFrame(fdc_records)

    try:
        metrics_csv = os.path.join(output_dir, "per_gauge_fdc_metrics_1985_1989.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics to {metrics_csv}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

    try:
        fdc_csv = os.path.join(output_dir, "per_gauge_fdc_percentiles_1985_1989.csv")
        fdc_df.to_csv(fdc_csv, index=False)
        print(f"Saved FDC percentiles to {fdc_csv}")
    except Exception as e:
        print(f"Error saving FDC CSV: {e}")

    # Plot map with Wasserstein distance and FDC panels for six gauges
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(nrows=3, ncols=4, width_ratios=[1.2, 1.2, 1.0, 1.0], height_ratios=[1, 1, 1], wspace=0.4, hspace=0.35)
        ax_map = fig.add_subplot(gs[:, :2], projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.coastlines(linewidth=0.5)
        if not metrics_df.empty:
            sc = ax_map.scatter(metrics_df['gauge_lon'].values, metrics_df['gauge_lat'].values,
                                c=metrics_df['wasserstein_m3s'].values, s=30, cmap='viridis',
                                transform=ccrs.PlateCarree(), edgecolor='k', linewidth=0.2)
            cb = plt.colorbar(sc, ax=ax_map, orientation='vertical', fraction=0.046, pad=0.04)
            cb.set_label("Wasserstein distance (m3/s)")
            ax_map.set_title("Wasserstein distance of river discharge (1985-1989)")
        else:
            ax_map.set_title("No metrics available to plot")

        six_gauges = [
            ("Amazon", "3629000"),
            ("Missouri", "4121801"),
            ("Columbia", "4115200"),
            ("Danube", "6742900"),
            ("Mekong", "2969100"),
            ("Orange", "1159100"),
        ]
        panel_axes = [
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[1, 3]),
            fig.add_subplot(gs[2, 2]),
            fig.add_subplot(gs[2, 3]),
        ]

        for ax, (river, gid) in zip(panel_axes, six_gauges):
            sub = fdc_df[fdc_df['gauge_id'] == gid]
            if sub.empty:
                ax.text(0.5, 0.5, f"No data\n{river} ({gid})", ha='center', va='center')
                ax.set_axis_off()
                continue
            p = sub['exceedance_percent'].values
            obs = sub['obs_flow_m3s'].values
            mod = sub['mod_flow_m3s'].values

            eps = 1e-6
            obs_plot = np.where(obs <= 0, eps, obs)
            mod_plot = np.where(mod <= 0, eps, mod)

            ax.semilogy(p, obs_plot, label="Obs", color="black")
            ax.semilogy(p, mod_plot, label="E3SM", color="tab:blue")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Exceedance probability (%)")
            ax.set_ylabel("Flow (m3/s)")
            mrow = metrics_df[metrics_df['gauge_id'] == gid]
            if not mrow.empty and np.isfinite(mrow['wasserstein_m3s'].values[0]):
                wd = mrow['wasserstein_m3s'].values[0]
                ax.set_title(f"{river} ({gid})\nWD={wd:.1f} m3/s")
            else:
                ax.set_title(f"{river} ({gid})")
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
            ax.legend(fontsize=8)

        figfile = os.path.join(output_dir, "fdc_map_and_panels_1985_1989.png")
        plt.savefig(figfile, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {figfile}")
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()