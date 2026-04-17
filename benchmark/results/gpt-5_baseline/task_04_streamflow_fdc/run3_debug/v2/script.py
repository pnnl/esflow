import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

def wrap_lon_to_180(lon):
    lon = np.asarray(lon)
    lon_wrapped = ((lon + 180) % 360) - 180
    return lon_wrapped

def build_kdtree_from_grid(lat_da, lon_da):
    # Build KDTree for nearest neighbor search on lat/lon grid
    lat_vals = lat_da.values
    lon_vals = lon_da.values
    # Case 1: both are 2D and share dims
    if lat_vals.ndim == 2 and lon_vals.ndim == 2:
        lat_flat = lat_vals.ravel()
        lon_flat = wrap_lon_to_180(lon_vals.ravel())
        coords = np.column_stack((lat_flat, lon_flat))
        tree = cKDTree(coords)
        meta = {
            "mode": "2d",
            "latdim": lat_da.dims[0],
            "londim": lat_da.dims[1],
            "nlat": lat_da.sizes[lat_da.dims[0]],
            "nlon": lat_da.sizes[lat_da.dims[1]],
        }
    # Case 2: both are 1D (rectilinear), possibly with different dims
    elif lat_vals.ndim == 1 and lon_vals.ndim == 1:
        # Broadcast to 2D grid
        lat_b, lon_b = xr.broadcast(lat_da, lon_da)
        lat_flat = lat_b.values.ravel()
        lon_flat = wrap_lon_to_180(lon_b.values.ravel())
        coords = np.column_stack((lat_flat, lon_flat))
        tree = cKDTree(coords)
        meta = {
            "mode": "2d",
            "latdim": lat_b.dims[0],
            "londim": lat_b.dims[1],
            "nlat": lat_b.sizes[lat_b.dims[0]],
            "nlon": lat_b.sizes[lat_b.dims[1]],
        }
    else:
        # Fallback: broadcast to 2D
        lat_b, lon_b = xr.broadcast(lat_da, lon_da)
        lat_flat = lat_b.values.ravel()
        lon_flat = wrap_lon_to_180(lon_b.values.ravel())
        coords = np.column_stack((lat_flat, lon_flat))
        tree = cKDTree(coords)
        meta = {
            "mode": "2d",
            "latdim": lat_b.dims[0],
            "londim": lat_b.dims[1],
            "nlat": lat_b.sizes[lat_b.dims[0]],
            "nlon": lat_b.sizes[lat_b.dims[1]],
        }
    return tree, meta

def grid_index_from_query(tree_meta, idx_flat):
    if tree_meta["mode"] == "2d":
        nlon = tree_meta["nlon"]
        ilat = idx_flat // nlon
        ilon = idx_flat % nlon
        return ("2d", int(ilat), int(ilon))
    else:
        return ("1d", int(idx_flat))

def select_timeseries(ds, varname, sel_info, tree_meta):
    var = ds[varname]
    if sel_info[0] == "2d":
        ilat, ilon = int(sel_info[1]), int(sel_info[2])
        # Determine which dims in var correspond to lat/lon
        latdim_candidate = None
        londim_candidate = None
        # Common names
        if 'lat' in var.dims and 'lon' in var.dims:
            latdim_candidate = 'lat'
            londim_candidate = 'lon'
        else:
            # Use tree_meta dims if present in var
            if tree_meta["latdim"] in var.dims and tree_meta["londim"] in var.dims:
                latdim_candidate = tree_meta["latdim"]
                londim_candidate = tree_meta["londim"]
            else:
                # Try to find dims with sizes matching nlat/nlon
                cand_lat = [d for d in var.dims if var.sizes[d] == tree_meta["nlat"]]
                cand_lon = [d for d in var.dims if var.sizes[d] == tree_meta["nlon"]]
                if cand_lat and cand_lon:
                    latdim_candidate = cand_lat[0]
                    londim_candidate = cand_lon[0]
        if latdim_candidate is None or londim_candidate is None:
            # Last resort: pick first two non-time dims
            spatial_dims = [d for d in var.dims if d != 'time']
            if len(spatial_dims) >= 2:
                latdim_candidate = spatial_dims[0]
                londim_candidate = spatial_dims[1]
            else:
                raise ValueError("Could not identify spatial dims for selection.")
        # Clamp indices to avoid OOB due to minor mismatches
        ilat = max(0, min(ilat, var.sizes[latdim_candidate] - 1))
        ilon = max(0, min(ilon, var.sizes[londim_candidate] - 1))
        ts = var.isel({latdim_candidate: ilat, londim_candidate: ilon})
    else:
        idx = int(sel_info[1])
        dimname = tree_meta.get("dim", None)
        if dimname and dimname in var.dims:
            idx = max(0, min(idx, var.sizes[dimname] - 1))
            ts = var.isel({dimname: idx})
        else:
            # Try to find any single spatial dim
            spatial_dims = [d for d in var.dims if d != 'time']
            if spatial_dims:
                dim = spatial_dims[0]
                idx = max(0, min(idx, var.sizes[dim] - 1))
                ts = var.isel({dim: idx})
            else:
                raise ValueError("Could not match 1D dim for selection.")
    return ts

def compute_fdc(values, p_exceed=np.linspace(0.01, 0.99, 99)):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return p_exceed, np.full_like(p_exceed, np.nan, dtype=float)
    q_nonexceed = 1.0 - p_exceed
    try:
        q = np.quantile(values, q_nonexceed, method='linear')
    except TypeError:
        q = np.quantile(values, q_nonexceed)
    return p_exceed, q

def main():
    # Paths and constants
    case_name = "sample.v3.LR.historical"
    base_dir = "./data/sample/e3sm"
    rof_dir = os.path.join(base_dir, "rof")
    obs_streamflow_dir = "./data/sample/obs/streamflow"
    gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_04_streamflow_fdc/run3_debug/v2/output"
    os.makedirs(output_dir, exist_ok=True)

    var_mosart = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    start_date = "1985-01-01"
    end_date = "1989-12-31"

    # Load gauge metadata
    try:
        gauge_meta = pd.read_csv(gauge_meta_path)
    except Exception as e:
        print(f"Error reading gauge metadata: {e}")
        return

    if 'gauge_id' not in gauge_meta.columns or 'lat' not in gauge_meta.columns or 'lon' not in gauge_meta.columns:
        print("Gauge metadata missing required columns: gauge_id, lat, lon")
        return

    # Read a MOSART file to get grid lat/lon
    lat_da, lon_da = None, None
    grid_ds = None
    try:
        monthly_files = sorted(glob.glob(os.path.join(rof_dir, f"{case_name}.mosart.h0.1985-*.nc")))
        if monthly_files:
            grid_ds = xr.open_dataset(monthly_files[0])
        else:
            daily_files_tmp = sorted(glob.glob(os.path.join(rof_dir, f"{case_name}.mosart.h1.1985-01-*.nc")))
            if daily_files_tmp:
                grid_ds = xr.open_dataset(daily_files_tmp[0])
        if grid_ds is None:
            print("No MOSART files found to infer grid.")
            return
        if 'lat' in grid_ds and 'lon' in grid_ds:
            lat_da = grid_ds['lat']
            lon_da = grid_ds['lon']
        else:
            # Attempt to find coordinates
            for k in list(grid_ds.data_vars) + list(grid_ds.coords):
                if isinstance(grid_ds[k], xr.DataArray) and ('lat' in grid_ds[k].coords) and ('lon' in grid_ds[k].coords):
                    lat_da = grid_ds[k].coords['lat']
                    lon_da = grid_ds[k].coords['lon']
                    break
        if lat_da is None or lon_da is None:
            print("Could not find lat/lon in MOSART file.")
            return
    except Exception as e:
        print(f"Error opening MOSART grid file: {e}")
        return

    # Build KDTree for gauge-to-grid mapping
    try:
        tree, tree_meta = build_kdtree_from_grid(lat_da, lon_da)
    except Exception as e:
        print(f"Error building KDTree: {e}")
        return

    # Map each gauge to nearest grid cell
    gauge_mappings = {}
    points = tree.data  # array of [lat, lon] for grid (flattened for 2D)
    for _, row in gauge_meta.iterrows():
        try:
            gid = str(row['gauge_id'])
            glat = float(row['lat'])
            glon_raw = float(row['lon'])
            glon = wrap_lon_to_180(glon_raw)
            dist, idx = tree.query([glat, glon])
            sel_info = grid_index_from_query(tree_meta, int(idx))
            matched_lat, matched_lon = points[int(idx)]
            gauge_mappings[gid] = {
                "sel_info": sel_info,
                "matched_lat": float(matched_lat),
                "matched_lon": float(matched_lon),
                "river_name": str(row.get('river_name', '')),
                "obs_area_km2": float(row.get('area_km2', np.nan)),
                "gauge_lat": glat,
                "gauge_lon": glon_raw,
                "dist_deg": float(dist),
            }
        except Exception as e:
            print(f"Gauge mapping failed for {row.get('gauge_id','?')}: {e}")

    # Open daily MOSART files for 1985-1989
    try:
        daily_files = []
        for yr in range(1985, 1990):
            daily_files.extend(sorted(glob.glob(os.path.join(rof_dir, f"{case_name}.mosart.h1.{yr}-*.nc"))))
        if len(daily_files) == 0:
            print("No daily MOSART files found for 1985-1989.")
            return

        def preprocess(ds):
            keep_vars = [v for v in [var_mosart] if v in ds.variables]
            ds2 = ds[keep_vars] if keep_vars else ds
            # ensure lat/lon retained as coords if present
            if 'lat' in ds:
                ds2 = ds2.assign_coords(lat=ds['lat'])
            if 'lon' in ds:
                ds2 = ds2.assign_coords(lon=ds['lon'])
            return ds2

        ds_daily = xr.open_mfdataset(daily_files, combine='by_coords', preprocess=preprocess, parallel=False, use_cftime=True)
        # Convert calendar to standard for pandas compatibility
        try:
            ds_daily = ds_daily.convert_calendar('standard', use_cftime=False)
        except Exception:
            try:
                ds_daily = ds_daily.convert_calendar('proleptic_gregorian', use_cftime=False)
            except Exception:
                # As a fallback, keep cftime but avoid pandas comparisons later
                pass
        ds_daily = ds_daily.sel(time=slice(start_date, end_date))
        if var_mosart not in ds_daily:
            print(f"Variable {var_mosart} not found in MOSART dataset.")
            return
    except Exception as e:
        print(f"Error opening daily MOSART files: {e}")
        return
    finally:
        try:
            if grid_ds is not None:
                grid_ds.close()
        except Exception:
            pass

    # Prepare outputs
    metrics_records = []
    fdc_records = []

    # Exceedance probabilities for FDC
    p_exceed = np.linspace(0.01, 0.99, 99)

    # Process each gauge
    for gid, mapping in gauge_mappings.items():
        # Read observation data
        obs_path = os.path.join(obs_streamflow_dir, f"{gid}.csv")
        try:
            obs_df = pd.read_csv(obs_path, parse_dates=['date'])
        except Exception as e:
            print(f"Skipping gauge {gid}: cannot read obs file ({e})")
            continue
        if 'discharge_m3s' not in obs_df.columns:
            print(f"Skipping gauge {gid}: 'discharge_m3s' column not found")
            continue

        obs_df = obs_df[['date', 'discharge_m3s']].dropna()
        obs_df = obs_df[(obs_df['date'] >= start_date) & (obs_df['date'] <= end_date)]
        obs_df = obs_df.set_index('date').sort_index()
        obs_series = obs_df['discharge_m3s']

        # Extract model series at mapped grid cell, slice in xarray to avoid calendar mismatch
        try:
            ts_model = select_timeseries(ds_daily, var_mosart, mapping["sel_info"], tree_meta)
            ts_model = ts_model.sel(time=slice(start_date, end_date))
            # Ensure model time is pandas-friendly; convert if necessary
            if isinstance(ts_model.indexes.get('time', None), xr.CFTimeIndex):
                # Convert calendar for just this series
                try:
                    ts_model = ts_model.convert_calendar('standard', use_cftime=False)
                except Exception:
                    try:
                        ts_model = ts_model.convert_calendar('proleptic_gregorian', use_cftime=False)
                    except Exception:
                        pass
            model_series = ts_model.to_series()
        except Exception as e:
            print(f"Skipping gauge {gid}: extraction error ({e})")
            continue

        # Align by intersection of dates
        try:
            df_join = pd.DataFrame({'obs': obs_series}).join(pd.DataFrame({'mod': model_series}), how='inner')
        except Exception as e:
            # As fallback, coerce model_series index to pandas datetime if possible
            try:
                model_series.index = pd.to_datetime(model_series.index.astype(str))
                df_join = pd.DataFrame({'obs': obs_series}).join(pd.DataFrame({'mod': model_series}), how='inner')
            except Exception as e2:
                print(f"Skipping gauge {gid}: alignment error ({e2})")
                continue

        df_join = df_join.dropna()
        if df_join.empty or len(df_join) < 10:
            print(f"Skipping gauge {gid}: insufficient overlapping data ({len(df_join)})")
            continue

        obs_vals = df_join['obs'].values.astype(float)
        mod_vals = df_join['mod'].values.astype(float)

        # Compute FDCs
        p_obs, q_obs = compute_fdc(obs_vals, p_exceed=p_exceed)
        p_mod, q_mod = compute_fdc(mod_vals, p_exceed=p_exceed)

        # Compute metrics
        try:
            wd = float(wasserstein_distance(mod_vals, obs_vals))
        except Exception:
            wd = np.nan

        sum_obs = float(np.nansum(obs_vals))
        sum_mod = float(np.nansum(mod_vals))
        if sum_obs != 0 and np.isfinite(sum_obs):
            vol_bias_frac = (sum_mod - sum_obs) / sum_obs
            vol_bias_pct = vol_bias_frac * 100.0
        else:
            vol_bias_frac = np.nan
            vol_bias_pct = np.nan

        # Quantiles at exceedance 0.1, 0.5, 0.9 -> non-exceed q=0.9,0.5,0.1
        ex_probs = [0.1, 0.5, 0.9]
        ne_probs = [1 - p for p in ex_probs]
        try:
            try:
                q_obs_10, q_obs_50, q_obs_90 = np.quantile(obs_vals, ne_probs, method='linear')
            except TypeError:
                q_obs_10, q_obs_50, q_obs_90 = np.quantile(obs_vals, ne_probs)
        except Exception:
            q_obs_10 = np.nan
            q_obs_50 = np.nan
            q_obs_90 = np.nan
        try:
            try:
                q_mod_10, q_mod_50, q_mod_90 = np.quantile(mod_vals, ne_probs, method='linear')
            except TypeError:
                q_mod_10, q_mod_50, q_mod_90 = np.quantile(mod_vals, ne_probs)
        except Exception:
            q_mod_10 = np.nan
            q_mod_50 = np.nan
            q_mod_90 = np.nan

        ratio_q10 = (q_mod_10 / q_obs_10) if (np.isfinite(q_obs_10) and q_obs_10 != 0) else np.nan
        ratio_q50 = (q_mod_50 / q_obs_50) if (np.isfinite(q_obs_50) and q_obs_50 != 0) else np.nan
        ratio_q90 = (q_mod_90 / q_obs_90) if (np.isfinite(q_obs_90) and q_obs_90 != 0) else np.nan

        metrics_records.append({
            "gauge_id": gid,
            "river_name": mapping.get("river_name", ""),
            "gauge_lat": mapping.get("gauge_lat", np.nan),
            "gauge_lon": mapping.get("gauge_lon", np.nan),
            "grid_lat": mapping.get("matched_lat", np.nan),
            "grid_lon": mapping.get("matched_lon", np.nan),
            "n_days": len(df_join),
            "wasserstein_distance_m3s": wd,
            "volume_bias_fraction": vol_bias_frac,
            "volume_bias_percent": vol_bias_pct,
            "Q10_obs_m3s": q_obs_10,
            "Q10_mod_m3s": q_mod_10,
            "Q10_ratio_mod_over_obs": ratio_q10,
            "Q50_obs_m3s": q_obs_50,
            "Q50_mod_m3s": q_mod_50,
            "Q50_ratio_mod_over_obs": ratio_q50,
            "Q90_obs_m3s": q_obs_90,
            "Q90_mod_m3s": q_mod_90,
            "Q90_ratio_mod_over_obs": ratio_q90
        })

        # Save FDC records
        for pe, qo, qm in zip(p_obs, q_obs, q_mod):
            fdc_records.append({
                "gauge_id": gid,
                "p_exceed": pe,
                "Q_obs_m3s": qo,
                "Q_mod_m3s": qm
            })

    # Convert to DataFrames and save
    metrics_df = pd.DataFrame(metrics_records)
    fdc_df = pd.DataFrame(fdc_records)

    try:
        metrics_csv_path = os.path.join(output_dir, "fdc_metrics_1985_1989.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics to {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

    try:
        fdc_csv_path = os.path.join(output_dir, "fdc_percentiles_1985_1989.csv")
        fdc_df.to_csv(fdc_csv_path, index=False)
        print(f"Saved FDC percentiles to {fdc_csv_path}")
    except Exception as e:
        print(f"Error saving FDC CSV: {e}")

    # Plot figure: map of Wasserstein distance and FDC panels for specified gauges
    target_gauges = [
        ("Amazon", "3629000"),
        ("Missouri", "4121801"),
        ("Columbia", "4115200"),
        ("Danube", "6742900"),
        ("Mekong", "2969100"),
        ("Orange", "1159100"),
    ]

    try:
        if metrics_df.empty:
            print("No metrics to plot.")
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.3)
            ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

            ax_map.set_global()
            ax_map.coastlines(linewidth=0.6)
            ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax_map.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')

            sc = ax_map.scatter(metrics_df['gauge_lon'].astype(float), metrics_df['gauge_lat'].astype(float),
                                c=metrics_df['wasserstein_distance_m3s'].astype(float),
                                cmap='viridis', s=40, edgecolor='k', linewidth=0.3, transform=ccrs.PlateCarree())
            cbar = plt.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.05, fraction=0.05)
            cbar.set_label("Wasserstein distance (m3/s)")
            ax_map.set_title("Wasserstein distance between E3SM (MOSART) and Observed Streamflow (1985–1989)")

            # Plot FDC panels
            fdc_axes = []
            positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            for (river_name, gid), pos in zip(target_gauges, positions):
                ax = fig.add_subplot(gs[pos[0], pos[1]])
                fdc_axes.append((ax, river_name, gid))

            for idx_ax, (ax, river_name, gid) in enumerate(fdc_axes):
                sub = fdc_df[fdc_df['gauge_id'] == gid].copy()
                if sub.empty and gid.isdigit():
                    sub = fdc_df[fdc_df['gauge_id'] == str(int(gid))]
                if sub.empty:
                    ax.text(0.5, 0.5, f"No data for gauge {gid}", ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                sub = sub.sort_values('p_exceed')
                ax.plot(sub['p_exceed'] * 100.0, sub['Q_obs_m3s'], label='Observed', color='k', lw=2)
                ax.plot(sub['p_exceed'] * 100.0, sub['Q_mod_m3s'], label='E3SM MOSART', color='tab:blue', lw=2)
                ax.set_yscale('log')
                ax.set_xlim(0, 100)
                ymin = np.nanmin([sub['Q_obs_m3s'].replace(0, np.nan).min(), sub['Q_mod_m3s'].replace(0, np.nan).min()])
                if np.isfinite(ymin) and ymin > 0:
                    ax.set_ylim(bottom=max(ymin * 0.5, 1e-3))
                ax.set_xlabel('Exceedance probability (%)')
                ax.set_ylabel('Discharge (m3/s)')
                mrow = metrics_df[metrics_df['gauge_id'] == gid]
                if mrow.empty and gid.isdigit():
                    mrow = metrics_df[metrics_df['gauge_id'] == str(int(gid))]
                wd_val = float(mrow['wasserstein_distance_m3s'].values[0]) if not mrow.empty else np.nan
                ax.set_title(f"{river_name} ({gid})  WD={wd_val:.1f} m3/s" if np.isfinite(wd_val) else f"{river_name} ({gid})")
                if idx_ax == 0:
                    ax.legend(loc='best', frameon=False)

            fig_path = os.path.join(output_dir, "map_and_fdc_1985_1989.png")
            try:
                fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                print(f"Saved figure to {fig_path}")
            except Exception as e:
                print(f"Error saving figure: {e}")
            plt.close(fig)
    except Exception as e:
        print(f"Error creating plot: {e}")

    # Close datasets
    try:
        ds_daily.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()