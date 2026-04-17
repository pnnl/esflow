import os
import sys
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

try:
    from shapely.geometry import shape as shapely_shape, Point as ShapelyPoint
except Exception:
    shapely_shape = None
    ShapelyPoint = None


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}", file=sys.stderr)


def normalize_lon(lon, grid_lons):
    # Normalize lon to match grid convention
    try:
        glon_min = np.nanmin(grid_lons)
        glon_max = np.nanmax(grid_lons)
    except Exception:
        glon_min, glon_max = -180, 180
    if glon_max > 180:
        # grid is 0..360; convert lon to 0..360
        lon = lon % 360.0
    else:
        # grid is -180..180; bring lon to [-180,180]
        lon = ((lon + 180) % 360) - 180
    return lon


def angular_lon_diff(lon_arr, lon0):
    # Compute minimum difference considering wrap-around
    # lon_arr and lon0 in same range
    d = np.abs(lon_arr - lon0)
    return np.minimum(d, 360 - d)


def find_lat_lon_vars(ds):
    lat_name = None
    lon_name = None
    for name in ds.variables:
        lname = name.lower()
        if lname in ["lat", "latitude"] and lat_name is None:
            lat_name = name
        if lname in ["lon", "longitude"] and lon_name is None:
            lon_name = name
    if lat_name is None or lon_name is None:
        # Try coords
        for name in ds.coords:
            lname = name.lower()
            if lname in ["lat", "latitude"] and lat_name is None:
                lat_name = name
            if lname in ["lon", "longitude"] and lon_name is None:
                lon_name = name
    return lat_name, lon_name


def find_nearest_grid_cell(ds, lat0, lon0):
    lat_name, lon_name = find_lat_lon_vars(ds)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not find 'lat' and 'lon' variables in dataset.")
    latvar = ds[lat_name]
    lonvar = ds[lon_name]

    lat_vals = np.asarray(latvar.values)
    lon_vals = np.asarray(lonvar.values)

    # Normalize target lon to grid convention
    lon0n = normalize_lon(lon0, lon_vals)

    # Determine case and compute distance
    if latvar.ndim == 1 and lonvar.ndim == 1:
        if latvar.dims == lonvar.dims:
            # Case A: 1D same dim (e.g., 'ncol')
            lat_diff = lat_vals - lat0
            lon_diff = angular_lon_diff(lon_vals, lon0n)
            dist2 = lat_diff**2 + lon_diff**2
            idx = int(np.nanargmin(dist2))
            indexers = {latvar.dims[0]: idx}
            grid_lat = float(lat_vals[idx])
            grid_lon = float(lon_vals[idx])
            return indexers, grid_lat, grid_lon
        else:
            # Case B: unstructured 1D lat and lon each on their own dims: treat as structured grid (lat x lon)
            lat_dim = latvar.dims[0]
            lon_dim = lonvar.dims[0]
            lat_diff = lat_vals - lat0  # shape (nlat,)
            lon_diff = angular_lon_diff(lon_vals, lon0n)  # shape (nlon,)
            # Create 2D grid of distances
            dist2 = (lat_diff[:, None]**2) + (lon_diff[None, :]**2)
            i_lat, j_lon = np.unravel_index(np.nanargmin(dist2), dist2.shape)
            indexers = {lat_dim: int(i_lat), lon_dim: int(j_lon)}
            grid_lat = float(lat_vals[i_lat])
            grid_lon = float(lon_vals[j_lon])
            return indexers, grid_lat, grid_lon
    elif latvar.ndim == 2 and lonvar.ndim == 2:
        # Case C: 2D arrays
        lat_diff = lat_vals - lat0
        lon_diff = angular_lon_diff(lon_vals, lon0n)
        dist2 = lat_diff**2 + lon_diff**2
        i, j = np.unravel_index(np.nanargmin(dist2), dist2.shape)
        dim_y, dim_x = latvar.dims
        indexers = {dim_y: int(i), dim_x: int(j)}
        grid_lat = float(lat_vals[i, j])
        grid_lon = float(lon_vals[i, j])
        return indexers, grid_lat, grid_lon
    else:
        # Fallback: flatten arrays if possible
        lat_flat = lat_vals.ravel()
        lon_flat = lon_vals.ravel()
        lat_diff = lat_flat - lat0
        lon_diff = angular_lon_diff(lon_flat, lon0n)
        dist2 = lat_diff**2 + lon_diff**2
        k = int(np.nanargmin(dist2))
        # Map flat index back to dims
        if latvar.ndim == 1:
            idx = k
            indexers = {latvar.dims[0]: idx}
            grid_lat = float(lat_flat[idx])
            grid_lon = float(lon_flat[idx])
            return indexers, grid_lat, grid_lon
        elif latvar.ndim == 2:
            i, j = np.unravel_index(k, lat_vals.shape)
            dim_y, dim_x = latvar.dims
            indexers = {dim_y: int(i), dim_x: int(j)}
            grid_lat = float(lat_vals[i, j])
            grid_lon = float(lon_vals[i, j])
            return indexers, grid_lat, grid_lon
        else:
            raise ValueError("Unable to determine nearest grid cell: unsupported lat/lon dimensions.")


def compute_metrics(obs, sim):
    # Both inputs are 1D numpy arrays with same length
    # Remove NaNs
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    n = len(obs)
    metrics = {"RMSE": np.nan, "NSE": np.nan, "KGE": np.nan, "PBIAS": np.nan, "N": n}
    if n == 0:
        return metrics
    # RMSE
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    metrics["RMSE"] = float(rmse)
    # NSE
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        nse = np.nan
    else:
        nse = 1 - np.sum((sim - obs) ** 2) / denom
    metrics["NSE"] = float(nse) if np.isfinite(nse) else np.nan
    # KGE
    mean_sim = np.mean(sim)
    mean_obs = np.mean(obs)
    std_sim = np.std(sim, ddof=0)
    std_obs = np.std(obs, ddof=0)
    # correlation
    if n > 1 and (np.std(sim) > 0) and (np.std(obs) > 0):
        r = np.corrcoef(sim, obs)[0, 1]
    else:
        r = np.nan
    alpha = (std_sim / std_obs) if std_obs != 0 else np.nan
    beta = (mean_sim / mean_obs) if mean_obs != 0 else np.nan
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        kge = np.nan
    metrics["KGE"] = float(kge) if np.isfinite(kge) else np.nan
    # PBIAS
    denom_pb = np.sum(obs)
    if denom_pb != 0:
        pbias = 100.0 * np.sum(sim - obs) / denom_pb
    else:
        pbias = np.nan
    metrics["PBIAS"] = float(pbias) if np.isfinite(pbias) else np.nan
    return metrics


def to_monthly_mean(series, start, end):
    # Ensure DatetimeIndex
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            s.index = pd.to_datetime(s.index.values)
    # Filter time range
    s = s[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
    # Resample to monthly mean (end of month)
    s_m = s.resample("M").mean()
    # Constrain to exact monthly range to ensure consistent indices
    monthly_index = pd.date_range(start=start, end=end, freq="M")
    s_m = s_m.reindex(monthly_index)
    return s_m


def load_basin_polygons(geojson_path):
    if shapely_shape is None:
        print("Shapely not available; basin polygons will not be plotted.", file=sys.stderr)
        return {}
    try:
        with open(geojson_path, "r") as f:
            gj = json.load(f)
    except Exception as e:
        print(f"Failed to load basin polygons GeoJSON: {e}", file=sys.stderr)
        return {}
    geom_map = {}
    try:
        for feature in gj.get("features", []):
            props = feature.get("properties", {})
            grdc_no = props.get("grdc_no", None)
            if grdc_no is None:
                continue
            try:
                gid = int(str(grdc_no))
            except Exception:
                continue
            geom = shapely_shape(feature.get("geometry", {}))
            geom_map[gid] = geom
    except Exception as e:
        print(f"Error parsing basin polygons: {e}", file=sys.stderr)
    return geom_map


def plot_basin_timeseries_map(output_path, basin_name, gauge_id, basin_geom, gauge_lat, gauge_lon,
                              grid_lat, grid_lon, df_monthly, metrics):
    try:
        fig = plt.figure(figsize=(12, 5))
        # Left: map
        ax_map = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='aliceblue')
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        # Plot basin polygon
        if basin_geom is not None and shapely_shape is not None:
            try:
                ax_map.add_geometries([basin_geom], crs=ccrs.PlateCarree(),
                                      facecolor='none', edgecolor='tab:green', linewidth=1.5, alpha=0.8)
                minx, miny, maxx, maxy = basin_geom.bounds
                dx = (maxx - minx) * 0.15 if (maxx - minx) > 0 else 5
                dy = (maxy - miny) * 0.15 if (maxy - miny) > 0 else 5
                ax_map.set_extent([minx - dx, maxx + dx, miny - dy, maxy + dy], crs=ccrs.PlateCarree())
            except Exception as e:
                print(f"Error plotting basin geometry for {gauge_id}: {e}", file=sys.stderr)
        # Plot gauge and grid cell
        ax_map.plot(gauge_lon, gauge_lat, marker='o', color='red', markersize=5,
                    transform=ccrs.PlateCarree(), label='Gauge')
        if grid_lat is not None and grid_lon is not None:
            # Normalize grid_lon for plotting in [-180,180]
            grid_lon_plot = ((grid_lon + 180) % 360) - 180
            ax_map.plot(grid_lon_plot, grid_lat, marker='x', color='blue', markersize=6,
                        transform=ccrs.PlateCarree(), label='Model cell')
        ax_map.legend(loc='lower left', fontsize=8)
        ax_map.set_title(f"{basin_name} (Gauge {gauge_id}) Basin and Locations")

        # Right: time series
        ax_ts = plt.subplot(1, 2, 2)
        ax_ts.plot(df_monthly.index, df_monthly["obs_m3s"], label="Observed", color='k', linewidth=1.5)
        ax_ts.plot(df_monthly.index, df_monthly["sim_m3s"], label="Simulated", color='tab:blue', linewidth=1.5)
        ax_ts.set_title(f"Monthly Discharge 1985-1989")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Discharge (m3/s)")
        ax_ts.grid(True, linestyle='--', alpha=0.5)
        ax_ts.legend()

        # Metrics box
        textstr = (
            f"N={metrics.get('N', np.nan)}\n"
            f"RMSE={metrics.get('RMSE', np.nan):.2f}\n"
            f"NSE={metrics.get('NSE', np.nan):.3f}\n"
            f"KGE={metrics.get('KGE', np.nan):.3f}\n"
            f"PBIAS={metrics.get('PBIAS', np.nan):.1f}%"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_ts.text(0.02, 0.98, textstr, transform=ax_ts.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)

        fig.suptitle(f"{basin_name} (Gauge {gauge_id})", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot and save figure {output_path}: {e}", file=sys.stderr)


def main():
    # Input settings
    case_name = "sample.v3.LR.historical"
    base_dir = "./data/sample/e3sm"
    rof_dir = os.path.join(base_dir, "rof")
    varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    years = list(range(1985, 1990))
    start_date = "1985-01-01"
    end_date = "1989-12-31"

    # Observation data
    gauge_metadata_csv = "./data/sample/obs/gauge_metadata.csv"
    obs_streamflow_dir = "./data/sample/obs/streamflow"
    basin_geojson = "./data/sample/obs/basin_polygons.geojson"

    # Output directory (as specified by user)
    outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_05_basin_streamflow/run3_output"
    ensure_dir(outdir)

    # Basins of interest: mapping basin names to gauge IDs
    basins = {
        "Amazon": 3629000,
        "Missouri": 4121801,
        "Columbia": 4115200,
        "Danube": 6742900,
        "Mekong": 2969100,
        "Orange": 1159100
    }

    # Prepare list of monthly MOSART files for 1985-1989
    mosart_files = []
    for y in years:
        for m in range(1, 13):
            fname = f"{case_name}.mosart.h0.{y:04d}-{m:02d}.nc"
            fpath = os.path.join(rof_dir, fname)
            if os.path.exists(fpath):
                mosart_files.append(fpath)
    if len(mosart_files) == 0:
        print("No MOSART monthly files found for the specified period.", file=sys.stderr)
        return

    # Open dataset
    try:
        ds = xr.open_mfdataset(
            mosart_files,
            combine='by_coords',
            parallel=False,
            decode_times=True
        )
    except Exception as e:
        print(f"Failed to open MOSART dataset: {e}", file=sys.stderr)
        return

    # Ensure variable exists
    if varname not in ds.variables:
        print(f"Variable {varname} not found in dataset.", file=sys.stderr)
        try:
            ds.close()
        except Exception:
            pass
        return

    # Load gauge metadata
    try:
        meta = pd.read_csv(gauge_metadata_csv)
    except Exception as e:
        print(f"Failed to read gauge metadata: {e}", file=sys.stderr)
        try:
            ds.close()
        except Exception:
            pass
        return

    # Load basin polygons
    basin_geoms = load_basin_polygons(basin_geojson)

    # Prepare output structures
    metrics_rows = []

    # Iterate over basins
    for basin_name, gauge_id in basins.items():
        try:
            # Get gauge metadata row
            row = meta.loc[meta['gauge_id'] == int(gauge_id)]
        except Exception:
            row = pd.DataFrame()
        if row.empty:
            print(f"Gauge ID {gauge_id} not found in metadata; skipping {basin_name}.", file=sys.stderr)
            continue
        gauge_lat = float(row.iloc[0]['lat'])
        gauge_lon = float(row.iloc[0]['lon'])

        # Find nearest grid cell
        try:
            indexers, grid_lat, grid_lon = find_nearest_grid_cell(ds, gauge_lat, gauge_lon)
        except Exception as e:
            print(f"Failed to find nearest grid cell for gauge {gauge_id} ({basin_name}): {e}", file=sys.stderr)
            indexers, grid_lat, grid_lon = None, None, None

        # Extract simulated time series
        sim_series_monthly = None
        if indexers is not None:
            try:
                da = ds[varname]
                da_sel = da.isel(indexers)
                # Ensure time slicing 1985-1989
                if 'time' in da_sel.coords:
                    da_sel = da_sel.sel(time=slice(np.datetime64(start_date), np.datetime64(end_date)))
                # Convert to pandas series
                sim_series = da_sel.to_series()
                # Ensure DatetimeIndex
                if not isinstance(sim_series.index, pd.DatetimeIndex):
                    try:
                        sim_series.index = pd.to_datetime(sim_series.index)
                    except Exception:
                        sim_series.index = pd.to_datetime(da_sel['time'].values)
                sim_series_monthly = to_monthly_mean(sim_series, start_date, end_date)
            except Exception as e:
                print(f"Failed to extract simulated series for gauge {gauge_id} ({basin_name}): {e}", file=sys.stderr)

        # Load observed daily discharge and compute monthly mean
        obs_series_monthly = None
        obs_csv = os.path.join(obs_streamflow_dir, f"{gauge_id}.csv")
        try:
            obs_df = pd.read_csv(obs_csv, parse_dates=['date'])
            obs_df = obs_df.set_index('date')
            obs_series = obs_df['discharge_m3s']
            obs_series_monthly = to_monthly_mean(obs_series, start_date, end_date)
        except Exception as e:
            print(f"Failed to load observation for gauge {gauge_id} ({basin_name}): {e}", file=sys.stderr)

        if sim_series_monthly is None or obs_series_monthly is None:
            print(f"Missing sim or obs for {basin_name} ({gauge_id}); skipping.", file=sys.stderr)
            continue

        # Align and compute metrics
        df = pd.DataFrame({
            "sim_m3s": sim_series_monthly,
            "obs_m3s": obs_series_monthly
        })
        df = df.dropna(subset=["sim_m3s", "obs_m3s"])

        if df.empty:
            print(f"No overlapping monthly data for {basin_name} ({gauge_id}); skipping.", file=sys.stderr)
            continue

        metrics = compute_metrics(df["obs_m3s"].values, df["sim_m3s"].values)
        metrics_row = {
            "basin": basin_name,
            "gauge_id": gauge_id,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            **metrics
        }
        metrics_rows.append(metrics_row)

        # Save per-basin CSV
        csv_out = os.path.join(outdir, f"{basin_name}_{gauge_id}_monthly_discharge_1985-1989.csv")
        try:
            df_out = df.copy()
            df_out = df_out.reset_index()
            df_out.rename(columns={"index": "time"}, inplace=True)
            df_out["time"] = df_out["time"].dt.strftime("%Y-%m")
            df_out.to_csv(csv_out, index=False)
        except Exception as e:
            print(f"Failed to save CSV for {basin_name} ({gauge_id}): {e}", file=sys.stderr)

        # Plot and save figure
        basin_geom = basin_geoms.get(int(gauge_id), None)
        fig_out = os.path.join(outdir, f"{basin_name}_{gauge_id}_map_timeseries.png")
        try:
            plot_basin_timeseries_map(
                fig_out, basin_name, gauge_id, basin_geom,
                gauge_lat, gauge_lon, grid_lat, grid_lon,
                df, metrics
            )
        except Exception as e:
            print(f"Error generating figure for {basin_name} ({gauge_id}): {e}", file=sys.stderr)

    # Save metrics summary CSV
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(outdir, "basin_streamflow_metrics_1985-1989.csv")
        try:
            metrics_df.to_csv(metrics_csv, index=False)
        except Exception as e:
            print(f"Failed to save metrics summary CSV: {e}", file=sys.stderr)
    else:
        print("No metrics to save.", file=sys.stderr)

    # Close dataset
    try:
        ds.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()