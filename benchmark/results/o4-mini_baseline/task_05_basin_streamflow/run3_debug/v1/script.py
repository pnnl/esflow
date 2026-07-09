import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import pearsonr

# define metrics
def compute_metrics(sim, obs):
    sim = np.array(sim)
    obs = np.array(obs)
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]
    if len(sim) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1 - np.sum((sim - obs) ** 2) / denom if denom != 0 else np.nan
    pbias = 100.0 * np.sum(sim - obs) / np.sum(obs) if np.sum(obs) != 0 else np.nan
    r = pearsonr(sim, obs)[0] if len(sim) > 1 else np.nan
    alpha = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return rmse, nse, kge, pbias

# paths and parameters
case_name = "sample.v3.LR.historical"
e3sm_root = "./data/sample/e3sm"
mosart_dir = os.path.join(e3sm_root, "rof")
obs_stream_dir = "./data/sample/obs/streamflow"
gauge_meta_file = "./data/sample/obs/gauge_metadata.csv"
basin_geojson = "./data/sample/obs/basin_polygons.geojson"
out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_05_basin_streamflow/run3_debug/v1/output"
os.makedirs(out_dir, exist_ok=True)

selected_ids = ["3629000", "4121801", "4115200", "6742900", "2969100", "1159100"]

meta = pd.read_csv(gauge_meta_file, dtype={"gauge_id": str})
meta = meta[meta["gauge_id"].isin(selected_ids)].set_index("gauge_id")

basins = gpd.read_file(basin_geojson)
basins["grdc_no"] = basins["grdc_no"].astype(str)
basins = basins[basins["grdc_no"].isin(selected_ids)].set_index("grdc_no")

pattern = os.path.join(mosart_dir, f"{case_name}.mosart.h1.*.nc")
all_files = sorted(glob.glob(pattern))

file_dates = []
for f in all_files:
    try:
        fname = os.path.basename(f)
        date_part = fname.split(".")[-2]
        date_str = "-".join(date_part.split("-")[:3])
        dt = pd.to_datetime(date_str)
        if 1985 <= dt.year <= 1989:
            file_dates.append((dt, f))
    except Exception:
        continue
file_dates = sorted(file_dates, key=lambda x: x[0])
sim_dates = [dt for dt, _ in file_dates]

if not file_dates:
    raise RuntimeError("No MOSART h1 files found for the period.")

first_file = file_dates[0][1]
ds0 = xr.open_dataset(first_file)
lats = ds0["lat"].values
lons = ds0["lon"].values
ds0.close()

gauge_idxs = {}
for gid, row in meta.iterrows():
    glat = row["lat"]
    glon = row["lon"]
    if lons.ndim == 1 and lats.ndim == 1:
        lon_arr = lons
        lat_arr = lats
        if lon_arr.max() > 180 and glon < 0:
            glon2 = glon + 360
        else:
            glon2 = glon
        idx_lat = np.argmin(np.abs(lat_arr - glat))
        idx_lon = np.argmin(np.abs(lon_arr - glon2))
        gauge_idxs[gid] = (idx_lat, idx_lon)
    else:
        flat_lats = lats.flatten()
        flat_lons = lons.flatten()
        if flat_lons.max() > 180 and glon < 0:
            glon2 = glon + 360
        else:
            glon2 = glon
        dist2 = (flat_lats - glat) ** 2 + (flat_lons - glon2) ** 2
        idx_flat = np.argmin(dist2)
        idx = np.unravel_index(idx_flat, lats.shape)
        gauge_idxs[gid] = idx

sim_data = {gid: [] for gid in selected_ids}
for dt, f in file_dates:
    try:
        ds = xr.open_dataset(f)
        var = ds["RIVER_DISCHARGE_OVER_LAND_LIQ"]
        for gid in selected_ids:
            idx = gauge_idxs[gid]
            try:
                if var.ndim == 3:
                    val = var.values[0, idx[0], idx[1]]
                elif var.ndim == 2:
                    val = var.values[idx[0], idx[1]]
                else:
                    val = np.nan
            except Exception:
                val = np.nan
            sim_data[gid].append(val)
        ds.close()
    except Exception as e:
        print(f"Error reading {f}: {e}")
        for gid in selected_ids:
            sim_data[gid].append(np.nan)

sim_series = {}
for gid in selected_ids:
    sim_series[gid] = pd.Series(data=sim_data[gid], index=pd.DatetimeIndex(sim_dates))

metrics_list = []
for gid in selected_ids:
    row = meta.loc[gid]
    river_name = row["river_name"]
    obs_file = os.path.join(obs_stream_dir, f"{gid}.csv")
    try:
        obs_df = pd.read_csv(obs_file, parse_dates=["date"])
        obs_df = obs_df.set_index("date")
        obs = obs_df["discharge_m3s"].sort_index()
        obs = obs[(obs.index.year >= 1985) & (obs.index.year <= 1989)]
    except Exception as e:
        print(f"Error loading obs for {gid}: {e}")
        continue
    sim = sim_series[gid]
    df = pd.DataFrame({"obs": obs, "sim": sim})
    df = df.dropna()
    if df.empty:
        print(f"No overlapping data for {gid}")
        continue
    rmse, nse, kge, pbias = compute_metrics(df["sim"], df["obs"])
    metrics_list.append({
        "gauge_id": gid,
        "river_name": river_name,
        "RMSE": rmse,
        "NSE": nse,
        "KGE": kge,
        "PBIAS": pbias
    })
    try:
        basin = basins.loc[[gid]]
        fig = plt.figure(figsize=(12,5))
        ax_map = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
        basin.plot(ax=ax_map, facecolor='none', edgecolor='blue', transform=ccrs.PlateCarree())
        minx, miny, maxx, maxy = basin.total_bounds
        ax_map.set_extent([minx-1, maxx+1, miny-1, maxy+1], crs=ccrs.PlateCarree())
        ax_map.coastlines()
        ax_map.set_title(f"{river_name} Basin")
        ax_ts = fig.add_subplot(1,2,2)
        ax_ts.plot(df.index, df["obs"], label="Observed", color="black", linewidth=1)
        ax_ts.plot(df.index, df["sim"], label="Simulated", color="red", linewidth=1)
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel("Discharge (m³/s)")
        ax_ts.legend()
        txt = f"RMSE: {rmse:.2f}\nNSE: {nse:.2f}\nKGE: {kge:.2f}\nPBIAS: {pbias:.1f}%"
        ax_ts.text(0.05, 0.95, txt, transform=ax_ts.transAxes, verticalalignment='top')
        ax_ts.set_title(f"{river_name} Streamflow 1985-1989")
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"{gid}_{river_name}_streamflow.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting for {gid}: {e}")

metrics_df = pd.DataFrame(metrics_list)
out_csv = os.path.join(out_dir, "basin_streamflow_metrics.csv")
try:
    metrics_df.to_csv(out_csv, index=False)
except Exception as e:
    print(f"Error saving metrics CSV: {e}")