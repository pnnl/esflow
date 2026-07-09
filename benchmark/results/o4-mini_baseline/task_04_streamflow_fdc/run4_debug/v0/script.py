import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance

# Directories and parameters
data_dir = "./data/sample/e3sm"
rof_dir = os.path.join(data_dir, "rof")
case = "sample.v3.LR.historical"
obs_dir = "./data/sample/obs/streamflow"
meta_file = "./data/sample/obs/gauge_metadata.csv"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_04_streamflow_fdc/run4_output"
os.makedirs(output_dir, exist_ok=True)

# Time range
start = np.datetime64("1985-01-01")
end = np.datetime64("1989-12-31")

# Percentile levels
q_levels = [0.1, 0.5, 0.9]

# Six gauges for detailed panels
panel_gauges = ["3629000","4121801","4115200","6742900","2969100","1159100"]

# Load gauge metadata
meta = pd.read_csv(meta_file, dtype={"gauge_id": str})
meta = meta.set_index("gauge_id")

# Load river routing grid lat/lon from a sample daily file
daily_pattern = os.path.join(rof_dir, f"{case}.mosart.h1.*-00000.nc")
daily_files = sorted(glob.glob(daily_pattern))
if not daily_files:
    raise FileNotFoundError("No daily MOSART files found")
# pick first file for grid
ds0 = xr.open_dataset(daily_files[0])
grid_lat = ds0["lat"]
grid_lon = ds0["lon"]
ds0.close()

# Find nearest grid cell for each gauge
gauge_info = {}
for gid, row in meta.iterrows():
    lat0 = row["lat"]
    lon0 = row["lon"]
    # adjust lon if needed (model lon 0-360)
    if grid_lon.min() >= 0 and lon0 < 0:
        lon0_mod = lon0 + 360
    else:
        lon0_mod = lon0
    # compute distance
    dist = np.sqrt((grid_lat - lat0)**2 + (grid_lon - lon0_mod)**2)
    idx = np.unravel_index(np.argmin(dist.values), dist.shape)
    gauge_info[gid] = {"i_lat": idx[0], "i_lon": idx[1], "lat": lat0, "lon": lon0_mod}

# Load all daily discharge
try:
    ds = xr.open_mfdataset(daily_files, combine="by_coords", parallel=False)
    ds = ds.sel(time=slice(start, end))
    var = ds["RIVER_DISCHARGE_OVER_LAND_LIQ"]
except Exception as e:
    print("Error loading model data:", e)
    raise

# Prepare results containers
metrics = []
fdc_data = []

# Loop gauges
for gid, info in gauge_info.items():
    # model series
    sim = var[:, info["i_lat"], info["i_lon"]].to_series().rename("sim")
    sim.index = pd.to_datetime(sim.index.values)
    sim = sim.sort_index()
    sim = sim[start:end]

    # obs series
    obs_file = os.path.join(obs_dir, f"{gid}.csv")
    try:
        obs = pd.read_csv(obs_file, parse_dates=["date"])
    except Exception as e:
        print(f"Failed to read obs for {gid}:", e)
        continue
    obs = obs.set_index("date")["discharge_m3s"].rename("obs")
    obs = obs.sort_index().loc[start:end]

    # align on intersection
    df = pd.concat([sim, obs], axis=1).dropna()
    if df.empty:
        print(f"No overlapping data for gauge {gid}")
        continue

    sim_al = df["sim"]
    obs_al = df["obs"]

    # volume bias
    vol_bias = (sim_al.sum() - obs_al.sum()) / obs_al.sum()

    # Wasserstein distance
    wd = wasserstein_distance(obs_al.values, sim_al.values)

    # quantiles and ratios
    q_obs = np.quantile(obs_al.values, q_levels)
    q_sim = np.quantile(sim_al.values, q_levels)
    ratios = q_sim / q_obs

    # store metrics
    metrics.append({
        "gauge_id": gid,
        "volume_bias": vol_bias,
        "wasserstein_distance": wd,
        "q10_ratio": ratios[0],
        "q50_ratio": ratios[1],
        "q90_ratio": ratios[2]
    })

    # flow duration curve data
    n = len(df)
    excp = np.arange(1, n+1) / (n+1)
    sim_sorted = np.sort(sim_al.values)[::-1]
    obs_sorted = np.sort(obs_al.values)[::-1]
    df_fdc = pd.DataFrame({
        "gauge_id": gid,
        "exceedance": excp,
        "sim": sim_sorted,
        "obs": obs_sorted
    })
    fdc_data.append(df_fdc)

# Save metrics and FDCs
metrics_df = pd.DataFrame(metrics)
try:
    metrics_df.to_csv(os.path.join(output_dir, "streamflow_fdc_metrics.csv"), index=False)
except Exception as e:
    print("Error saving metrics CSV:", e)

fdc_all = pd.concat(fdc_data, ignore_index=True)
try:
    fdc_all.to_csv(os.path.join(output_dir, "streamflow_fdc_curves.csv"), index=False)
except Exception as e:
    print("Error saving FDC curves CSV:", e)

# Plotting map of Wasserstein distances and FDC panels
try:
    fig = plt.figure(figsize=(15,10))
    # Map subplot
    ax_map = plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=3, projection=ccrs.PlateCarree())
    ax_map.coastlines()
    sc = ax_map.scatter(
        [meta.loc[gid,"lon"] for gid in metrics_df["gauge_id"]],
        [meta.loc[gid,"lat"] for gid in metrics_df["gauge_id"]],
        c=metrics_df["wasserstein_distance"], cmap="viridis", s=50,
        transform=ccrs.PlateCarree()
    )
    plt.colorbar(sc, ax=ax_map, label="Wasserstein distance")
    ax_map.set_title("Gauge Wasserstein distances")

    # FDC panels
    for idx, gid in enumerate(panel_gauges):
        r = idx // 2
        c = idx % 2
        ax = plt.subplot2grid((3,4), (r,2+c), projection=None)
        sub = fdc_all[fdc_all["gauge_id"]==gid]
        if sub.empty: continue
        ax.plot(sub["exceedance"], sub["obs"], label="obs", color="black")
        ax.plot(sub["exceedance"], sub["sim"], label="sim", color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{meta.loc[gid,'river_name']} ({gid})")
        if r==2:
            ax.set_xlabel("Exceedance probability")
        if c==0:
            ax.set_ylabel("Discharge (m3/s)")
        ax.legend(fontsize="small")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "streamflow_fdc_map_and_panels.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
except Exception as e:
    print("Error in plotting:", e)