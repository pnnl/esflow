import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def main():
    warnings.filterwarnings("ignore")
    # User settings
    case_name = "sample.v3.LR.historical"
    start_date = "1985-01-01"
    end_date = "1989-12-31"
    model_dir = os.path.join("data", "sample", "e3sm", "rof")
    obs_dir = os.path.join("data", "sample", "obs", "streamflow")
    metadata_file = os.path.join("data", "sample", "obs", "gauge_metadata.csv")
    out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_04_streamflow_fdc/run1_debug/v1/output"
    os.makedirs(out_dir, exist_ok=True)

    # Load gauge metadata
    try:
        gmeta = pd.read_csv(metadata_file, dtype={"gauge_id": str})
        gmeta["gauge_id"] = gmeta["gauge_id"].astype(str)
    except Exception as e:
        print(f"Error reading gauge metadata: {e}")
        return

    # Find MOSART daily files
    pattern = os.path.join(model_dir, f"{case_name}.mosart.h1.*-*-*-00000.nc")
    mosart_files = sorted(glob.glob(pattern))
    if not mosart_files:
        print("No MOSART daily files found.")
        return

    # Open as one dataset
    try:
        ds = xr.open_mfdataset(mosart_files, combine="by_coords", decode_times=True, parallel=False)
        ds = ds.sel(time=slice(start_date, end_date))
        varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
        if varname not in ds:
            print(f"Variable {varname} not in dataset.")
            return
        var_da = ds[varname]
        lat_da = ds["lat"]
        lon_da = ds["lon"]
        # prepare 1d lat/lon arrays
        if lat_da.ndim == 2:
            lats = lat_da[:, 0].values
            lons = lon_da[0, :].values
            dim_y, dim_x = lat_da.dims
        else:
            lats = lat_da.values
            lons = lon_da.values
            dim_y = lat_da.dims[0]
            dim_x = lon_da.dims[0]
    except Exception as e:
        print(f"Error opening MOSART data: {e}")
        return

    # Prepare storage
    metrics = []
    fdc_dict = {}

    # Loop gauges
    for _, row in gmeta.iterrows():
        gid = row["gauge_id"]
        lat_g = row["lat"]
        lon_g = row["lon"]
        river_name = row.get("river_name", "")
        try:
            # nearest grid cell
            idx_y = int(np.argmin(np.abs(lats - lat_g)))
            idx_x = int(np.argmin(np.abs(lons - lon_g)))
            sim_da = var_da.isel({dim_y: idx_y, dim_x: idx_x})
            # convert to pandas series
            sim_series = sim_da.to_series()
            # convert cftime to pandas datetime
            try:
                sim_series.index = pd.to_datetime(sim_series.index.values)
            except Exception:
                sim_series.index = [pd.Timestamp(f"{t.year}-{t.month:02d}-{t.day:02d}") for t in sim_series.index]
            # select window
            sim_series = sim_series[(sim_series.index >= start_date) & (sim_series.index <= end_date)]
            # load observations
            obs_file = os.path.join(obs_dir, f"{gid}.csv")
            obs_df = pd.read_csv(obs_file, parse_dates=["date"])
            obs_df = obs_df.set_index("date")
            obs_series = obs_df["discharge_m3s"].loc[start_date:end_date]
            # align
            df = pd.DataFrame({"sim": sim_series, "obs": obs_series}).dropna()
            if df.empty:
                continue
            # metrics
            total_sim = df["sim"].sum()
            total_obs = df["obs"].sum()
            volume_bias = (total_sim - total_obs) / total_obs if total_obs != 0 else np.nan
            wdist = wasserstein_distance(df["sim"].values, df["obs"].values)
            # FDC percentiles
            ptiles = np.arange(0, 101)
            sim_pt = np.percentile(df["sim"].values, ptiles)
            obs_pt = np.percentile(df["obs"].values, ptiles)
            # quantile ratios
            def qr(p):
                return sim_pt[p] / obs_pt[p] if obs_pt[p] != 0 else np.nan
            Q10 = qr(10); Q50 = qr(50); Q90 = qr(90)
            # store
            metrics.append({
                "gauge_id": gid,
                "river_name": river_name,
                "lat": lat_g,
                "lon": lon_g,
                "volume_bias": volume_bias,
                "wasserstein_distance": wdist,
                "Q10_ratio": Q10,
                "Q50_ratio": Q50,
                "Q90_ratio": Q90
            })
            fdc_df = pd.DataFrame({
                "percentile": ptiles,
                "sim": sim_pt,
                "obs": obs_pt
            })
            fdc_dict[gid] = fdc_df
            # save FDC CSV
            try:
                fdc_df.to_csv(os.path.join(out_dir, f"{gid}_fdc.csv"), index=False)
            except Exception as e:
                print(f"Error saving FDC for {gid}: {e}")
        except Exception as e:
            print(f"Error processing gauge {gid}: {e}")
            continue

    # save metrics and plot if available
    if metrics:
        try:
            metrics_df = pd.DataFrame(metrics).set_index("gauge_id")
            metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"))
        except Exception as e:
            print(f"Error saving metrics: {e}")

        # Plot map + FDC panels for selected gauges
        selected = ["3629000", "4121801", "4115200", "6742900", "2969100", "1159100"]
        try:
            fig = plt.figure(figsize=(12, 16))
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
            # Map
            ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
            ax_map.set_global()
            ax_map.coastlines()
            ax_map.add_feature(cfeature.BORDERS, linestyle=':')
            # scatter by W-distance
            cmap = plt.cm.viridis
            wvals = [metrics_df.loc[gid, "wasserstein_distance"] for gid in selected if gid in metrics_df.index]
            if wvals:
                norm = plt.Normalize(min(wvals), max(wvals))
                for gid in selected:
                    if gid in metrics_df.index:
                        lat_g = metrics_df.loc[gid, "lat"]
                        lon_g = metrics_df.loc[gid, "lon"]
                        w = metrics_df.loc[gid, "wasserstein_distance"]
                        ax_map.scatter(lon_g, lat_g, color=cmap(norm(w)), s=80, edgecolor='k',
                                       transform=ccrs.PlateCarree(), zorder=5)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_map, orientation='horizontal', pad=0.05, fraction=0.05)
                cbar.set_label("Wasserstein Distance")
            ax_map.set_title("Gauge Locations Colored by Wasserstein Distance")

            # FDC panels
            for i, gid in enumerate(selected):
                row = 1 + i // 3
                col = i % 3
                ax = fig.add_subplot(gs[row, col])
                if gid in fdc_dict and gid in metrics_df.index:
                    df_fdc = fdc_dict[gid]
                    ax.plot(df_fdc["percentile"], df_fdc["obs"], label="Obs", color='black')
                    ax.plot(df_fdc["percentile"], df_fdc["sim"], label="Sim", color='red')
                    ax.set_yscale('log')
                    rname = metrics_df.loc[gid, "river_name"]
                    ax.set_title(f"{rname} ({gid})")
                    if row == 2:
                        ax.set_xlabel("Percentile (%)")
                    ax.set_ylabel("Discharge (m3/s)")
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, f"No data for {gid}", ha='center')
            fig.suptitle("Flow Duration Curves and Wasserstein Distances (1985-1989)", fontsize=16)
            figfile = os.path.join(out_dir, "fdc_wasserstein_map.png")
            plt.savefig(figfile, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting figure: {e}")
    else:
        print("No metrics computed; skipping save and plot steps.")

if __name__ == "__main__":
    main()