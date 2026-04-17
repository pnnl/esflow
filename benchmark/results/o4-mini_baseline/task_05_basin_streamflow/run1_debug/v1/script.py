import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import mapping
import warnings

warnings.filterwarnings('ignore')

def calc_metrics(sim, obs):
    sim = np.array(sim)
    obs = np.array(obs)
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    if mask.sum() == 0:
        return {'RMSE': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan}
    sim = sim[mask]
    obs = obs[mask]
    rmse = np.sqrt(np.mean((sim - obs)**2))
    denom = np.sum((obs - np.mean(obs))**2)
    nse = 1 - np.sum((obs - sim)**2) / denom if denom != 0 else np.nan
    r = np.corrcoef(obs, sim)[0,1] if obs.size > 1 else np.nan
    alpha = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    pbias = 100.0 * np.sum(sim - obs) / np.sum(obs) if np.sum(obs) != 0 else np.nan
    return {'RMSE': rmse, 'NSE': nse, 'KGE': kge, 'PBIAS': pbias}

def main():
    data_dir = "./data/sample/e3sm"
    case_name = "sample.v3.LR.historical"
    obs_dir = "./data/sample/obs/streamflow"
    gauge_meta_file = "./data/sample/obs/gauge_metadata.csv"
    basin_geojson = "./data/sample/obs/basin_polygons.geojson"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_05_basin_streamflow/run1_debug/v1/output"
    os.makedirs(output_dir, exist_ok=True)

    basins = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]

    try:
        gauge_meta = pd.read_csv(gauge_meta_file, dtype={'gauge_id': str})
        gauge_meta['gauge_id'] = gauge_meta['gauge_id'].astype(int)
    except Exception as e:
        print(f"Error reading gauge metadata: {e}")
        sys.exit(1)

    try:
        basin_gdf = gpd.read_file(basin_geojson)
        basin_gdf['grdc_no'] = basin_gdf['grdc_no'].astype(int)
    except Exception as e:
        print(f"Error reading basin polygons: {e}")
        sys.exit(1)

    try:
        pattern = os.path.join(data_dir, 'rof', f"{case_name}.mosart.h1.*-00000.nc")
        ds = xr.open_mfdataset(pattern, combine='by_coords', parallel=False)
        ds = ds.sel(time=slice('1985-01-01', '1989-12-31'))
    except Exception as e:
        print(f"Error loading MOSART data: {e}")
        sys.exit(1)

    try:
        lat = ds['lat'].values
        lon = ds['lon'].values
        lat_dim = ds['lat'].dims[0]
        lon_dim = ds['lon'].dims[0]
    except Exception as e:
        print(f"Error extracting grid coordinates: {e}")
        sys.exit(1)

    varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    metrics_list = []

    for basin in basins:
        print(f"Processing basin {basin}...")
        try:
            bpoly = basin_gdf[basin_gdf['grdc_no'] == basin]
            if bpoly.empty:
                print(f"No polygon found for basin {basin}, skipping.")
                continue
            gm = gauge_meta[gauge_meta['gauge_id'] == basin]
            if gm.empty:
                print(f"No gauge metadata for {basin}, skipping.")
                continue
            glon = gm.iloc[0]['lon']
            glat = gm.iloc[0]['lat']
            obs_file = os.path.join(obs_dir, f"{basin}.csv")
            obs_df = pd.read_csv(obs_file, parse_dates=['date'])
            obs_df = obs_df.set_index('date').sort_index()
            obs_series = obs_df['discharge_m3s']
            lat_idx = int(np.argmin(np.abs(lat - glat)))
            lon_idx = int(np.argmin(np.abs(lon - glon)))
            sim_da = ds[varname].isel({lat_dim: lat_idx, lon_dim: lon_idx})
            sim_series = sim_da.to_series().rename("sim")
            df = pd.concat([obs_series.rename("obs"), sim_series], axis=1).dropna()
            mets = calc_metrics(df['sim'], df['obs'])
            mets['basin'] = basin
            metrics_list.append(mets)
            out_ts_file = os.path.join(output_dir, f"basin_{basin}_streamflow.csv")
            try:
                df.to_csv(out_ts_file, index_label='date')
            except Exception as e:
                print(f"Error saving time series CSV for basin {basin}: {e}")
            try:
                fig = plt.figure(figsize=(10,8))
                ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
                ax1.coastlines(resolution='50m', color='black', linewidth=1)
                bpoly = bpoly.to_crs(epsg=4326)
                ax1.add_geometries(bpoly.geometry, crs=ccrs.PlateCarree(),
                                   facecolor='none', edgecolor='blue', linewidth=2)
                minx, miny, maxx, maxy = bpoly.total_bounds
                padx = (maxx - minx) * 0.1
                pady = (maxy - miny) * 0.1
                ax1.set_extent([minx-padx, maxx+padx, miny-pady, maxy+pady], crs=ccrs.PlateCarree())
                ax1.set_title(f"Basin {basin} Boundary")
                ax2 = fig.add_subplot(2,1,2)
                ax2.plot(df.index, df['obs'], label='Obs', color='black')
                ax2.plot(df.index, df['sim'], label='Sim', color='red')
                ax2.set_ylabel("Discharge (m3/s)")
                ax2.set_xlabel("Date")
                ax2.legend()
                txt = (f"RMSE: {mets['RMSE']:.2f}\n"
                       f"NSE: {mets['NSE']:.2f}\n"
                       f"KGE: {mets['KGE']:.2f}\n"
                       f"PBIAS: {mets['PBIAS']:.2f}%")
                ax2.text(0.98, 0.95, txt, transform=ax2.transAxes,
                         ha='right', va='top', bbox=dict(boxstyle="round", fc="w"))
                fig.tight_layout()
                out_png = os.path.join(output_dir, f"basin_{basin}_timeseries.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting for basin {basin}: {e}")
        except Exception as e:
            print(f"Error processing basin {basin}: {e}")

    try:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    except Exception as e:
        print(f"Error saving metrics summary: {e}")

    print("Processing completed.")

if __name__ == "__main__":
    main()