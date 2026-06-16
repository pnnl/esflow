import os
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def make_output_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def load_elm_means(case_name, data_dir, start_year, end_year):
    lnd_dir = os.path.join(data_dir, 'e3sm', 'lnd')
    pattern = os.path.join(lnd_dir, f"{case_name}.elm.h0.*.nc")
    try:
        ds = xr.open_mfdataset(pattern, combine='by_coords')
        ds = ds.sel(time=slice(f"{start_year}-01", f"{end_year}-12"))
        rain = ds['RAIN'] + ds['SNOW']
        et = ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']
        runoff = ds['QRUNOFF']
        # convert to mm/day
        rain = (rain * 86400).mean(dim='time')
        et = (et * 86400).mean(dim='time')
        runoff = (runoff * 86400).mean(dim='time')
        return {'P': rain, 'ET': et, 'Q': runoff}
    except Exception as e:
        print("Error loading ELM data:", e)
        return {}

def load_ilamb_mean(variable, dataset, filename, varname, start_year, end_year):
    url = f'https://www.ilamb.org/ILAMB-Data/DATA/{variable}/{dataset}/{filename}'
    try:
        ds = xr.open_dataset(url)
        da = ds[varname]
        # adjust lon to -180:180 if needed
        if np.any(da.lon > 180):
            da = da.assign_coords(lon=((da.lon + 180) % 360 - 180)).sortby('lon')
        da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        da_mean = da.mean(dim='time')
        return da_mean
    except Exception as e:
        print(f"Error loading ILAMB data {url}:", e)
        return None

def compute_basin_means(da, basins, geojson):
    # da dims: lat, lon
    try:
        gdf = gpd.read_file(geojson)
    except Exception as e:
        print("Error reading geojson:", e)
        return {}
    # get lat/lon arrays
    if 'lat' in da.coords and 'lon' in da.coords:
        lat = da['lat'].values
        lon = da['lon'].values
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lat2d = lat
            lon2d = lon
    else:
        print("DataArray has no lat/lon coords")
        return {}
    results = {}
    for basin in basins:
        sub = gdf[gdf['grdc_no'] == basin]
        if len(sub) == 0:
            print(f"Basin {basin} not found in geojson")
            continue
        poly = sub.geometry.values[0]
        # build mask
        mask = np.zeros(lat2d.shape, dtype=bool)
        for i in range(lat2d.shape[0]):
            for j in range(lon2d.shape[1]):
                mask[i,j] = poly.contains(Point(lon2d[i,j], lat2d[i,j]))
        # area weights
        weights = np.cos(np.deg2rad(lat2d))
        w = weights * mask
        vals = da.values
        masked = np.where(mask, vals, np.nan)
        num = np.nansum(masked * weights)
        den = np.nansum(w)
        mean = num/den if den != 0 else np.nan
        results[basin] = mean
    return results

def load_mosart_grid(case_name, data_dir):
    path = os.path.join(data_dir, 'e3sm', 'rof', f"{case_name}.mosart.h0.*.nc")
    try:
        ds = xr.open_mfdataset(path, combine='by_coords')
        lat = ds['lat'].values
        lon = ds['lon'].values
        return lat, lon
    except Exception as e:
        print("Error loading MOSART grid:", e)
        return None, None

def find_nearest_grid_point(lat2d, lon2d, plat, plon):
    # brute-force
    flat_lat = lat2d.flatten()
    flat_lon = lon2d.flatten()
    # haversine or simple squared difference
    dist = (flat_lat - plat)**2 + (flat_lon - plon)**2
    idx = np.argmin(dist)
    i, j = np.unravel_index(idx, lat2d.shape)
    return i, j

def load_daily_mosart(case_name, data_dir, start_date, end_date):
    pattern = os.path.join(data_dir, 'e3sm', 'rof', f"{case_name}.mosart.h1.*-*-*-00000.nc")
    try:
        ds = xr.open_mfdataset(pattern, combine='by_coords')
        ds = ds.sel(time=slice(str(start_date), str(end_date)))
        return ds
    except Exception as e:
        print("Error loading daily MOSART:", e)
        return None

def compute_fdc_metrics(obs, sim):
    # align times
    df = pd.DataFrame({'obs': obs, 'sim': sim})
    df = df.dropna()
    if len(df) == 0:
        return np.nan, np.nan
    vol_bias = (df['sim'].sum() - df['obs'].sum()) / df['obs'].sum() if df['obs'].sum() != 0 else np.nan
    try:
        wdist = wasserstein_distance(df['obs'], df['sim'])
    except Exception:
        wdist = np.nan
    return vol_bias, wdist

def plot_bar_comparison(basins, model_vals, obs_vals, varname, outfile):
    try:
        x = np.arange(len(basins))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(x - width/2, [model_vals[b] for b in basins], width, label='Model')
        ax.bar(x + width/2, [obs_vals[b] for b in basins], width, label='Obs')
        ax.set_xticks(x)
        ax.set_xticklabels(basins)
        ax.set_title(f"{varname} model vs obs")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting bar chart {varname}:", e)

def plot_radar(df, basin, outfile):
    try:
        metrics = df.columns.tolist()
        values = df.loc[basin].values
        N = len(metrics)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles = angles + [angles[0]]
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_title(f"Diagnostics Radar: {basin}")
        plt.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting radar for {basin}:", e)

def main():
    # settings
    case_name = "sample.v3.LR.historical"
    data_dir = "./data/sample"
    geojson = os.path.join(data_dir, "obs", "basin_polygons.geojson")
    gauge_meta = os.path.join(data_dir, "obs", "gauge_metadata.csv")
    gauge_obs_dir = os.path.join(data_dir, "obs", "streamflow")
    basins = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
    start_year, end_year = 1985, 1989
    start_date = datetime.date(1985,1,1)
    end_date = datetime.date(1989,12,31)
    out_dir = "./output/task07_integrated_diagnostic"
    make_output_dir(out_dir)

    # Part 1
    elm = load_elm_means(case_name, data_dir, start_year, end_year)
    # Part 2
    pr_obs = load_ilamb_mean('pr', 'GPCCv2018', 'pr.nc', 'pr', start_year, end_year)
    et_obs = load_ilamb_mean('evspsbl', 'MODIS', 'et_0.5x0.5.nc', 'et', start_year, end_year)
    q_obs = load_ilamb_mean('mrro', 'LORA', 'LORA.nc', 'mrro', start_year, end_year)

    # Part 3: basin means
    model_P = compute_basin_means(elm['P'], basins, geojson)
    model_ET = compute_basin_means(elm['ET'], basins, geojson)
    model_Q = compute_basin_means(elm['Q'], basins, geojson)
    obs_P = compute_basin_means(pr_obs, basins, geojson)
    obs_ET = compute_basin_means(et_obs, basins, geojson)
    obs_Q = compute_basin_means(q_obs, basins, geojson)

    # Part 4: streamflow FDC
    lat_m, lon_m = load_mosart_grid(case_name, data_dir)
    if lat_m is None:
        return
    if lat_m.ndim == 1 and lon_m.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_m, lat_m)
    else:
        lat2d, lon2d = lat_m, lon_m
    ds_daily = load_daily_mosart(case_name, data_dir, start_date, end_date)
    try:
        df_meta = pd.read_csv(gauge_meta, dtype={'gauge_id':str})
    except Exception as e:
        print("Error reading gauge metadata:", e)
        df_meta = pd.DataFrame()
    sf_vol_bias = {}
    sf_wdist = {}
    for basin in basins:
        sub = df_meta[df_meta['gauge_id'] == str(basin)]
        if sub.empty:
            sf_vol_bias[basin] = np.nan
            sf_wdist[basin] = np.nan
            continue
        row = sub.iloc[0]
        gid = row['gauge_id']
        plat, plon = row['lat'], row['lon']
        # load observed
        obs_file = os.path.join(gauge_obs_dir, f"{gid}.csv")
        try:
            df = pd.read_csv(obs_file, parse_dates=['date'])
            df = df.set_index('date').loc[str(start_date):str(end_date)]
            obs_q = df['discharge_m3s']
        except Exception as e:
            print(f"Error reading obs streamflow for {gid}:", e)
            sf_vol_bias[basin] = np.nan
            sf_wdist[basin] = np.nan
            continue
        # find grid point
        i, j = find_nearest_grid_point(lat2d, lon2d, plat, plon)
        try:
            sim = ds_daily['RIVER_DISCHARGE_OVER_LAND_LIQ'].isel(nrow=i, ncol=j).to_series()
            sim = sim.loc[str(start_date):str(end_date)]
        except Exception as e:
            print(f"Error extracting sim streamflow for {gid}:", e)
            sf_vol_bias[basin] = np.nan
            sf_wdist[basin] = np.nan
            continue
        vol_bias, wdist = compute_fdc_metrics(obs_q, sim)
        sf_vol_bias[basin] = vol_bias
        sf_wdist[basin] = wdist

    # Part 5: combine and visualize
    records = []
    for basin in basins:
        rec = {
            'basin': basin,
            'P_model': model_P.get(basin, np.nan),
            'P_obs': obs_P.get(basin, np.nan),
            'P_bias': model_P.get(basin, np.nan) - obs_P.get(basin, np.nan),
            'ET_model': model_ET.get(basin, np.nan),
            'ET_obs': obs_ET.get(basin, np.nan),
            'ET_bias': model_ET.get(basin, np.nan) - obs_ET.get(basin, np.nan),
            'Q_model': model_Q.get(basin, np.nan),
            'Q_obs': obs_Q.get(basin, np.nan),
            'Q_bias': model_Q.get(basin, np.nan) - obs_Q.get(basin, np.nan),
            'water_balance_residual': model_P.get(basin, np.nan) - model_ET.get(basin, np.nan) - model_Q.get(basin, np.nan),
            'streamflow_vol_bias': sf_vol_bias.get(basin, np.nan),
            'streamflow_wasserstein': sf_wdist.get(basin, np.nan)
        }
        records.append(rec)
    df_summary = pd.DataFrame(records).set_index('basin')
    # save summary
    summary_csv = os.path.join(out_dir, "basin_summary_metrics.csv")
    try:
        df_summary.to_csv(summary_csv)
    except Exception as e:
        print("Error saving summary CSV:", e)
    # bar charts for P, ET, Q
    plot_bar_comparison(basins, model_P, obs_P, "Precipitation", os.path.join(out_dir, "bar_P.png"))
    plot_bar_comparison(basins, model_ET, obs_ET, "Evapotranspiration", os.path.join(out_dir, "bar_ET.png"))
    plot_bar_comparison(basins, model_Q, obs_Q, "Runoff", os.path.join(out_dir, "bar_Q.png"))
    # radar charts per basin
    metrics = ['P_bias','ET_bias','Q_bias','streamflow_vol_bias','water_balance_residual','streamflow_wasserstein']
    df_metrics = df_summary[metrics]
    for basin in basins:
        outfile = os.path.join(out_dir, f"radar_{basin}.png")
        plot_radar(df_metrics, basin, outfile)

if __name__ == "__main__":
    main()