import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import wasserstein_distance
import tempfile
import requests

def ensure_lon(ds, lon_name='lon'):
    lon = ds[lon_name]
    if lon.max() > 180:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon})
        ds = ds.sortby(lon_name)
    return ds

def load_dataset_from_url(url):
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.flush()
        tmp.close()
        ds = xr.open_dataset(tmp.name)
        os.unlink(tmp.name)
        return ds
    except Exception as e:
        raise

def compute_basin_masks(lon, lat, basins):
    masks = {}
    for idx, row in basins.iterrows():
        basin_id = row['grdc_no']
        poly = row['geometry']
        if not poly.is_valid:
            poly = poly.buffer(0)
        mask = np.zeros((lat.size, lon.size), dtype=bool)
        for i in range(lat.size):
            for j in range(lon.size):
                try:
                    if poly.covers(Point(lon[j], lat[i])):
                        mask[i, j] = True
                except Exception:
                    pass
        masks[basin_id] = mask
    return masks

def main():
    # Settings
    elm_dir = './data/sample/e3sm/lnd'
    mosart_dir = './data/sample/e3sm/rof'
    case = 'sample.v3.LR.historical'
    years = slice('1985-01', '1989-12')
    basin_ids = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
    basins_file = './data/sample/obs/basin_polygons.geojson'
    gauge_meta_file = './data/sample/obs/gauge_metadata.csv'
    obs_streamflow_dir = './data/sample/obs/streamflow'
    output_dir = '/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_07_integrated_diagnostic/run4_debug/v1/output'
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: Load model ELM fields
    try:
        elm_pattern = os.path.join(elm_dir, f'{case}.elm.h0.*.nc')
        ds_elm = xr.open_mfdataset(elm_pattern, combine='by_coords')
        ds_elm = ds_elm.sel(time=years)
        precip = (ds_elm['RAIN'] + ds_elm['SNOW']) * 86400.0
        precip_mean = precip.mean(dim='time')
        et = (ds_elm['QVEGT'] + ds_elm['QVEGE'] + ds_elm['QSOIL']) * 86400.0
        et_mean = et.mean(dim='time')
        runoff = ds_elm['QRUNOFF'] * 86400.0
        runoff_mean = runoff.mean(dim='time')
        model_fields = {'P': precip_mean, 'ET': et_mean, 'Q': runoff_mean}
    except Exception as e:
        warnings.warn(f"Failed to load ELM data: {e}")
        return

    # Part 2: Load observation fields from ILAMB
    obs_fields = {}
    try:
        pr_url = 'https://www.ilamb.org/ILAMB-Data/DATA/pr/GPCCv2018/pr.nc'
        ds_pr = load_dataset_from_url(pr_url)
        ds_pr = ensure_lon(ds_pr, 'lon')
        pr_mean = ds_pr['pr'].sel(time=years).mean(dim='time')
        obs_fields['P'] = pr_mean
    except Exception as e:
        warnings.warn(f"Failed to load GPCC precipitation: {e}")
    try:
        et_url = 'https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc'
        ds_et_obs = load_dataset_from_url(et_url)
        ds_et_obs = ensure_lon(ds_et_obs, 'lon')
        et_mean = ds_et_obs['et'].sel(time=years).mean(dim='time')
        obs_fields['ET'] = et_mean
    except Exception as e:
        warnings.warn(f"Failed to load MODIS ET: {e}")
    try:
        qobs_url = 'https://www.ilamb.org/ILAMB-Data/DATA/mrro/LORA/LORA.nc'
        ds_qobs = load_dataset_from_url(qobs_url)
        ds_qobs = ensure_lon(ds_qobs, 'lon')
        qobs_mean = ds_qobs['mrro'].sel(time=years).mean(dim='time')
        obs_fields['Q'] = qobs_mean
    except Exception as e:
        warnings.warn(f"Failed to load LORA runoff: {e}")

    # Part 3: Basin means
    basins = gpd.read_file(basins_file)
    basins = basins[basins['grdc_no'].isin(basin_ids)].reset_index(drop=True)
    basins['geometry'] = basins['geometry'].buffer(0)
    mlon = model_fields['P']['lon'].values
    mlat = model_fields['P']['lat'].values
    model_masks = compute_basin_masks(mlon, mlat, basins)
    olon = obs_fields['P']['lon'].values if 'P' in obs_fields else None
    olat = obs_fields['P']['lat'].values if 'P' in obs_fields else None
    obs_masks = compute_basin_masks(olon, olat, basins) if olon is not None else {}

    model_basin = {}
    obs_basin = {}
    for bid in basin_ids:
        model_basin[bid] = {}
        obs_basin[bid] = {}
        mask_m = model_masks.get(bid, None)
        mask_o = obs_masks.get(bid, None)
        for var in ['P', 'ET', 'Q']:
            try:
                val_m = np.nanmean(model_fields[var].values[mask_m])
            except Exception:
                val_m = np.nan
            try:
                val_o = np.nanmean(obs_fields[var].values[mask_o])
            except Exception:
                val_o = np.nan
            model_basin[bid][var] = val_m
            obs_basin[bid][var] = val_o

    # Part 4: Streamflow FDC metrics
    try:
        mosart_pattern = os.path.join(mosart_dir, f'{case}.mosart.h1.*-00000.nc')
        ds_mos = xr.open_mfdataset(mosart_pattern, combine='by_coords')
        ds_mos = ds_mos.sel(time=slice('1985-01-01', '1989-12-31'))
        ds_mos = ensure_lon(ds_mos, 'lon')
    except Exception as e:
        warnings.warn(f"Failed to load MOSART data: {e}")
        ds_mos = None

    gmeta = pd.read_csv(gauge_meta_file, dtype={'gauge_id': str})
    gdf = gpd.GeoDataFrame(gmeta, geometry=gpd.points_from_xy(gmeta.lon, gmeta.lat), crs="EPSG:4326")
    try:
        gdf = gpd.sjoin(gdf, basins[['grdc_no', 'geometry']], how='inner', predicate='within')
    except Exception:
        gdf = gpd.sjoin(gdf, basins[['grdc_no', 'geometry']], how='inner', op='within')
    sf_metrics = {}
    if ds_mos is not None:
        mos_lons = ds_mos['lon'].values
        mos_lats = ds_mos['lat'].values
        for idx, row in gdf.iterrows():
            gid = row['gauge_id']
            bid = row['grdc_no']
            lon0, lat0 = row['lon'], row['lat']
            j = np.argmin(np.abs(mos_lons - lon0))
            i = np.argmin(np.abs(mos_lats - lat0))
            try:
                sim = ds_mos['RIVER_DISCHARGE_OVER_LAND_LIQ'].sel(lat=mos_lats[i], lon=mos_lons[j], method='nearest').values
                obs_df = pd.read_csv(os.path.join(obs_streamflow_dir, f"{gid}.csv"), parse_dates=['date'])
                obs_df = obs_df.set_index('date').sort_index()
                obs = obs_df.loc['1985-01-01':'1989-12-31', 'discharge_m3s'].values
                minlen = min(len(sim), len(obs))
                sim = sim[:minlen]
                obs = obs[:minlen]
                vol_bias = (np.nansum(sim) - np.nansum(obs)) / np.nansum(obs) if np.nansum(obs)!=0 else np.nan
                sim_sorted = np.sort(sim[~np.isnan(sim)])
                obs_sorted = np.sort(obs[~np.isnan(obs)])
                wass = wasserstein_distance(sim_sorted, obs_sorted)
                sf_metrics.setdefault(bid, []).append({'vol_bias': vol_bias, 'wass': wass})
            except Exception as e:
                warnings.warn(f"Streamflow metric failed for gauge {gid}: {e}")
    basin_sf = {}
    for bid, lst in sf_metrics.items():
        if len(lst)>0:
            vb = np.nanmean([x['vol_bias'] for x in lst])
            wass = np.nanmean([x['wass'] for x in lst])
        else:
            vb, wass = np.nan, np.nan
        basin_sf[bid] = {'sf_vol_bias': vb, 'sf_wass': wass}

    # Part 5: Summarize and visualize
    rows = []
    for bid in basin_ids:
        mb = model_basin.get(bid, {})
        ob = obs_basin.get(bid, {})
        sf = basin_sf.get(bid, {'sf_vol_bias': np.nan, 'sf_wass': np.nan})
        Pm, Pe = mb.get('P', np.nan), ob.get('P', np.nan)
        ETm, ETe = mb.get('ET', np.nan), ob.get('ET', np.nan)
        Qm, Qe = mb.get('Q', np.nan), ob.get('Q', np.nan)
        P_bias = (Pm - Pe) / Pe * 100 if Pe != 0 else np.nan
        ET_bias = (ETm - ETe) / ETe * 100 if ETe != 0 else np.nan
        Q_bias = (Qm - Qe) / Qe * 100 if Qe != 0 else np.nan
        wbal = Pm - ETm - Qm
        rows.append({
            'basin_id': bid,
            'P_obs': Pe, 'P_mod': Pm, 'P_bias_%': P_bias,
            'ET_obs': ETe, 'ET_mod': ETm, 'ET_bias_%': ET_bias,
            'Q_obs': Qe, 'Q_mod': Qm, 'Q_bias_%': Q_bias,
            'water_balance_resid': wbal,
            'sf_vol_bias_%': sf['sf_vol_bias']*100 if sf['sf_vol_bias'] is not None else np.nan,
            'sf_wasserstein': sf['sf_wass']
        })
    df = pd.DataFrame(rows).set_index('basin_id')
    df.to_csv(os.path.join(output_dir, 'basin_summary_metrics.csv'))

    try:
        bas_names = [str(b) for b in basin_ids]
        x = np.arange(len(basin_ids))
        width = 0.35
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for i, var in enumerate(['P', 'ET', 'Q']):
            ax = axs[i]
            obs_v = df[f'{var}_obs']
            mod_v = df[f'{var}_mod']
            ax.bar(x - width/2, obs_v, width, label='Obs')
            ax.bar(x + width/2, mod_v, width, label='Mod')
            ax.set_xticks(x)
            ax.set_xticklabels(bas_names, rotation=45)
            ax.set_title(f'{var} basin mean')
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'bar_chart_P_ET_Q.png'))
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Failed to create bar charts: {e}")

    try:
        metrics = ['P_bias_%', 'ET_bias_%', 'Q_bias_%', 'water_balance_resid', 'sf_vol_bias_%', 'sf_wasserstein']
        df_norm = df.copy()
        for m in metrics:
            vals = df[m].abs()
            vmax = vals.max()
            df_norm[m] = vals / vmax if vmax > 0 else 0
        N = len(metrics)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(12, 8))
        for idx, bid in enumerate(basin_ids):
            ax = fig.add_subplot(2, 3, idx+1, polar=True)
            vals = df_norm.loc[bid, metrics].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', linewidth=2)
            ax.fill(angles, vals, alpha=0.25)
            ax.set_title(str(bid))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=8)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'radar_chart_metrics.png'))
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Failed to create radar chart: {e}")

if __name__ == "__main__":
    main()