import os
import glob
import json
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.stats import wasserstein_distance

try:
    import cftime
except Exception:
    cftime = None

# ----------------------------
# Configuration and utilities
# ----------------------------

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")

# Output directory as requested by user
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_07_integrated_diagnostic/run1_debug/v1/output"
ensure_dir(output_dir)
fig_dir = os.path.join(output_dir, "figs")
ensure_dir(fig_dir)
data_cache_dir = os.path.join(output_dir, "ilamb_data")
ensure_dir(data_cache_dir)

# Data paths
base_e3sm = "./data/sample/e3sm"
base_obs_streamflow = "./data/sample/obs/streamflow"
gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
basin_geojson_path = "./data/sample/obs/basin_polygons.geojson"

case_name = "sample.v3.LR.historical"

# Period
start_date = "1985-01-01"
end_date   = "1989-12-31"
start_year = 1985
end_year   = 1989

# Basins of interest: name and GRDC number (gauge_id)
basins = [
    {"name": "Amazon",   "gauge_id": "3629000"},
    {"name": "Missouri", "gauge_id": "4121801"},
    {"name": "Columbia", "gauge_id": "4115200"},
    {"name": "Danube",   "gauge_id": "6742900"},
    {"name": "Mekong",   "gauge_id": "2969100"},
    {"name": "Orange",   "gauge_id": "1159100"},
]

# ----------------------------
# Helper functions
# ----------------------------

def to_mm_per_day(x, units_hint=None):
    return x * 86400.0

def standardize_lon(lon):
    lon_mod = lon.copy()
    lon_mod = ((lon_mod + 180) % 360) - 180
    return lon_mod

def get_latlon_2d(ds):
    if 'lat' in ds and 'lon' in ds:
        lat = ds['lat']
        lon = ds['lon']
    else:
        lat = ds.coords.get('lat', None)
        lon = ds.coords.get('lon', None)
        if lat is None or lon is None:
            raise ValueError("Dataset does not contain 'lat' and 'lon' variables or coordinates.")
    lat_vals = lat.values
    lon_vals = lon.values
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        lat2d, lon2d = lat_vals, lon_vals
    else:
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    lon2d = standardize_lon(lon2d)
    return lat2d, lon2d

def load_geojson_polygons(geojson_path):
    try:
        with open(geojson_path, 'r') as f:
            gj = json.load(f)
    except Exception as e:
        print(f"Failed to load GeoJSON {geojson_path}: {e}")
        return {}

    basins_map = {}
    features = gj.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        gauge_id = str(props.get("grdc_no", "")).strip()
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        polygons = []
        if gtype == "Polygon":
            if len(coords) > 0:
                ext = coords[0]
                poly = [(float(x[0]), float(x[1])) for x in ext]
                polygons.append(poly)
        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                if len(poly_coords) > 0:
                    ext = poly_coords[0]
                    poly = [(float(x[0]), float(x[1])) for x in ext]
                    polygons.append(poly)
        else:
            continue
        if gauge_id:
            basins_map[gauge_id] = polygons
    return basins_map

def mask_grid_by_polygons(lat2d, lon2d, polygons):
    ny, nx = lat2d.shape
    points = np.vstack([lon2d.ravel(), lat2d.ravel()]).T
    mask = np.zeros(points.shape[0], dtype=bool)
    for poly in polygons:
        try:
            path = MplPath(poly)
            mask = mask | path.contains_points(points)
        except Exception as e:
            print(f"Polygon masking error: {e}")
            continue
    return mask.reshape(ny, nx)

def area_weighted_mean(field2d, lat2d, mask=None):
    weights = np.cos(np.deg2rad(lat2d))
    valid = np.isfinite(field2d)
    m = np.ones_like(valid, dtype=bool)
    if mask is not None:
        m = m & mask
    m = m & valid
    if not np.any(m):
        return np.nan
    w = np.where(m, weights, 0.0)
    v = np.where(m, field2d, 0.0)
    num = np.sum(v * w)
    den = np.sum(w)
    if den == 0:
        return np.nan
    return num / den

def download_file(url, dest_path, retries=2):
    try:
        if os.path.exists(dest_path):
            print(f"Using cached file {dest_path}")
            return dest_path
        attempt = 0
        while attempt <= retries:
            try:
                print(f"Downloading {url} -> {dest_path}")
                urllib.request.urlretrieve(url, dest_path)
                return dest_path
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    raise e
        return None
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def subset_time_mean(da, start_date=None, end_date=None):
    try:
        if 'time' in da.dims:
            if start_date and end_date:
                da = da.sel(time=slice(start_date, end_date))
            return da.mean(dim='time', skipna=True)
        else:
            return da
    except Exception as e:
        print(f"Time subset/mean failed: {e}")
        return da

def find_nearest_gridpoint(lat2d, lon2d, lat_pt, lon_pt):
    lon_pt = ((lon_pt + 180) % 360) - 180
    d2 = (lat2d - lat_pt)**2 + (lon2d - lon_pt)**2
    idx = np.argmin(d2)
    iy, ix = np.unravel_index(idx, lat2d.shape)
    return iy, ix

def to_datetime_index_from_timecoord(time_coord):
    try:
        return pd.to_datetime(time_coord.values)
    except Exception:
        pass
    # Try cftime objects with year,month,day
    try:
        vals = time_coord.values
        dates = []
        for t in vals:
            y = int(getattr(t, 'year'))
            m = int(getattr(t, 'month'))
            d = int(getattr(t, 'day'))
            dates.append(datetime(y, m, d))
        return pd.DatetimeIndex(dates)
    except Exception:
        pass
    # Try num2date with units
    if cftime is not None:
        try:
            vals = time_coord.values
            units = time_coord.attrs.get('units', None)
            calendar = time_coord.attrs.get('calendar', 'standard')
            if units is not None:
                dts = cftime.num2date(vals, units, calendar=calendar, only_use_cftime_datetimes=True)
                dates = [datetime(int(t.year), int(t.month), int(t.day)) for t in dts]
                return pd.DatetimeIndex(dates)
        except Exception:
            pass
    # Fallback empty index
    return pd.DatetimeIndex([], dtype='datetime64[ns]')

# ----------------------------
# Part 1 — Extract model fields (ELM monthly, 1985-1989)
# ----------------------------

print("Part 1: Loading ELM monthly fields...")

elm_files = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        fn = os.path.join(base_e3sm, "lnd", f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fn):
            elm_files.append(fn)

if len(elm_files) == 0:
    print("No ELM files found for the period. Exiting.")
    try:
        pd.DataFrame().to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    except Exception as e:
        print(f"Failed to write empty summary: {e}")
    raise SystemExit(0)

try:
    elm_ds = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
except Exception as e:
    print(f"Failed to open ELM datasets: {e}")
    raise

try:
    P_rate = elm_ds['RAIN'] + elm_ds['SNOW']
    ET_rate = elm_ds['QVEGE'] + elm_ds['QVEGT'] + elm_ds['QSOIL']
    Q_rate = elm_ds['QRUNOFF']
except Exception as e:
    print(f"Failed to compute ELM variables: {e}")
    raise

P_mmday_tm = subset_time_mean(to_mm_per_day(P_rate), start_date, end_date)
ET_mmday_tm = subset_time_mean(to_mm_per_day(ET_rate), start_date, end_date)
Q_mmday_tm = subset_time_mean(to_mm_per_day(Q_rate), start_date, end_date)

elm_lat2d, elm_lon2d = get_latlon_2d(elm_ds)

# ----------------------------
# Part 2 — Fetch and extract observation fields (ILAMB)
# ----------------------------

print("Part 2: Downloading and loading ILAMB observational datasets...")

# GPCC precipitation
gpcc_url = "https://www.ilamb.org/ILAMB-Data/DATA/pr/GPCCv2018/pr.nc"
gpcc_path = os.path.join(data_cache_dir, "GPCC_pr.nc")
download_file(gpcc_url, gpcc_path)

# MODIS ET
modis_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
modis_path = os.path.join(data_cache_dir, "MODIS_et.nc")
download_file(modis_url, modis_path)

# LORA runoff
lora_url = "https://www.ilamb.org/ILAMB-Data/DATA/mrro/LORA/LORA.nc"
lora_path = os.path.join(data_cache_dir, "LORA_mrro.nc")
download_file(lora_url, lora_path)

obs_P = None
obs_ET = None
obs_Q = None
obs_latlon = {}

# GPCC precipitation (variable: pr, mm/day)
try:
    if os.path.exists(gpcc_path):
        with xr.open_dataset(gpcc_path) as ds_gpcc:
            varname = 'pr' if 'pr' in ds_gpcc else list(ds_gpcc.data_vars.keys())[0]
            da = ds_gpcc[varname]
            if 'lon' in ds_gpcc:
                da = da.assign_coords(lon=standardize_lon(ds_gpcc['lon'].values))
            obs_P = subset_time_mean(da, start_date, end_date)
            gpcc_lat2d, gpcc_lon2d = get_latlon_2d(ds_gpcc.assign_coords(lon=standardize_lon(ds_gpcc['lon'])))
            obs_latlon['P'] = (gpcc_lat2d, gpcc_lon2d)
    else:
        print("GPCC file not available; skipping P observations.")
except Exception as e:
    print(f"Failed to load GPCC precipitation: {e}")

# MODIS ET (variable 'et' inside file, mm/day)
try:
    if os.path.exists(modis_path):
        with xr.open_dataset(modis_path) as ds_modis:
            varname = 'et' if 'et' in ds_modis else list(ds_modis.data_vars.keys())[0]
            da = ds_modis[varname]
            if 'lon' in ds_modis:
                da = da.assign_coords(lon=standardize_lon(ds_modis['lon'].values))
            obs_ET = subset_time_mean(da, start_date, end_date)
            modis_lat2d, modis_lon2d = get_latlon_2d(ds_modis.assign_coords(lon=standardize_lon(ds_modis['lon'])))
            obs_latlon['ET'] = (modis_lat2d, modis_lon2d)
    else:
        print("MODIS ET file not available; skipping ET observations.")
except Exception as e:
    print(f"Failed to load MODIS ET: {e}")

# LORA runoff (variable 'mrro', mm/day)
try:
    if os.path.exists(lora_path):
        with xr.open_dataset(lora_path) as ds_lora:
            varname = 'mrro' if 'mrro' in ds_lora else list(ds_lora.data_vars.keys())[0]
            da = ds_lora[varname]
            if 'lon' in ds_lora:
                da = da.assign_coords(lon=standardize_lon(ds_lora['lon'].values))
            obs_Q = subset_time_mean(da, start_date, end_date)
            lora_lat2d, lora_lon2d = get_latlon_2d(ds_lora.assign_coords(lon=standardize_lon(ds_lora['lon'])))
            obs_latlon['Q'] = (lora_lat2d, lora_lon2d)
    else:
        print("LORA runoff file not available; skipping runoff observations.")
except Exception as e:
    print(f"Failed to load LORA runoff: {e}")

# ----------------------------
# Part 3 — Clip to basin means
# ----------------------------

print("Part 3: Computing basin-averaged P, ET, Q for model and observations...")

basin_polygons_map = load_geojson_polygons(basin_geojson_path)

per_basin_stats = []

for b in basins:
    name = b["name"]
    gid = str(b["gauge_id"])
    print(f"Processing basin {name} ({gid})")
    polygons = basin_polygons_map.get(gid, None)
    if polygons is None:
        print(f"No polygons found for basin {name} ({gid}); skipping.")
        continue

    try:
        P_model_2d = P_mmday_tm.values
        ET_model_2d = ET_mmday_tm.values
        Q_model_2d = Q_mmday_tm.values
        lat2d_m, lon2d_m = elm_lat2d, elm_lon2d
        mask_m = mask_grid_by_polygons(lat2d_m, lon2d_m, polygons)
        Pm_mean = area_weighted_mean(P_model_2d, lat2d_m, mask=mask_m)
        ETm_mean = area_weighted_mean(ET_model_2d, lat2d_m, mask=mask_m)
        Qm_mean = area_weighted_mean(Q_model_2d, lat2d_m, mask=mask_m)
    except Exception as e:
        print(f"Failed model basin averaging for {name}: {e}")
        Pm_mean = np.nan
        ETm_mean = np.nan
        Qm_mean = np.nan

    try:
        if obs_P is not None:
            da = obs_P
            lat2d, lon2d = obs_latlon.get('P', get_latlon_2d(obs_P.to_dataset(name="P")))
            field2d = np.array(da.values)
            field2d = field2d if field2d.ndim == 2 else np.squeeze(field2d)
            mask = mask_grid_by_polygons(lat2d, lon2d, polygons)
            Po_mean = area_weighted_mean(field2d, lat2d, mask=mask)
        else:
            Po_mean = np.nan
    except Exception as e:
        print(f"Failed P obs basin averaging for {name}: {e}")
        Po_mean = np.nan

    try:
        if obs_ET is not None:
            da = obs_ET
            lat2d, lon2d = obs_latlon.get('ET', get_latlon_2d(obs_ET.to_dataset(name="ET")))
            field2d = np.array(da.values)
            field2d = field2d if field2d.ndim == 2 else np.squeeze(field2d)
            mask = mask_grid_by_polygons(lat2d, lon2d, polygons)
            ETo_mean = area_weighted_mean(field2d, lat2d, mask=mask)
        else:
            ETo_mean = np.nan
    except Exception as e:
        print(f"Failed ET obs basin averaging for {name}: {e}")
        ETo_mean = np.nan

    try:
        if obs_Q is not None:
            da = obs_Q
            lat2d, lon2d = obs_latlon.get('Q', get_latlon_2d(obs_Q.to_dataset(name="Q")))
            field2d = np.array(da.values)
            field2d = field2d if field2d.ndim == 2 else np.squeeze(field2d)
            mask = mask_grid_by_polygons(lat2d, lon2d, polygons)
            Qo_mean = area_weighted_mean(field2d, lat2d, mask=mask)
        else:
            Qo_mean = np.nan
    except Exception as e:
        print(f"Failed Q obs basin averaging for {name}: {e}")
        Qo_mean = np.nan

    per_basin_stats.append({
        "basin": name,
        "gauge_id": gid,
        "P_model_mmday": Pm_mean,
        "ET_model_mmday": ETm_mean,
        "Q_model_mmday": Qm_mean,
        "P_obs_mmday": Po_mean,
        "ET_obs_mmday": ETo_mean,
        "Q_obs_mmday": Qo_mean,
    })

# ----------------------------
# Part 4 — Streamflow FDC metrics
# ----------------------------

print("Part 4: Computing streamflow FDC metrics from MOSART and gauge observations...")

try:
    gauge_meta = pd.read_csv(gauge_meta_path, dtype={"gauge_id": str})
except Exception as e:
    print(f"Failed to load gauge metadata: {e}")
    gauge_meta = pd.DataFrame(columns=["gauge_id","lat","lon","area_km2","river_name"])

mosart_sample_file = os.path.join(base_e3sm, "rof", f"{case_name}.mosart.h0.{start_year:04d}-01.nc")
if not os.path.exists(mosart_sample_file):
    sample_glob = glob.glob(os.path.join(base_e3sm, "rof", f"{case_name}.mosart.h0.{start_year:04d}-*.nc"))
    if len(sample_glob) > 0:
        mosart_sample_file = sample_glob[0]
    else:
        sample_glob = glob.glob(os.path.join(base_e3sm, "rof", f"{case_name}.mosart.h1.{start_year:04d}-*-*-00000.nc"))
        if len(sample_glob) > 0:
            mosart_sample_file = sample_glob[0]

try:
    with xr.open_dataset(mosart_sample_file) as ds_m:
        mos_lat2d, mos_lon2d = get_latlon_2d(ds_m)
except Exception as e:
    print(f"Failed to open MOSART sample file or read lat/lon: {e}")
    mos_lat2d = None
    mos_lon2d = None

def load_mosart_daily_timeseries_for_cell(case_name, iy, ix, start_date, end_date):
    daily_files = []
    for year in range(start_year, end_year + 1):
        pattern = os.path.join(base_e3sm, "rof", f"{case_name}.mosart.h1.{year:04d}-*-*-00000.nc")
        daily_files.extend(sorted(glob.glob(pattern)))
    if len(daily_files) == 0:
        print("No MOSART daily files found.")
        return pd.Series(dtype=float)

    series_list = []
    for f in daily_files:
        try:
            with xr.open_dataset(f, decode_times=True) as ds:
                varname = 'RIVER_DISCHARGE_OVER_LAND_LIQ'
                if varname not in ds.data_vars:
                    varname = list(ds.data_vars.keys())[0]
                da = ds[varname]
                spatial_dims = [d for d in da.dims if d != 'time']
                if len(spatial_dims) == 2:
                    d0, d1 = spatial_dims
                    da_cell = da.isel({d0: iy, d1: ix})
                elif len(spatial_dims) == 1:
                    # Flattened grid; approximate linear index
                    if mos_lat2d is not None:
                        flat_index = np.ravel_multi_index((iy, ix), mos_lat2d.shape)
                    else:
                        flat_index = 0
                    da_cell = da.isel({spatial_dims[0]: flat_index})
                else:
                    da_cell = da
                times_idx = to_datetime_index_from_timecoord(da_cell['time'])
                vals = da_cell.values
                if len(times_idx) != len(vals):
                    # Attempt decode_times=False then convert
                    with xr.open_dataset(f, decode_times=False) as ds2:
                        da2 = ds2[varname]
                        if len(spatial_dims) == 2:
                            d0, d1 = spatial_dims
                            da_cell2 = da2.isel({d0: iy, d1: ix})
                        elif len(spatial_dims) == 1:
                            if mos_lat2d is not None:
                                flat_index = np.ravel_multi_index((iy, ix), mos_lat2d.shape)
                            else:
                                flat_index = 0
                            da_cell2 = da2.isel({spatial_dims[0]: flat_index})
                        else:
                            da_cell2 = da2
                        tcoord = da_cell2['time']
                        if cftime is not None:
                            units = tcoord.attrs.get('units', None)
                            calendar = tcoord.attrs.get('calendar', 'standard')
                            if units is not None:
                                dts = cftime.num2date(tcoord.values, units, calendar=calendar, only_use_cftime_datetimes=True)
                                times_idx = pd.DatetimeIndex([datetime(int(t.year), int(t.month), int(t.day)) for t in dts])
                        vals = da_cell2.values
                if len(times_idx) == len(vals):
                    s = pd.Series(vals, index=times_idx)
                    series_list.append(s)
                else:
                    print(f"Time conversion mismatch in {f}")
        except Exception as e:
            print(f"Failed to process MOSART file {f}: {e}")
            continue

    if len(series_list) == 0:
        return pd.Series(dtype=float)
    s = pd.concat(series_list).sort_index()
    s = s[~s.index.duplicated(keep='first')]
    s = s.loc[(s.index >= pd.to_datetime(start_date)) & (s.index <= pd.to_datetime(end_date))]
    return s

fdc_metrics = {}

for b in basins:
    name = b["name"]
    gid = str(b["gauge_id"])
    gmeta = gauge_meta[gauge_meta['gauge_id'] == gid]
    if gmeta.empty:
        print(f"No gauge metadata for {gid}")
        lat_g, lon_g = np.nan, np.nan
    else:
        lat_g = float(gmeta.iloc[0]['lat'])
        lon_g = float(gmeta.iloc[0]['lon'])

    obs_path = os.path.join(base_obs_streamflow, f"{gid}.csv")
    try:
        obs_df = pd.read_csv(obs_path, parse_dates=['date'])
        obs_df = obs_df.set_index('date').sort_index()
        obs_df = obs_df.loc[(obs_df.index >= pd.to_datetime(start_date)) & (obs_df.index <= pd.to_datetime(end_date))]
        q_obs = obs_df['discharge_m3s'].astype(float)
    except Exception as e:
        print(f"Failed to load obs streamflow for {gid}: {e}")
        q_obs = pd.Series(dtype=float)

    if mos_lat2d is not None and mos_lon2d is not None and np.isfinite(lat_g) and np.isfinite(lon_g):
        iy, ix = find_nearest_gridpoint(mos_lat2d, mos_lon2d, lat_g, lon_g)
        q_sim = load_mosart_daily_timeseries_for_cell(case_name, iy, ix, start_date, end_date)
    else:
        q_sim = pd.Series(dtype=float)

    try:
        idx = q_obs.index.intersection(q_sim.index)
        q_obs_a = q_obs.reindex(idx).dropna()
        q_sim_a = q_sim.reindex(idx).dropna()
        idx2 = q_obs_a.index.intersection(q_sim_a.index)
        q_obs_a = q_obs_a.reindex(idx2)
        q_sim_a = q_sim_a.reindex(idx2)
    except Exception:
        q_obs_a = pd.Series(dtype=float)
    if len(q_obs_a) == 0 or len(q_sim_a) == 0:
        vol_bias = np.nan
        wd = np.nan
    else:
        vol_bias = (q_sim_a.sum() - q_obs_a.sum()) / q_obs_a.sum() if q_obs_a.sum() != 0 else np.nan
        try:
            wd = wasserstein_distance(q_sim_a.values, q_obs_a.values)
        except Exception as e:
            print(f"Wasserstein distance failed for {gid}: {e}")
            wd = np.nan

    fdc_metrics[gid] = {
        "streamflow_volume_bias": vol_bias,
        "streamflow_wasserstein": wd
    }

# ----------------------------
# Part 5 — Combine and visualize
# ----------------------------

print("Part 5: Combining metrics and creating visualizations...")

rows = []
for rec in per_basin_stats:
    gid = rec["gauge_id"]
    def rel_bias(m, o):
        try:
            if np.isfinite(m) and np.isfinite(o) and o != 0:
                return (m - o) / o
            else:
                return np.nan
        except Exception:
            return np.nan

    P_bias = rel_bias(rec["P_model_mmday"], rec["P_obs_mmday"])
    ET_bias = rel_bias(rec["ET_model_mmday"], rec["ET_obs_mmday"])
    Q_bias = rel_bias(rec["Q_model_mmday"], rec["Q_obs_mmday"])

    WB_model = rec["P_model_mmday"] - rec["ET_model_mmday"] - rec["Q_model_mmday"]
    WB_obs = rec["P_obs_mmday"] - rec["ET_obs_mmday"] - rec["Q_obs_mmday"]
    WB_diff = WB_model - WB_obs

    vol_bias = fdc_metrics.get(gid, {}).get("streamflow_volume_bias", np.nan)
    wd = fdc_metrics.get(gid, {}).get("streamflow_wasserstein", np.nan)

    rows.append({
        "basin": rec["basin"],
        "gauge_id": gid,
        "P_model_mmday": rec["P_model_mmday"],
        "P_obs_mmday": rec["P_obs_mmday"],
        "P_bias_rel": P_bias,
        "ET_model_mmday": rec["ET_model_mmday"],
        "ET_obs_mmday": rec["ET_obs_mmday"],
        "ET_bias_rel": ET_bias,
        "Q_model_mmday": rec["Q_model_mmday"],
        "Q_obs_mmday": rec["Q_obs_mmday"],
        "Q_bias_rel": Q_bias,
        "WB_model_mmday": WB_model,
        "WB_obs_mmday": WB_obs,
        "WB_diff_mmday": WB_diff,
        "Streamflow_volume_bias": vol_bias,
        "Streamflow_wasserstein_m3s": wd
    })

summary_df = pd.DataFrame(rows)
summary_csv_path = os.path.join(output_dir, "summary_metrics.csv")
try:
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary metrics to {summary_csv_path}")
except Exception as e:
    print(f"Failed to save summary metrics CSV: {e}")

try:
    n = len(summary_df)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    for i, row in summary_df.iterrows():
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        vars_labels = ['P', 'ET', 'Q']
        model_vals = [row['P_model_mmday'], row['ET_model_mmday'], row['Q_model_mmday']]
        obs_vals = [row['P_obs_mmday'], row['ET_obs_mmday'], row['Q_obs_mmday']]
        x = np.arange(len(vars_labels))
        width = 0.35
        ax.bar(x - width/2, obs_vals, width, label='Obs')
        ax.bar(x + width/2, model_vals, width, label='Model')
        ax.set_xticks(x)
        ax.set_xticklabels(vars_labels)
        ax.set_ylabel("mm/day")
        ax.set_title(f"{row['basin']} ({row['gauge_id']})")
        ax.legend()
    for j in range(i+1, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis('off')
    plt.tight_layout()
    bar_fig_path = os.path.join(fig_dir, "basin_P_ET_Q_model_vs_obs.png")
    plt.savefig(bar_fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart to {bar_fig_path}")
except Exception as e:
    print(f"Failed to create/save bar charts: {e}")

try:
    metrics_names = ["|P bias|", "|ET bias|", "|Q bias|", "|Stream vol bias|", "|WB residual diff|", "Wasserstein (norm)"]
    wd_vals = summary_df["Streamflow_wasserstein_m3s"].abs().replace([np.inf, -np.inf], np.nan).dropna()
    wd_max = wd_vals.max() if len(wd_vals) > 0 else 1.0
    if not np.isfinite(wd_max) or wd_max == 0:
        wd_max = 1.0

    n = len(summary_df)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig = plt.figure(figsize=(4.5*ncols, 4.5*nrows))
    for i, row in summary_df.iterrows():
        ax = plt.subplot(nrows, ncols, i+1, polar=True)
        P_b = abs(row["P_bias_rel"]) if pd.notnull(row["P_bias_rel"]) else np.nan
        ET_b = abs(row["ET_bias_rel"]) if pd.notnull(row["ET_bias_rel"]) else np.nan
        Q_b = abs(row["Q_bias_rel"]) if pd.notnull(row["Q_bias_rel"]) else np.nan
        S_b = abs(row["Streamflow_volume_bias"]) if pd.notnull(row["Streamflow_volume_bias"]) else np.nan
        WB_d = abs(row["WB_diff_mmday"]) if pd.notnull(row["WB_diff_mmday"]) else np.nan
        WD_n = (abs(row["Streamflow_wasserstein_m3s"]) / wd_max) if pd.notnull(row["Streamflow_wasserstein_m3s"]) else np.nan

        values = [P_b, ET_b, Q_b, S_b, WB_d, WD_n]
        values = [0.0 if (v is None or not np.isfinite(v)) else v for v in values]
        theta = np.linspace(0, 2*np.pi, len(values), endpoint=False)
        values = np.array(values)
        values = np.append(values, values[0])
        theta = np.append(theta, theta[0])
        ax.plot(theta, values, '-o', label=row['basin'])
        ax.fill(theta, values, alpha=0.2)
        ax.set_xticks(np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False))
        ax.set_xticklabels(metrics_names, fontsize=8)
        ax.set_title(f"{row['basin']}", y=1.1)
        ax.set_rlabel_position(0)
    plt.tight_layout()
    radar_fig_path = os.path.join(fig_dir, "basin_radar_diagnostics.png")
    plt.savefig(radar_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved radar chart to {radar_fig_path}")
except Exception as e:
    print(f"Failed to create/save radar charts: {e}")

try:
    ds_out = xr.Dataset({
        "P_model_mmday": P_mmday_tm,
        "ET_model_mmday": ET_mmday_tm,
        "Q_model_mmday": Q_mmday_tm
    })
    model_nc_path = os.path.join(output_dir, "ELM_climatology_1985_1989.nc")
    ds_out.to_netcdf(model_nc_path)
    print(f"Saved model climatology to {model_nc_path}")
except Exception as e:
    print(f"Failed to save model climatology NetCDF: {e}")

print("Analysis complete.")