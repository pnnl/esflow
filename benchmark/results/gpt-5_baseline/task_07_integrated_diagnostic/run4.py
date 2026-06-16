#!/usr/bin/env python3
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

# ----------------------------
# Configuration
# ----------------------------
CASE_NAME = "sample.v3.LR.historical"
DATA_BASE = "./data/sample"
E3SM_LND_DIR = os.path.join(DATA_BASE, "e3sm", "lnd")
E3SM_ROF_DIR = os.path.join(DATA_BASE, "e3sm", "rof")
OBS_STREAMFLOW_DIR = os.path.join(DATA_BASE, "obs", "streamflow")
GAUGE_METADATA_CSV = os.path.join(DATA_BASE, "obs", "gauge_metadata.csv")
BASIN_GEOJSON = os.path.join(DATA_BASE, "obs", "basin_polygons.geojson")
ILAMB_BASE_URL = "https://www.ilamb.org/ILAMB-Data/DATA"
ILAMB_CACHE_DIR = "ilamb_cache"

# Period
START_DATE = "1985-01-01"
END_DATE = "1989-12-31"

# Basins: Name -> Gauge ID
BASINS = [
    ("Amazon", "3629000"),
    ("Missouri", "4121801"),
    ("Columbia", "4115200"),
    ("Danube", "6742900"),
    ("Mekong", "2969100"),
    ("Orange", "1159100"),
]

# Output directory (from user)
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_07_integrated_diagnostic/run4_output"

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create directory {path}: {e}")

def normalize_lon(lon, to_range="-180_180"):
    if to_range == "-180_180":
        lon = ((lon + 180) % 360) - 180
    elif to_range == "0_360":
        lon = lon % 360
    return lon

def detect_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "LAT", "nav_lat"]
    lon_candidates = ["lon", "longitude", "LON", "nav_lon"]
    lat_name = None
    lon_name = None
    for nm in ds.variables:
        if nm in lat_candidates:
            lat_name = nm
        if nm in lon_candidates:
            lon_name = nm
    if lat_name is None:
        # Check coords
        for nm in ds.coords:
            if nm in lat_candidates:
                lat_name = nm
    if lon_name is None:
        for nm in ds.coords:
            if nm in lon_candidates:
                lon_name = nm
    if lat_name is None or lon_name is None:
        # Try find by attrs standard_name
        for nm in ds.variables:
            v = ds[nm]
            if hasattr(v, "standard_name"):
                if v.standard_name in ["latitude"]:
                    lat_name = nm
                if v.standard_name in ["longitude"]:
                    lon_name = nm
    return lat_name, lon_name

def get_latlon_grids(ds):
    lat_name, lon_name = detect_lat_lon_names(ds)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not detect latitude/longitude variable names in dataset")
    lat = ds[lat_name]
    lon = ds[lon_name]
    # Normalize longitude to -180..180
    lon_vals = lon.values
    lon_vals = normalize_lon(lon_vals, "-180_180")
    # Assign back to a copy
    try:
        ds = ds.assign_coords({lon_name: (lon.dims, lon_vals)})
    except Exception:
        pass
    # Create 2D grids if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(ds[lon_name].values, ds[lat_name].values)
        return lat2d, lon2d, lat_name, lon_name, ds
    elif lat.ndim == 2 and lon.ndim == 2:
        return lat.values, lon_vals, lat_name, lon_name, ds
    elif lat.ndim == 1 and lon.ndim == 2:
        # Rare case
        lat2d = np.tile(lat.values[:, None], (1, lon.shape[-1]))
        return lat2d, lon_vals, lat_name, lon_name, ds
    elif lat.ndim == 2 and lon.ndim == 1:
        lon2d = np.tile(ds[lon_name].values[None, :], (lat.shape[0], 1))
        return lat.values, lon2d, lat_name, lon_name, ds
    else:
        # Unstructured grid: lat/lon as 1D with ncol dim
        return lat.values, lon_vals, lat_name, lon_name, ds

def days_in_time(da_time):
    # Return array of days per time step (monthly-aware)
    if "time" not in da_time.dims and da_time.name != "time":
        # da_time is DataArray of time
        t = da_time
    else:
        t = da_time
    try:
        return xr.DataArray(t.dt.days_in_month.astype(np.float64), coords={"time": t.values}, dims=("time",))
    except Exception:
        # fallback assume 30 days
        vals = np.full(t.shape[0], 30.0)
        return xr.DataArray(vals, coords={"time": t.values}, dims=("time",))

def convert_rate_to_mmd(da, units_attr=None, time=None):
    # Convert data array to mm/day based on units
    units = units_attr
    try:
        if units is None and hasattr(da, "attrs") and "units" in da.attrs:
            units = da.attrs["units"]
    except Exception:
        pass
    # default fallback:
    if units is None:
        # If time is monthly and values look large, might be mm/month
        # We'll try to infer based on magnitude and available time
        if time is not None and "time" in da.dims:
            # treat as mm/day if magnitude < 50
            return da
        else:
            return da
    units_l = str(units).lower()
    da_out = da
    if ("kg" in units_l and "s-1" in units_l) or ("m s-1" in units_l) or ("mm s-1" in units_l):
        da_out = da * 86400.0
        da_out.attrs["units"] = "mm/day"
    elif "mm/day" in units_l or "mm d-1" in units_l or "mm/day" in units_l or "mm d^-1" in units_l:
        da_out = da
        da_out.attrs["units"] = "mm/day"
    elif "mm/month" in units_l or "mm mon-1" in units_l or (("mm" in units_l) and ("day" not in units_l) and ("s" not in units_l)):
        # Convert to mm/day via days_in_month
        if time is not None and "time" in da.dims:
            dim_axis = da.dims.index("time")
            try:
                dim_days = da["time"].dt.days_in_month
                da_out = da / dim_days
            except Exception:
                # Fallback average 30 days
                da_out = da / 30.0
        else:
            da_out = da
        da_out.attrs["units"] = "mm/day"
    else:
        # Unknown units: leave as is
        pass
    return da_out

def weighted_time_mean(da):
    # Expect dimension "time"
    if "time" not in da.dims:
        return da
    # Convert to mm/day if possible
    da2 = convert_rate_to_mmd(da, time=da["time"] if "time" in da.dims else None)
    w = days_in_time(da2["time"])
    # Align weights
    w = xr.DataArray(w.values, coords={"time": da2["time"]}, dims=("time",))
    num = (da2 * w).sum(dim="time", skipna=True)
    den = w.where(xr.ones_like(da2).isel({d: 0 for d in da2.dims if d != "time"}), drop=False).sum(dim="time")
    # If den is zero due to NaN times, fallback to simple mean
    try:
        out = num / den
    except Exception:
        out = da2.mean(dim="time", skipna=True)
    return out

def polygon_mask(lat2d, lon2d, polygons):
    # polygons: list of list of [lon, lat] coordinates (exterior rings)
    # Generate boolean mask for points inside ANY polygon
    # lat2d/lon2d can be 1D (n,) or 2D (ny,nx)
    lons = lon2d.copy()
    lats = lat2d.copy()
    lons = normalize_lon(lons, "-180_180")
    # Flatten
    pts = np.column_stack((lons.ravel(), lats.ravel()))
    mask_flat = np.zeros(pts.shape[0], dtype=bool)
    for poly in polygons:
        try:
            path = MplPath(np.array(poly))
            inside = path.contains_points(pts)
            mask_flat = mask_flat | inside
        except Exception:
            continue
    mask = mask_flat.reshape(lons.shape)
    return mask

def area_weighted_mean_field(field_da, lat2d):
    # area weights = cos(lat) in radians
    lat_rad = np.deg2rad(lat2d)
    weights = np.cos(lat_rad)
    # Align shapes
    # field_da should be 2D (y,x) or (n,) matching lat2d shape
    arr = field_da.values
    if arr.ndim == 1 and lat2d.ndim == 2:
        # Likely unstructured grid flattened
        pass
    w = weights
    # Handle mask for NaNs
    valid = np.isfinite(arr)
    w_eff = np.where(valid, w, 0.0)
    num = np.nansum(arr * w_eff)
    den = np.nansum(w_eff)
    if den == 0:
        return np.nan
    return num / den

def area_weighted_basin_mean(field_da, lat2d, lon2d, polygons):
    # Generate mask from polygons
    mask = polygon_mask(lat2d, lon2d, polygons)
    # Extract field values on mask
    arr = field_da.values
    if arr.shape != mask.shape:
        # Try to broadcast or handle as unstructured grid
        try:
            # If arr is (ny,nx) and mask is (ny,nx)
            pass
        except Exception:
            raise ValueError("Field array shape does not match mask shape for area averaging")
    field_masked = np.where(mask, arr, np.nan)
    return area_weighted_mean_field(xr.DataArray(field_masked), lat2d)

def download_ilamb(variable, dataset, filename, cache_dir):
    ensure_dir(cache_dir)
    url = f"{ILAMB_BASE_URL}/{variable}/{dataset}/{filename}"
    local_path = os.path.join(cache_dir, f"{variable}_{dataset}_{os.path.basename(filename)}")
    if not os.path.exists(local_path):
        try:
            print(f"Downloading {url} -> {local_path}")
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    else:
        print(f"Using cached ILAMB file: {local_path}")
    return local_path

def open_e3sm_elm_files(case_name, start_date, end_date):
    pattern = os.path.join(E3SM_LND_DIR, f"{case_name}.elm.h0.*.nc")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No ELM files found with pattern {pattern}")
    # Filter by date range in filename
    selected_files = []
    for f in files:
        base = os.path.basename(f)
        # Expect ...YYYY-MM.nc
        try:
            yymm = base.split(".")[-2]
            dt = datetime.strptime(yymm, "%Y-%m")
            if datetime.strptime(start_date, "%Y-%m-%d") <= dt <= datetime.strptime(end_date, "%Y-%m-%d"):
                selected_files.append(f)
        except Exception:
            continue
    if len(selected_files) == 0:
        # Fallback: open all and subset time
        selected_files = files
    ds = xr.open_mfdataset(selected_files, combine="by_coords")
    # Subset time
    if "time" in ds.dims:
        ds = ds.sel(time=slice(start_date, end_date))
    return ds

def compute_elm_climatology(ds):
    # Variables:
    # Precipitation: RAIN + SNOW (kg/m2/s) -> mm/day
    # Evapotranspiration: QVEGE + QVEGT + QSOIL (mm/s) -> mm/day
    # Runoff: QRUNOFF (mm/s) -> mm/day
    needed = []
    for v in ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]:
        if v not in ds.variables:
            print(f"Warning: Variable {v} not in dataset")
        else:
            needed.append(v)
    # Build DataArrays with consistent dims
    def cv(da):
        return convert_rate_to_mmd(da, units_attr=da.attrs.get("units", None), time=da["time"] if "time" in da.dims else None)
    P = None
    if ("RAIN" in ds) or ("SNOW" in ds):
        rain = ds["RAIN"] if "RAIN" in ds else 0
        snow = ds["SNOW"] if "SNOW" in ds else 0
        P = cv(rain) + cv(snow)
    ET = None
    if ("QVEGE" in ds) or ("QVEGT" in ds) or ("QSOIL" in ds):
        qvege = ds["QVEGE"] if "QVEGE" in ds else 0
        qvegt = ds["QVEGT"] if "QVEGT" in ds else 0
        qsoil = ds["QSOIL"] if "QSOIL" in ds else 0
        ET = cv(qvege) + cv(qvegt) + cv(qsoil)
    Q = None
    if "QRUNOFF" in ds:
        Q = cv(ds["QRUNOFF"])
    # Weighted time means
    clim = {}
    if P is not None:
        clim["P_mmd"] = weighted_time_mean(P)
    if ET is not None:
        clim["ET_mmd"] = weighted_time_mean(ET)
    if Q is not None:
        clim["Q_mmd"] = weighted_time_mean(Q)
    return xr.Dataset(clim)

def open_ilamb_dataset(variable, dataset, filename, varname=None):
    cache_path = download_ilamb(variable, dataset, filename, os.path.join(OUTPUT_DIR, ILAMB_CACHE_DIR))
    ds = xr.open_dataset(cache_path)
    if varname is None:
        varname = variable
    if varname not in ds.variables:
        # Sometimes variable nested
        # Try first data var
        if len(ds.data_vars) > 0:
            varname = list(ds.data_vars)[0]
        else:
            raise KeyError(f"Variable {varname} not found in {cache_path}")
    da = ds[varname]
    # Unify lon coords to -180..180
    lat_name, lon_name = detect_lat_lon_names(ds)
    if lon_name:
        try:
            ds = ds.assign_coords({lon_name: (ds[lon_name].dims, normalize_lon(ds[lon_name].values, "-180_180"))})
            da = ds[varname]
        except Exception:
            pass
    return ds, da

def subset_time_and_climatology(da, start_date, end_date):
    da2 = da
    if "time" in da2.dims:
        # Subset time
        try:
            da2 = da2.sel(time=slice(start_date, end_date))
        except Exception:
            pass
        # Convert units to mm/day if needed
        da2 = convert_rate_to_mmd(da2, units_attr=da2.attrs.get("units", None), time=da2["time"])
        da2 = weighted_time_mean(da2)
    else:
        # No time dimension, assume climatology already
        da2 = convert_rate_to_mmd(da2, units_attr=da2.attrs.get("units", None), time=None)
    return da2

def find_nearest_grid_cell(lat2d, lon2d, lat0, lon0):
    lon0n = normalize_lon(lon0, "-180_180")
    # Compute squared distance adjusting lon by cos(lat)
    if lat2d.ndim == 1 and lon2d.ndim == 1:
        lonm, latm = np.meshgrid(lon2d, lat2d)
    else:
        latm, lonm = lat2d, lon2d
    dlat = latm - lat0
    dlon = (lonm - lon0n) * np.cos(np.deg2rad(lat0))
    dist2 = dlat**2 + dlon**2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return idx

def extract_mosart_discharge_timeseries(case_name, start_date, end_date, lat0, lon0):
    # Try daily h1 files
    pattern_daily = os.path.join(E3SM_ROF_DIR, f"{case_name}.mosart.h1.*.nc")
    files_daily = sorted(glob.glob(pattern_daily))
    if len(files_daily) > 0:
        try:
            ds = xr.open_mfdataset(files_daily, combine="by_coords", parallel=False)
            ds = ds.sel(time=slice(start_date, end_date))
            lat2d, lon2d, lat_name, lon_name, ds = get_latlon_grids(ds)
            varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
            if varname not in ds.variables:
                raise KeyError(f"{varname} not found in MOSART daily files")
            idx = find_nearest_grid_cell(lat2d, lon2d, lat0, lon0)
            # Index extraction
            var = ds[varname]
            if lat2d.ndim == 2:
                ts = var[:, idx[0], idx[1]].to_series()
            else:
                # Handle 1D/unstructured
                # Try to build nearest index along "n" dimension
                # Build flattened index
                lat_flat = lat2d.ravel()
                lon_flat = lon2d.ravel()
                # Closest index:
                dlat = lat_flat - lat0
                dlon = (lon_flat - normalize_lon(lon0)) * np.cos(np.deg2rad(lat0))
                flat_idx = np.argmin(dlat**2 + dlon**2)
                ts = var[:, flat_idx].to_series()
            ts.name = "sim_discharge_m3s"
            ts = ts.dropna()
            return ts
        except Exception as e:
            print(f"Warning: failed to read daily MOSART files; {e}")
    # Fallback to monthly h0
    pattern_monthly = os.path.join(E3SM_ROF_DIR, f"{case_name}.mosart.h0.*.nc")
    files_monthly = sorted(glob.glob(pattern_monthly))
    if len(files_monthly) == 0:
        print(f"Warning: No MOSART rof files found in {E3SM_ROF_DIR}")
        return pd.Series(dtype=float)
    try:
        ds = xr.open_mfdataset(files_monthly, combine="by_coords", parallel=False)
        ds = ds.sel(time=slice(start_date, end_date))
        lat2d, lon2d, lat_name, lon_name, ds = get_latlon_grids(ds)
        varname = "RIVER_DISCHARGE_OVER_LAND_LIQ"
        if varname not in ds.variables:
            print(f"Warning: {varname} not found in MOSART monthly files")
            return pd.Series(dtype=float)
        idx = find_nearest_grid_cell(lat2d, lon2d, lat0, lon0)
        var = ds[varname]
        if lat2d.ndim == 2:
            ts = var[:, idx[0], idx[1]].to_series()
        else:
            lat_flat = lat2d.ravel()
            lon_flat = lon2d.ravel()
            dlat = lat_flat - lat0
            dlon = (lon_flat - normalize_lon(lon0)) * np.cos(np.deg2rad(lat0))
            flat_idx = np.argmin(dlat**2 + dlon**2)
            ts = var[:, flat_idx].to_series()
        # Repeat monthly values to daily (naive)
        ts_daily = ts.resample("D").ffill()
        ts_daily.name = "sim_discharge_m3s"
        ts_daily = ts_daily.dropna()
        return ts_daily
    except Exception as e:
        print(f"Warning: Error reading MOSART monthly files: {e}")
        return pd.Series(dtype=float)

def read_obs_discharge_timeseries(gauge_id, start_date, end_date):
    csv_path = os.path.join(OBS_STREAMFLOW_DIR, f"{gauge_id}.csv")
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        series = df["discharge_m3s"].loc[start_date:end_date]
        series.name = "obs_discharge_m3s"
        return series
    except Exception as e:
        print(f"Warning: Failed to read observed streamflow for gauge {gauge_id} at {csv_path}: {e}")
        return pd.Series(dtype=float)

def compute_streamflow_metrics(sim_series, obs_series):
    # Align on common dates
    df = pd.DataFrame({"sim": sim_series, "obs": obs_series}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    sim = df["sim"].values
    obs = df["obs"].values
    vol_bias = (np.sum(sim) - np.sum(obs)) / np.sum(obs) * 100.0 if np.sum(obs) != 0 else np.nan
    try:
        wd = wasserstein_distance(sim, obs)
    except Exception:
        wd = np.nan
    med_obs = np.median(obs) if len(obs) > 0 else np.nan
    wd_rel_pct = (wd / med_obs * 100.0) if (med_obs is not None and med_obs != 0 and not np.isnan(med_obs)) else np.nan
    return vol_bias, wd, wd_rel_pct

def create_bar_chart(df_summary, out_png):
    try:
        # Variables to plot: P, ET, Q (mm/yr)
        basins = df_summary["basin_name"].tolist()
        x = np.arange(len(basins))
        width = 0.12
        # Convert mm/day to mm/yr (approx)
        P_mod = (df_summary["P_mod_mmd"] * 365.0).values
        P_obs = (df_summary["P_obs_mmd"] * 365.0).values
        ET_mod = (df_summary["ET_mod_mmd"] * 365.0).values
        ET_obs = (df_summary["ET_obs_mmd"] * 365.0).values
        Q_mod = (df_summary["Q_mod_mmd"] * 365.0).values
        Q_obs = (df_summary["Q_obs_mmd"] * 365.0).values
        fig, ax = plt.subplots(figsize=(14, 6))
        # Positions
        ax.bar(x - 2*width, P_obs, width=width, color="#77b7ff", label="P obs")
        ax.bar(x - width, P_mod, width=width, color="#1f78b4", label="P model")
        ax.bar(x, ET_obs, width=width, color="#a6d854", label="ET obs")
        ax.bar(x + width, ET_mod, width=width, color="#33a02c", label="ET model")
        ax.bar(x + 2*width, Q_obs, width=width, color="#fdb462", label="Q obs")
        ax.bar(x + 3*width, Q_mod, width=width, color="#e31a1c", label="Q model")
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(basins, rotation=0)
        ax.set_ylabel("Flux (mm/yr)")
        ax.set_title("Model vs Observation: P, ET, Q (1985-1989)")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to create bar chart: {e}")

def create_radar_charts(df_summary, out_png):
    try:
        # Metrics:
        metrics = ["P_bias_pct", "ET_bias_pct", "Runoff_bias_pct", "Streamflow_vol_bias_pct", "WB_diff_pct", "WD_rel_pct"]
        labels = ["P bias %", "ET bias %", "Runoff bias %", "Streamflow bias %", "WB diff %", "WD rel %"]
        n_metrics = len(metrics)
        n_basins = df_summary.shape[0]
        ncols = 3
        nrows = int(np.ceil(n_basins / ncols))
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True), figsize=(16, 9))
        axes = axes.flatten()
        for i, (_, row) in enumerate(df_summary.iterrows()):
            values = [row[m] if np.isfinite(row[m]) else 0.0 for m in metrics]
            # Cap values to a reasonable range for plotting
            capped = []
            for v in values:
                if np.isnan(v):
                    v = 0.0
                if v > 200:
                    v = 200
                if v < -200:
                    v = -200
                capped.append(v)
            data = capped + capped[:1]
            ax = axes[i]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, data, color="#1f78b4", linewidth=2)
            ax.fill(angles, data, color="#1f78b4", alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticks([-200, -100, 0, 100, 200])
            ax.set_yticklabels(["-200", "-100", "0", "100", "200"], fontsize=7)
            ax.set_title(f"{row['basin_name']}", y=1.1, fontsize=12)
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Multi-variable Diagnostic Radar (1985-1989)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to create radar charts: {e}")

# ----------------------------
# Main process
# ----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    # Part 1: ELM climatological time-mean fields
    try:
        elm_ds = open_e3sm_elm_files(CASE_NAME, START_DATE, END_DATE)
        elm_clim = compute_elm_climatology(elm_ds)
        # Attach lat/lon coords and normalize lon
        lat2d, lon2d, lat_name, lon_name, elm_ds = get_latlon_grids(elm_ds)
        # Reassign coords to clim
        try:
            if lat2d.ndim == 2:
                elm_clim = elm_clim.assign_coords({lat_name: (("y", "x"), lat2d),
                                                   lon_name: (("y", "x"), lon2d)})
            else:
                # 1D case
                elm_clim = elm_clim.assign_coords({lat_name: (elm_ds[lat_name].dims, elm_ds[lat_name].values),
                                                   lon_name: (elm_ds[lon_name].dims, elm_ds[lon_name].values)})
        except Exception:
            pass
        # Save model climatologies
        try:
            model_nc_path = os.path.join(OUTPUT_DIR, "model_climatology_P_ET_Q_mmday.nc")
            elm_clim.to_netcdf(model_nc_path)
        except Exception as e:
            print(f"Warning: Failed to save model climatology NetCDF: {e}")
    except Exception as e:
        print(f"Error in Part 1 (ELM extraction): {e}")
        return

    # Part 2: ILAMB observations
    try:
        # GPCC precipitation
        gpcc_ds, gpcc_da = open_ilamb_dataset("pr", "GPCCv2018", "pr.nc", varname="pr")
        gpcc_clim = subset_time_and_climatology(gpcc_da, START_DATE, END_DATE)
        # MODIS ET
        modis_ds, modis_da = open_ilamb_dataset("evspsbl", "MODIS", "et_0.5x0.5.nc", varname="et")
        modis_clim = subset_time_and_climatology(modis_da, START_DATE, END_DATE)
        # LORA runoff
        lora_ds, lora_da = open_ilamb_dataset("mrro", "LORA", "LORA.nc", varname="mrro")
        lora_clim = subset_time_and_climatology(lora_da, START_DATE, END_DATE)
        # Save obs climatologies
        try:
            obs_clim_ds = xr.Dataset({
                "pr_mmd": gpcc_clim,
                "et_mmd": modis_clim,
                "mrro_mmd": lora_clim
            })
            obs_nc_path = os.path.join(OUTPUT_DIR, "obs_climatology_pr_et_mrro_mmday.nc")
            obs_clim_ds.to_netcdf(obs_nc_path)
        except Exception as e:
            print(f"Warning: Failed to save obs climatology NetCDF: {e}")
    except Exception as e:
        print(f"Error in Part 2 (ILAMB extraction): {e}")
        return

    # Prepare grids for basin clipping
    try:
        elm_lat2d, elm_lon2d, elm_lat_name, elm_lon_name, _ = get_latlon_grids(elm_ds)
    except Exception as e:
        print(f"Error: Failed to get ELM grid lat/lon: {e}")
        return
    try:
        gpcc_lat2d, gpcc_lon2d, gpcc_lat_name, gpcc_lon_name, gpcc_ds = get_latlon_grids(gpcc_ds)
    except Exception as e:
        print(f"Error: Failed to get GPCC grid lat/lon: {e}")
        return
    try:
        modis_lat2d, modis_lon2d, modis_lat_name, modis_lon_name, modis_ds = get_latlon_grids(modis_ds)
    except Exception as e:
        print(f"Error: Failed to get MODIS grid lat/lon: {e}")
        return
    try:
        lora_lat2d, lora_lon2d, lora_lat_name, lora_lon_name, lora_ds = get_latlon_grids(lora_ds)
    except Exception as e:
        print(f"Error: Failed to get LORA grid lat/lon: {e}")
        return

    # Part 3: Basin means
    # Load basin polygons
    try:
        with open(BASIN_GEOJSON, "r") as f:
            geo = json.load(f)
        features = geo.get("features", [])
    except Exception as e:
        print(f"Error: Failed to read basin polygons {BASIN_GEOJSON}: {e}")
        return

    # Map gauge_id -> polygons (list of exterior ring coordinate lists)
    basin_polygons = {}
    for feat in features:
        props = feat.get("properties", {})
        gid = str(props.get("grdc_no", "")).strip()
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        polygons = []
        if gtype == "Polygon":
            # coords: list of linear rings; use exterior ring coords[0]
            if len(coords) > 0:
                exterior = coords[0]
                polygons.append(exterior)
        elif gtype == "MultiPolygon":
            for poly in coords:
                if len(poly) > 0:
                    exterior = poly[0]
                    polygons.append(exterior)
        else:
            continue
        basin_polygons[gid] = polygons

    # Read gauge metadata
    try:
        gauge_df = pd.read_csv(GAUGE_METADATA_CSV)
        gauge_df["gauge_id"] = gauge_df["gauge_id"].astype(str)
        gauge_df = gauge_df.set_index("gauge_id")
    except Exception as e:
        print(f"Warning: Failed to read gauge metadata: {e}")
        gauge_df = pd.DataFrame()

    # Initialize results storage
    results = []

    # Part 4: Streamflow FDC metrics and combine
    for basin_name, gid in BASINS:
        print(f"Processing basin {basin_name} ({gid})")
        polygons = basin_polygons.get(gid, None)
        if polygons is None:
            print(f"Warning: No polygon found for gauge {gid}")
            polygons = []

        # Basin mean for model climatologies
        try:
            P_mod_field = elm_clim["P_mmd"]
            ET_mod_field = elm_clim["ET_mmd"]
            Q_mod_field = elm_clim["Q_mmd"]
            # Ensure variables are 2D aligned with lat/lon
            # Get 2D lat/lon
            P_mod_val = area_weighted_basin_mean(P_mod_field, elm_lat2d, elm_lon2d, polygons)
            ET_mod_val = area_weighted_basin_mean(ET_mod_field, elm_lat2d, elm_lon2d, polygons)
            Q_mod_val = area_weighted_basin_mean(Q_mod_field, elm_lat2d, elm_lon2d, polygons)
        except Exception as e:
            print(f"Warning: Failed computing model basin means for {basin_name}: {e}")
            P_mod_val = np.nan
            ET_mod_val = np.nan
            Q_mod_val = np.nan

        # Basin mean for observation climatologies
        try:
            P_obs_val = area_weighted_basin_mean(gpcc_clim, gpcc_lat2d, gpcc_lon2d, polygons)
        except Exception as e:
            print(f"Warning: Failed GPCC basin mean for {basin_name}: {e}")
            P_obs_val = np.nan
        try:
            ET_obs_val = area_weighted_basin_mean(modis_clim, modis_lat2d, modis_lon2d, polygons)
        except Exception as e:
            print(f"Warning: Failed MODIS ET basin mean for {basin_name}: {e}")
            ET_obs_val = np.nan
        try:
            Q_obs_val = area_weighted_basin_mean(lora_clim, lora_lat2d, lora_lon2d, polygons)
        except Exception as e:
            print(f"Warning: Failed LORA runoff basin mean for {basin_name}: {e}")
            Q_obs_val = np.nan

        # Compute biases (%)
        def pct_bias(mod, obs):
            if obs is None or np.isnan(obs) or obs == 0:
                return np.nan
            return (mod - obs) / obs * 100.0

        P_bias_pct = pct_bias(P_mod_val, P_obs_val)
        ET_bias_pct = pct_bias(ET_mod_val, ET_obs_val)
        Runoff_bias_pct = pct_bias(Q_mod_val, Q_obs_val)

        # Water balance residual difference in % of Pobs: (Pmod - ETmod - Qmod) - (Pobs - ETobs - Qobs)
        WB_model = P_mod_val - ET_mod_val - Q_mod_val
        WB_obs = P_obs_val - ET_obs_val - Q_obs_val
        if P_obs_val is None or np.isnan(P_obs_val) or P_obs_val == 0:
            WB_diff_pct = np.nan
        else:
            WB_diff_pct = (WB_model - WB_obs) / P_obs_val * 100.0

        # Streamflow metrics
        # Extract gauge location
        lat0 = np.nan
        lon0 = np.nan
        area_km2 = np.nan
        river_name = ""
        try:
            if gid in gauge_df.index:
                row = gauge_df.loc[gid]
                lat0 = float(row["lat"])
                lon0 = float(row["lon"])
                area_km2 = float(row.get("area_km2", np.nan))
                river_name = str(row.get("river_name", ""))
        except Exception as e:
            print(f"Warning: Could not find metadata for gauge {gid}: {e}")

        sim_ts = extract_mosart_discharge_timeseries(CASE_NAME, START_DATE, END_DATE, lat0, lon0)
        obs_ts = read_obs_discharge_timeseries(gid, START_DATE, END_DATE)
        Streamflow_vol_bias_pct, Wasserstein_dist, WD_rel_pct = compute_streamflow_metrics(sim_ts, obs_ts)

        # Compile results
        results.append({
            "basin_name": basin_name,
            "gauge_id": gid,
            "river_name": river_name,
            "area_km2": area_km2,
            "P_mod_mmd": P_mod_val,
            "P_obs_mmd": P_obs_val,
            "ET_mod_mmd": ET_mod_val,
            "ET_obs_mmd": ET_obs_val,
            "Q_mod_mmd": Q_mod_val,
            "Q_obs_mmd": Q_obs_val,
            "P_bias_pct": P_bias_pct,
            "ET_bias_pct": ET_bias_pct,
            "Runoff_bias_pct": Runoff_bias_pct,
            "WB_model_mmd": WB_model,
            "WB_obs_mmd": WB_obs,
            "WB_diff_pct": WB_diff_pct,
            "Streamflow_vol_bias_pct": Streamflow_vol_bias_pct,
            "Wasserstein_distance_m3s": Wasserstein_dist,
            "WD_rel_pct": WD_rel_pct
        })

    # Build summary DataFrame and save
    df_summary = pd.DataFrame(results)
    summary_csv = os.path.join(OUTPUT_DIR, "basin_integrated_diagnostic_summary.csv")
    try:
        df_summary.to_csv(summary_csv, index=False)
    except Exception as e:
        print(f"Warning: Failed to save summary CSV: {e}")

    # Part 5: Visualizations
    bar_png = os.path.join(OUTPUT_DIR, "model_vs_obs_P_ET_Q_bar.png")
    create_bar_chart(df_summary, bar_png)

    radar_png = os.path.join(OUTPUT_DIR, "radar_multivariable_diagnostic.png")
    create_radar_charts(df_summary, radar_png)

    print(f"Analysis complete. Outputs are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()