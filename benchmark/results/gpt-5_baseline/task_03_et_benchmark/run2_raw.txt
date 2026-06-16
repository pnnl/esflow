#!/usr/bin/env python3
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}", file=sys.stderr)

def download_file(url, dest_path):
    try:
        import urllib.request
        print(f"Downloading {url} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}", file=sys.stderr)
        return False

def find_coord_names(da):
    # Attempt to identify latitude and longitude coordinate names
    candidates_lat = ["lat", "latitude", "LAT", "nav_lat", "y"]
    candidates_lon = ["lon", "longitude", "LON", "nav_lon", "x"]
    lat_name = None
    lon_name = None
    for name in da.coords:
        if name.lower() in candidates_lat and lat_name is None:
            lat_name = name
        if name.lower() in candidates_lon and lon_name is None:
            lon_name = name
    # Some datasets keep lat/lon as variables not coords
    if lat_name is None:
        for name in da.dims:
            if name.lower() in candidates_lat:
                lat_name = name
                break
    if lon_name is None:
        for name in da.dims:
            if name.lower() in candidates_lon:
                lon_name = name
                break
    if lat_name is None and "lat" in da._to_temp_dataset().variables:
        lat_name = "lat"
    if lon_name is None and "lon" in da._to_temp_dataset().variables:
        lon_name = "lon"
    return lat_name, lon_name

def wrap_lon_to_match(source_lon, target_lon):
    # Convert source_lon to have same wrap as target_lon grid
    src = np.array(source_lon)
    tgt = np.array(target_lon)
    if np.nanmax(tgt) > 180:
        # target uses 0..360
        wrapped = np.mod(src, 360.0)
    else:
        # target uses -180..180
        wrapped = ((src + 180.0) % 360.0) - 180.0
    return wrapped

def to_2d(lat, lon):
    # Ensure 2D lat/lon arrays
    lat_arr = np.array(lat)
    lon_arr = np.array(lon)
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_arr, lat_arr)
        return lat2d, lon2d
    elif lat_arr.ndim == 2 and lon_arr.ndim == 2:
        return lat_arr, lon_arr
    else:
        # Attempt broadcasting if one is 1D and the other is 2D
        if lat_arr.ndim == 2 and lon_arr.ndim == 1:
            lon2d = np.tile(lon_arr[np.newaxis, :], (lat_arr.shape[0], 1))
            return lat_arr, lon2d
        if lat_arr.ndim == 1 and lon_arr.ndim == 2:
            lat2d = np.tile(lat_arr[:, np.newaxis], (1, lon_arr.shape[1]))
            return lat2d, lon_arr
        raise ValueError("Cannot form 2D lat/lon arrays from given inputs.")

def regrid_nearest(source_da, target_lat_1d, target_lon_1d):
    # Build a nearest-neighbor mapper from source_da to target regular grid
    lat_name, lon_name = find_coord_names(source_da)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not find latitude/longitude coordinates in source DataArray.")
    # Extract lat/lon values
    src_lat_vals = source_da[lat_name].values
    src_lon_vals = source_da[lon_name].values
    # Wrap source lon to target wrap
    wrapped_src_lon = wrap_lon_to_match(src_lon_vals, target_lon_1d)
    # Make 2D lat/lon grids
    src_lat2d, src_lon2d = to_2d(src_lat_vals, wrapped_src_lon)
    # Prepare target 2D grids (wrap not necessary if we applied to source)
    tgt_lon2d, tgt_lat2d = np.meshgrid(target_lon_1d, target_lat_1d)
    # Flatten points
    pts_src = np.column_stack([src_lat2d.ravel(), src_lon2d.ravel()])
    vals_src = source_da.values.ravel()
    valid = np.isfinite(vals_src) & np.isfinite(pts_src[:,0]) & np.isfinite(pts_src[:,1])
    if np.count_nonzero(valid) == 0:
        raise ValueError("No valid source data for regridding.")
    tree = cKDTree(pts_src[valid, :])
    dists, idxs = tree.query(np.column_stack([tgt_lat2d.ravel(), tgt_lon2d.ravel()]), k=1)
    mapped_vals = np.full(tgt_lat2d.size, np.nan, dtype=float)
    mapped_vals[:] = vals_src[valid][idxs]
    mapped_vals2d = mapped_vals.reshape(tgt_lat2d.shape)
    da_out = xr.DataArray(
        mapped_vals2d,
        coords={"lat": target_lat_1d, "lon": target_lon_1d},
        dims=("lat", "lon"),
        name=source_da.name
    )
    da_out.attrs.update(source_da.attrs)
    return da_out

def compute_edges_1d(coords):
    coords = np.asarray(coords)
    # Compute edges by midpoints
    diffs = np.diff(coords)
    # For constant grids, edges are simple; handle irregular too
    edges = np.zeros(coords.size + 1, dtype=float)
    edges[1:-1] = coords[:-1] + diffs / 2.0
    edges[0] = coords[0] - diffs[0] / 2.0
    edges[-1] = coords[-1] + diffs[-1] / 2.0
    return edges

def weighted_global_mean(data2d, lat1d):
    # area weight by cos(lat)
    weights = np.cos(np.deg2rad(lat1d))
    W = np.tile(weights[:, np.newaxis], (1, data2d.shape[1]))
    mask = np.isfinite(data2d)
    num = np.nansum(data2d[mask] * W[mask])
    den = np.nansum(W[mask])
    return np.nan if den == 0 else (num / den)

def main():
    # Paths and case info
    case_name = "sample.v3.LR.historical"
    lnd_dir = "./data/sample/e3sm/lnd"
    start_year = 1985
    end_year = 1989

    # Output directory (as specified by user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_03_et_benchmark/run2_output"
    ensure_dir(output_dir)

    # 1) Fetch MODIS ET from ILAMB
    ilamb_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
    modis_local_path = os.path.join(output_dir, "MODIS_et_0.5x0.5.nc")
    if not os.path.exists(modis_local_path):
        success = download_file(ilamb_url, modis_local_path)
        if not success:
            print("Attempting to open remote dataset directly...", file=sys.stderr)
            modis_local_path = ilamb_url  # fallback to remote open

    # 2) Open MODIS dataset and compute time-mean ET
    try:
        ds_obs = xr.open_dataset(modis_local_path)
    except Exception as e:
        print(f"Failed to open MODIS dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine variable 'et'
    if "et" in ds_obs.variables:
        obs_var = "et"
    else:
        # Fallback to any variable with 'et' in name
        obs_vars = [v for v in ds_obs.data_vars if "et" in v.lower()]
        if len(obs_vars) == 0:
            print("Could not find 'et' variable in MODIS dataset.", file=sys.stderr)
            sys.exit(1)
        obs_var = obs_vars[0]

    da_obs = ds_obs[obs_var]
    # Convert observations to mm/day if needed
    obs_units = da_obs.attrs.get("units", "").lower()
    obs_to_mm_per_day = 1.0
    if "kg m-2 s-1" in obs_units or "kg/m2/s" in obs_units or "mm s-1" in obs_units or "mm/s" in obs_units:
        obs_to_mm_per_day = 86400.0
    elif "mm/day" in obs_units or "mm d-1" in obs_units:
        obs_to_mm_per_day = 1.0
    elif obs_units == "" or obs_units is None:
        # Assume native units are mm/day per ILAMB docs for some datasets, but MODIS ET is ambiguous.
        # We will assume mm/day if units missing.
        obs_to_mm_per_day = 1.0
    else:
        # Unknown units; try to infer typical: MODIS ET in mm/day
        obs_to_mm_per_day = 1.0

    try:
        # Mean over time dimension (if exists)
        if "time" in da_obs.dims:
            obs_et_clim = (da_obs * obs_to_mm_per_day).mean(dim="time", skipna=True)
        else:
            obs_et_clim = da_obs * obs_to_mm_per_day
    except Exception as e:
        print(f"Failed to compute observation climatology: {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Open ELM files and compute ET climatology
    pattern = os.path.join(lnd_dir, f"{case_name}.elm.h0.*.nc")
    files = sorted(glob.glob(pattern))
    # Filter years 1985-1989
    selected_files = []
    for f in files:
        # parse year from filename
        base = os.path.basename(f)
        # Expected pattern ...elm.h0.YYYY-MM.nc
        try:
            parts = base.split(".")
            y_m = parts[-1].replace(".nc", "")
            if "-" in y_m:
                yr = int(y_m.split("-")[0])
                if start_year <= yr <= end_year:
                    selected_files.append(f)
        except Exception:
            continue

    if len(selected_files) == 0:
        print("No ELM files found for years 1985-1989.", file=sys.stderr)
        sys.exit(1)

    try:
        # Only open variables needed to reduce memory
        def preprocess(ds):
            to_keep = [v for v in ["QVEGE", "QVEGT", "QSOIL", "lat", "lon"] if v in ds.variables or v in ds.coords]
            return ds[to_keep]
        ds_lnd = xr.open_mfdataset(selected_files, combine="by_coords", preprocess=preprocess)
    except Exception as e:
        print(f"Failed to open ELM land dataset: {e}", file=sys.stderr)
        sys.exit(1)

    missing = [v for v in ["QVEGE", "QVEGT", "QSOIL"] if v not in ds_lnd.variables]
    if missing:
        print(f"Missing ELM variables: {missing}", file=sys.stderr)
        sys.exit(1)
    try:
        et_model = ds_lnd["QVEGE"] + ds_lnd["QVEGT"] + ds_lnd["QSOIL"]  # mm/s
        et_model_mm_per_day = et_model * 86400.0  # convert to mm/day
        if "time" in et_model_mm_per_day.dims:
            et_model_clim = et_model_mm_per_day.mean(dim="time", skipna=True)
        else:
            et_model_clim = et_model_mm_per_day
        et_model_clim.name = "et_model_mm_per_day"
        et_model_clim.attrs["units"] = "mm/day"
    except Exception as e:
        print(f"Failed to compute ELM ET climatology: {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Regrid model climatology to MODIS grid
    # Extract target grid
    lat_obs_name, lon_obs_name = find_coord_names(obs_et_clim)
    if lat_obs_name is None or lon_obs_name is None:
        print("Could not determine lat/lon names in MODIS dataset.", file=sys.stderr)
        sys.exit(1)

    lat_obs = ds_obs[lat_obs_name].values
    lon_obs = ds_obs[lon_obs_name].values
    # Ensure 1D
    if lat_obs.ndim != 1 or lon_obs.ndim != 1:
        # Attempt to collapse if 2D regular grid
        try:
            lat_obs_unique = np.unique(lat_obs)
            lon_obs_unique = np.unique(lon_obs)
            lat_obs = lat_obs_unique
            lon_obs = lon_obs_unique
        except Exception as e:
            print(f"Observation grid lat/lon not 1D and cannot be simplified: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        et_model_on_obs = regrid_nearest(et_model_clim, lat_obs, lon_obs)
        et_model_on_obs.name = "et_model_mm_per_day_on_obs_grid"
        et_model_on_obs.attrs["units"] = "mm/day"
    except Exception as e:
        print(f"Regridding failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 5) Align observation climatology to ensure dims 'lat','lon'
    try:
        if obs_et_clim.dims != ("lat", "lon"):
            # Rename dims if necessary
            rename_dict = {}
            if lat_obs_name != "lat":
                rename_dict[lat_obs_name] = "lat"
            if lon_obs_name != "lon":
                rename_dict[lon_obs_name] = "lon"
            obs_et_clim = obs_et_clim.rename(rename_dict)
        # Convert to mm/day if not already
        obs_et_clim_mm_per_day = obs_et_clim * 1.0  # already applied conversion above
        obs_et_clim_mm_per_day.name = "et_modis_mm_per_day"
        obs_et_clim_mm_per_day.attrs["units"] = "mm/day"
    except Exception as e:
        print(f"Failed to align observation climatology: {e}", file=sys.stderr)
        sys.exit(1)

    # 6) Compute bias and metrics
    # Align coordinates
    obs_on_grid = obs_et_clim_mm_per_day
    model_on_grid = et_model_on_obs

    # Ensure same coords (lat, lon)
    model_on_grid = model_on_grid.transpose("lat", "lon")
    obs_on_grid = obs_on_grid.transpose("lat", "lon")
    # Mask where obs is NaN
    bias = model_on_grid - obs_on_grid
    bias = bias.where(np.isfinite(obs_on_grid))

    # Global mean bias (area-weighted over available cells)
    try:
        gmb = weighted_global_mean(bias.values, obs_on_grid["lat"].values)
    except Exception as e:
        print(f"Failed to compute global mean bias: {e}", file=sys.stderr)
        gmb = np.nan

    # Spatial correlation across grid cells (unweighted)
    try:
        mask = np.isfinite(model_on_grid.values) & np.isfinite(obs_on_grid.values)
        mvals = model_on_grid.values[mask]
        ovals = obs_on_grid.values[mask]
        if mvals.size > 1 and np.nanstd(mvals) > 0 and np.nanstd(ovals) > 0:
            r, p = stats.pearsonr(mvals, ovals)
            spatial_corr = r
        else:
            spatial_corr = np.nan
    except Exception as e:
        print(f"Failed to compute spatial correlation: {e}", file=sys.stderr)
        spatial_corr = np.nan

    # 7) Save metrics
    metrics = {
        "global_mean_bias_mm_per_day": [float(gmb) if np.isfinite(gmb) else np.nan],
        "spatial_correlation": [float(spatial_corr) if np.isfinite(spatial_corr) else np.nan]
    }
    metrics_df = pd.DataFrame(metrics)
    try:
        metrics_csv = os.path.join(output_dir, "metrics_et_e3sm_vs_modis.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics to {metrics_csv}")
    except Exception as e:
        print(f"Failed to save metrics CSV: {e}", file=sys.stderr)

    # 8) Save NetCDF with model, obs, bias on common grid
    try:
        ds_out = xr.Dataset(
            {
                "model_et_mm_per_day": model_on_grid,
                "modis_et_mm_per_day": obs_on_grid,
                "bias_model_minus_modis_mm_per_day": bias
            }
        )
        ds_out["model_et_mm_per_day"].attrs["long_name"] = "E3SM ELM evapotranspiration climatology (1985-1989)"
        ds_out["modis_et_mm_per_day"].attrs["long_name"] = "MODIS evapotranspiration climatology (time-mean)"
        ds_out["bias_model_minus_modis_mm_per_day"].attrs["long_name"] = "Bias (model - MODIS)"
        ds_out.attrs["description"] = "E3SM ELM vs MODIS ET climatology benchmark; model years 1985-1989, MODIS time-mean"
        ds_out.attrs["units"] = "mm/day"
        nc_path = os.path.join(output_dir, "et_benchmark_e3sm_vs_modis.nc")
        ds_out.to_netcdf(nc_path)
        print(f"Saved NetCDF to {nc_path}")
    except Exception as e:
        print(f"Failed to save NetCDF: {e}", file=sys.stderr)

    # 9) Plot bias map
    try:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

        lat = obs_on_grid["lat"].values
        lon = obs_on_grid["lon"].values
        # Compute color scale symmetric around zero using robust 95th percentile
        abs95 = np.nanpercentile(np.abs(bias.values), 95) if np.isfinite(bias.values).any() else 1.0
        vmax = max(1e-6, abs95)
        vmin = -vmax
        # Edges for pcolormesh
        lat_edges = compute_edges_1d(lat)
        lon_edges = compute_edges_1d(lon)
        Lon, Lat = np.meshgrid(lon_edges, lat_edges)
        # Plot
        pcm = ax.pcolormesh(Lon, Lat, bias.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        cb = plt.colorbar(pcm, orientation="horizontal", pad=0.05, fraction=0.05)
        cb.set_label("Bias in ET (mm/day) [E3SM - MODIS]")

        title = f"E3SM ELM vs MODIS ET Bias (1985–1989)\nGlobal mean bias: {gmb:.3f} mm/day, Spatial corr: {spatial_corr:.3f}"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()

        png_path = os.path.join(output_dir, "bias_map_et_e3sm_minus_modis.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved bias map to {png_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Failed to create/save bias map: {e}", file=sys.stderr)

    print("Benchmark complete.")

if __name__ == "__main__":
    main()