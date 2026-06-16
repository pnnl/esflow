import os
import sys
import glob
import re
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}", file=sys.stderr)

def download_file(url, dest_path):
    try:
        if os.path.exists(dest_path):
            print(f"File already exists, skipping download: {dest_path}")
            return dest_path
        print(f"Downloading {url} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
        return dest_path
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return None

def spherical_to_cartesian(lat_deg, lon_deg):
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z

def nearest_neighbor_regrid(model_lat2d, model_lon2d, model_field2d, target_lat2d, target_lon2d):
    try:
        valid = np.isfinite(model_field2d) & np.isfinite(model_lat2d) & np.isfinite(model_lon2d)
        if valid.sum() == 0:
            print("No valid model data points found for regridding.", file=sys.stderr)
            return np.full(target_lat2d.shape, np.nan)

        lat_m = model_lat2d[valid]
        lon_m = model_lon2d[valid]
        data_m = model_field2d[valid]

        # Convert to 3D unit sphere coordinates for KDTree (handles lon periodicity better)
        x_m, y_m, z_m = spherical_to_cartesian(lat_m, lon_m)
        tree = cKDTree(np.column_stack([x_m, y_m, z_m]))

        lat_t = target_lat2d.ravel()
        lon_t = target_lon2d.ravel()
        x_t, y_t, z_t = spherical_to_cartesian(lat_t, lon_t)
        dist, idx = tree.query(np.column_stack([x_t, y_t, z_t]), k=1)
        mapped = data_m[idx].reshape(target_lat2d.shape)
        return mapped
    except Exception as e:
        print(f"Error during nearest-neighbor regridding: {e}", file=sys.stderr)
        return np.full(target_lat2d.shape, np.nan)

def find_elm_files(base_dir, case_name, start_year, end_year):
    pattern = os.path.join(base_dir, f"{case_name}.elm.h0.*.nc")
    files = glob.glob(pattern)
    selected = []
    for f in files:
        # Extract year from filename using regex
        m = re.search(r"\.(\d{4})\-(\d{2})\.nc$", f)
        if m:
            year = int(m.group(1))
            if start_year <= year <= end_year:
                selected.append(f)
    return sorted(selected)

def compute_area_weighted_mean_bias(bias2d, lat2d, valid_mask):
    try:
        lat_rad = np.deg2rad(lat2d)
        weights = np.cos(lat_rad)
        weights = np.where(valid_mask, weights, 0.0)
        num = np.nansum(bias2d * weights)
        den = np.nansum(weights)
        if den == 0 or np.isnan(den):
            return np.nan
        return num / den
    except Exception as e:
        print(f"Error computing area-weighted mean bias: {e}", file=sys.stderr)
        return np.nan

def compute_spatial_correlation(a2d, b2d, valid_mask):
    try:
        a = a2d[valid_mask].ravel()
        b = b2d[valid_mask].ravel()
        if a.size < 2:
            return np.nan
        r, p = pearsonr(a, b)
        return r
    except Exception as e:
        print(f"Error computing spatial correlation: {e}", file=sys.stderr)
        return np.nan

def main():
    # Paths and settings
    case_name = "sample.v3.LR.historical"
    elm_dir = "./data/sample/e3sm/lnd"
    ilamb_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_03_et_benchmark/run3_output"
    ensure_dir(output_dir)

    # Step 1: Download MODIS ET from ILAMB
    modis_local_path = os.path.join(output_dir, "MODIS_et_0.5x0.5.nc")
    modis_nc = download_file(ilamb_url, modis_local_path)
    if modis_nc is None or not os.path.exists(modis_nc):
        print("MODIS ET file is not available. Exiting.")
        return

    # Step 2: Read MODIS ET and compute time-mean
    try:
        ds_obs = xr.open_dataset(modis_nc)
        if "et" not in ds_obs.data_vars:
            raise KeyError("Variable 'et' not found in MODIS dataset")
        et_obs = ds_obs["et"]
        # Units are mm/day according to instructions
        et_obs_mean = et_obs.mean(dim=[d for d in et_obs.dims if d in ("time", "Time", "t")], skipna=True)
        # Ensure dims are (lat, lon)
        lat_name = "lat" if "lat" in ds_obs.coords else [c for c in ds_obs.coords if "lat" in c.lower()][0]
        lon_name = "lon" if "lon" in ds_obs.coords else [c for c in ds_obs.coords if "lon" in c.lower()][0]
        lats_obs = ds_obs[lat_name]
        lons_obs = ds_obs[lon_name]
        # Broadcast to 2D
        lat2d_obs, lon2d_obs = xr.broadcast(lats_obs, lons_obs)
        # Convert lons to 0-360 for consistency in KDTree
        lon2d_obs_vals = lon2d_obs.values.copy()
        lon2d_obs_vals = np.mod(lon2d_obs_vals, 360.0)
        lat2d_obs_vals = lat2d_obs.values
        et_obs_mean_vals = et_obs_mean.values
    except Exception as e:
        print(f"Error reading MODIS dataset: {e}", file=sys.stderr)
        return

    # Step 3: Read E3SM ELM files and compute 1985-1989 climatology ET
    try:
        elm_files = find_elm_files(elm_dir, case_name, 1985, 1989)
        if len(elm_files) == 0:
            print("No ELM files found for years 1985-1989.", file=sys.stderr)
            return
        print(f"Found {len(elm_files)} ELM files for 1985-1989.")
        ds_elm = xr.open_mfdataset(elm_files, combine="by_coords")
        # Check variables
        for v in ["QVEGE", "QVEGT", "QSOIL", "lat", "lon"]:
            if v not in ds_elm.variables and v not in ds_elm.coords:
                print(f"Warning: Variable '{v}' not found in ELM dataset.")
        qvege = ds_elm["QVEGE"]
        qvegt = ds_elm["QVEGT"]
        qsoil = ds_elm["QSOIL"]
        et_elm = qvege + qvegt + qsoil  # mm/s
        # Convert to mm/day
        et_elm_mmday = et_elm * 86400.0
        et_elm_mean = et_elm_mmday.mean(dim=[d for d in et_elm_mmday.dims if d.lower() in ("time",)], skipna=True)
        # Get model grid
        lat_elm = ds_elm["lat"]
        lon_elm = ds_elm["lon"]
        lat2d_elm, lon2d_elm = xr.broadcast(lat_elm, lon_elm)
        # Convert lons to 0-360 for KDTree consistency
        lon2d_elm_vals = np.mod(lon2d_elm.values, 360.0)
        lat2d_elm_vals = lat2d_elm.values
        et_elm_mean_vals = et_elm_mean.values
    except Exception as e:
        print(f"Error reading ELM dataset: {e}", file=sys.stderr)
        return

    # Step 4: Regrid ELM ET mean to MODIS grid using nearest neighbor on sphere
    try:
        model_on_obs = nearest_neighbor_regrid(
            lat2d_elm_vals, lon2d_elm_vals, et_elm_mean_vals, lat2d_obs_vals, lon2d_obs_vals
        )
    except Exception as e:
        print(f"Error in regridding step: {e}", file=sys.stderr)
        return

    # Step 5: Compute bias and metrics
    try:
        bias = model_on_obs - et_obs_mean_vals
        valid_mask = np.isfinite(model_on_obs) & np.isfinite(et_obs_mean_vals)
        global_mean_bias = compute_area_weighted_mean_bias(bias, lat2d_obs_vals, valid_mask)
        spatial_corr = compute_spatial_correlation(model_on_obs, et_obs_mean_vals, valid_mask)
        print(f"Global mean bias (mm/day): {global_mean_bias:.4f}")
        print(f"Spatial correlation: {spatial_corr:.4f}" if spatial_corr == spatial_corr else "Spatial correlation: NaN")
    except Exception as e:
        print(f"Error computing bias and metrics: {e}", file=sys.stderr)
        global_mean_bias = np.nan
        spatial_corr = np.nan

    # Step 6: Save metrics to CSV
    try:
        metrics_df = pd.DataFrame([
            {"metric": "global_mean_bias_mm_per_day", "value": global_mean_bias},
            {"metric": "spatial_correlation", "value": spatial_corr},
        ])
        metrics_csv_path = os.path.join(output_dir, "et_benchmark_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics CSV: {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}", file=sys.stderr)

    # Step 7: Save bias and fields to NetCDF on obs grid
    try:
        ds_out = xr.Dataset()
        ds_out = ds_out.assign_coords({
            "lat": (("lat",), lats_obs.values),
            "lon": (("lon",), lons_obs.values),
        })
        # Ensure obs grid is 2D but dims are lat, lon with broadcasting
        model_on_obs_da = xr.DataArray(
            model_on_obs,
            dims=("lat", "lon"),
            coords={"lat": lats_obs.values, "lon": lons_obs.values},
            name="model_et_mean_on_obs"
        )
        obs_et_mean_da = xr.DataArray(
            et_obs_mean_vals,
            dims=("lat", "lon"),
            coords={"lat": lats_obs.values, "lon": lons_obs.values},
            name="obs_et_mean"
        )
        bias_da = xr.DataArray(
            bias,
            dims=("lat", "lon"),
            coords={"lat": lats_obs.values, "lon": lons_obs.values},
            name="bias_model_minus_obs"
        )
        model_on_obs_da.attrs["units"] = "mm/day"
        obs_et_mean_da.attrs["units"] = "mm/day"
        bias_da.attrs["units"] = "mm/day"
        ds_out["model_et_mean_on_obs"] = model_on_obs_da
        ds_out["obs_et_mean"] = obs_et_mean_da
        ds_out["bias_model_minus_obs"] = bias_da
        ds_out.attrs["description"] = "E3SM ELM ET (QVEGE+QVEGT+QSOIL) 1985-1989 climatology regridded to MODIS grid; MODIS ET time-mean; bias=model-obs."
        ds_out.attrs["global_mean_bias_mm_per_day"] = float(global_mean_bias) if np.isfinite(global_mean_bias) else np.nan
        ds_out.attrs["spatial_correlation"] = float(spatial_corr) if np.isfinite(spatial_corr) else np.nan
        nc_out_path = os.path.join(output_dir, "et_benchmark_bias_on_obs_grid.nc")
        ds_out.to_netcdf(nc_out_path)
        print(f"Saved NetCDF: {nc_out_path}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}", file=sys.stderr)

    # Step 8: Plot bias map
    try:
        # Prepare data for plotting (handle 1D lon, lat)
        lons_plot = lons_obs.values
        lats_plot = lats_obs.values

        # Convert longitudes to [-180, 180] for plotting
        lons_plot_wrapped = np.where(lons_plot > 180, lons_plot - 360, lons_plot)

        # Add cyclic point if lon is 1D
        if lons_plot_wrapped.ndim == 1 and lats_plot.ndim == 1:
            bias_plot, lons_cyclic = add_cyclic_point(bias, coord=lons_plot_wrapped)
            LON2D, LAT2D = np.meshgrid(lons_cyclic, lats_plot)
        else:
            # If 2D, just use as-is without cyclic point
            bias_plot = bias
            LON2D = np.where(lon2d_obs_vals > 180, lon2d_obs_vals - 360, lon2d_obs_vals)
            LAT2D = lat2d_obs_vals

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        vlim = np.nanmax(np.abs(bias_plot)) if np.isfinite(bias_plot).any() else 1.0
        vlim = max(vlim, 1e-6)
        p = ax.pcolormesh(LON2D, LAT2D, bias_plot, transform=ccrs.PlateCarree(),
                          cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
        ax.coastlines(linewidth=0.6)
        cb = plt.colorbar(p, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
        cb.set_label("ET Bias (Model - MODIS) [mm/day]")
        title = "ELM ET (1985-1989 mean) - MODIS ET (time-mean)\n"
        title += f"Global mean bias: {global_mean_bias:.3f} mm/day; Spatial corr: {spatial_corr:.3f}" if np.isfinite(spatial_corr) else f"Global mean bias: {global_mean_bias:.3f} mm/day; Spatial corr: NaN"
        ax.set_title(title)
        png_path = os.path.join(output_dir, "et_bias_map.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved bias map: {png_path}")
    except Exception as e:
        print(f"Error creating bias map: {e}", file=sys.stderr)

    print("Benchmarking complete.")

if __name__ == "__main__":
    main()