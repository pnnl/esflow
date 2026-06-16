import os
import sys
import urllib.request
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# Output directory
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_03_et_benchmark/run1_output"
os.makedirs(outdir, exist_ok=True)

# ============================================================
# Step 1: Fetch MODIS ET dataset from ILAMB
# ============================================================
modis_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
modis_local = os.path.join(outdir, "modis_et_0.5x0.5.nc")

if not os.path.exists(modis_local):
    print(f"Downloading MODIS ET from {modis_url} ...")
    try:
        urllib.request.urlretrieve(modis_url, modis_local)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading MODIS ET: {e}")
        sys.exit(1)
else:
    print(f"MODIS ET file already exists: {modis_local}")

# ============================================================
# Step 2: Load E3SM ELM output for 1985-1989
# ============================================================
print("Loading E3SM ELM output ...")
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"

elm_files = []
for year in range(1985, 1990):
    for month in range(1, 13):
        fname = os.path.join(elm_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fname):
            elm_files.append(fname)
        else:
            print(f"  Warning: missing {fname}")

if len(elm_files) == 0:
    print("ERROR: No ELM files found!")
    sys.exit(1)

print(f"  Found {len(elm_files)} ELM files for 1985-1989")

try:
    ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
except Exception as e:
    print(f"  Trying with concat_dim='time' ... ({e})")
    try:
        ds_elm = xr.open_mfdataset(elm_files, combine='nested', concat_dim='time', decode_times=True)
    except Exception as e2:
        print(f"  Error loading ELM files: {e2}")
        sys.exit(1)

print(f"  ELM dataset dimensions: {dict(ds_elm.dims)}")
print(f"  ELM dataset variables: {list(ds_elm.data_vars)[:20]}")

# Check available variables
for v in ['QVEGE', 'QVEGT', 'QSOIL']:
    if v in ds_elm:
        print(f"  {v}: shape={ds_elm[v].shape}, units={ds_elm[v].attrs.get('units', 'unknown')}")
    else:
        print(f"  WARNING: {v} not found in ELM output!")

# Compute ET = QVEGE + QVEGT + QSOIL (mm/s)
print("Computing E3SM ET = QVEGE + QVEGT + QSOIL ...")
try:
    et_elm = ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']
except Exception as e:
    print(f"Error computing ET: {e}")
    sys.exit(1)

# Compute time-mean
et_elm_mean = et_elm.mean(dim='time')

# Convert from mm/s to kg/m2/s (numerically same) then to mm/day
# mm/s * 86400 s/day = mm/day
et_elm_mean_mmday = et_elm_mean * 86400.0
print(f"  E3SM ET mean range: {float(et_elm_mean_mmday.min()):.4f} to {float(et_elm_mean_mmday.max()):.4f} mm/day")

# Get lat/lon from ELM
# ELM can have 1D or 2D lat/lon
if 'lat' in ds_elm.coords and 'lon' in ds_elm.coords:
    elm_lat = ds_elm['lat']
    elm_lon = ds_elm['lon']
elif 'lat' in ds_elm and 'lon' in ds_elm:
    elm_lat = ds_elm['lat']
    elm_lon = ds_elm['lon']
else:
    print("  Looking for alternative coordinate names...")
    for v in ds_elm.coords:
        print(f"    coord: {v}, shape={ds_elm[v].shape}")
    for v in ds_elm.data_vars:
        if 'lat' in v.lower() or 'lon' in v.lower():
            print(f"    var: {v}, shape={ds_elm[v].shape}")
    sys.exit(1)

print(f"  ELM lat shape: {elm_lat.shape}, lon shape: {elm_lon.shape}")

# ============================================================
# Step 3: Load MODIS ET and compute time-mean
# ============================================================
print("Loading MODIS ET observation ...")
try:
    ds_modis = xr.open_dataset(modis_local)
    print(f"  MODIS variables: {list(ds_modis.data_vars)}")
    print(f"  MODIS dimensions: {dict(ds_modis.dims)}")
    
    # The variable inside is "et"
    if 'et' in ds_modis:
        et_modis = ds_modis['et']
    elif 'evspsbl' in ds_modis:
        et_modis = ds_modis['evspsbl']
    else:
        print(f"  Available vars: {list(ds_modis.data_vars)}")
        # Try the first data variable
        varname = list(ds_modis.data_vars)[0]
        print(f"  Using variable: {varname}")
        et_modis = ds_modis[varname]
    
    print(f"  MODIS ET shape: {et_modis.shape}")
    print(f"  MODIS ET units: {et_modis.attrs.get('units', 'unknown')}")
    
    # Compute time-mean
    if 'time' in et_modis.dims:
        et_modis_mean = et_modis.mean(dim='time')
    else:
        et_modis_mean = et_modis
    
    # Get MODIS coordinates
    # Handle potential 0-360 longitude convention
    modis_lat_name = None
    modis_lon_name = None
    for coord_name in list(ds_modis.coords) + list(ds_modis.dims):
        if 'lat' in coord_name.lower():
            modis_lat_name = coord_name
        if 'lon' in coord_name.lower():
            modis_lon_name = coord_name
    
    if modis_lat_name is None or modis_lon_name is None:
        print(f"  Could not find lat/lon. Coords: {list(ds_modis.coords)}")
        sys.exit(1)
    
    modis_lat = ds_modis[modis_lat_name].values
    modis_lon = ds_modis[modis_lon_name].values
    
    print(f"  MODIS lat range: {modis_lat.min():.2f} to {modis_lat.max():.2f}")
    print(f"  MODIS lon range: {modis_lon.min():.2f} to {modis_lon.max():.2f}")
    
    # Convert 0-360 to -180-180 if needed
    if modis_lon.max() > 180:
        print("  Converting MODIS lon from 0-360 to -180-180 ...")
        modis_lon_shifted = np.where(modis_lon > 180, modis_lon - 360, modis_lon)
        sort_idx = np.argsort(modis_lon_shifted)
        modis_lon = modis_lon_shifted[sort_idx]
        et_modis_mean = et_modis_mean.values
        if et_modis_mean.ndim == 2:
            et_modis_mean = et_modis_mean[:, sort_idx]
        else:
            print(f"  Unexpected dims for MODIS mean: {et_modis_mean.shape}")
    else:
        et_modis_mean = et_modis_mean.values
    
    # Convert MODIS ET units to mm/day if needed
    modis_units = et_modis.attrs.get('units', '')
    print(f"  MODIS ET units: '{modis_units}'")
    # MODIS ET from ILAMB is typically in kg/m2/s
    if 'kg' in modis_units and 's' in modis_units:
        print("  Converting MODIS ET from kg/m2/s to mm/day ...")
        et_modis_mean = et_modis_mean * 86400.0
    elif 'mm' in modis_units and 'd' in modis_units.lower():
        print("  MODIS ET already in mm/day")
    else:
        # Assume kg/m2/s
        print(f"  Assuming MODIS ET in kg/m2/s, converting to mm/day ...")
        et_modis_mean = et_modis_mean * 86400.0
    
    print(f"  MODIS ET mean shape: {et_modis_mean.shape}")
    print(f"  MODIS ET range: {np.nanmin(et_modis_mean):.4f} to {np.nanmax(et_modis_mean):.4f} mm/day")
    
except Exception as e:
    print(f"Error loading MODIS: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Step 4: Regrid E3SM data to MODIS grid for comparison
# ============================================================
print("Regridding E3SM ET to MODIS grid ...")

# Determine E3SM grid structure
elm_lat_vals = elm_lat.values
elm_lon_vals = elm_lon.values
et_elm_vals = et_elm_mean_mmday.values

print(f"  E3SM ET shape: {et_elm_vals.shape}")
print(f"  E3SM lat shape: {elm_lat_vals.shape}, lon shape: {elm_lon_vals.shape}")

# Handle different grid structures
if elm_lat_vals.ndim == 1 and elm_lon_vals.ndim == 1 and et_elm_vals.ndim == 2:
    # Regular grid - use xarray for regridding
    print("  E3SM appears to be on a regular grid")
    
    # Convert E3SM lon to -180-180 if needed
    if elm_lon_vals.max() > 180:
        print("  Converting E3SM lon from 0-360 to -180-180 ...")
        elm_lon_shifted = np.where(elm_lon_vals > 180, elm_lon_vals - 360, elm_lon_vals)
        sort_idx = np.argsort(elm_lon_shifted)
        elm_lon_vals = elm_lon_shifted[sort_idx]
        et_elm_vals = et_elm_vals[:, sort_idx]
    
    # Create xarray DataArrays for interpolation
    da_elm = xr.DataArray(
        et_elm_vals,
        dims=['lat', 'lon'],
        coords={'lat': elm_lat_vals, 'lon': elm_lon_vals}
    )
    
    # Interpolate to MODIS grid
    try:
        et_elm_regridded = da_elm.interp(lat=modis_lat, lon=modis_lon, method='linear').values
        print(f"  Regridded E3SM ET shape: {et_elm_regridded.shape}")
    except Exception as e:
        print(f"  Error in xarray interpolation: {e}")
        print("  Trying scipy griddata ...")
        elm_lon_2d, elm_lat_2d = np.meshgrid(elm_lon_vals, elm_lat_vals)
        points = np.column_stack([elm_lat_2d.ravel(), elm_lon_2d.ravel()])
        values = et_elm_vals.ravel()
        
        modis_lon_2d, modis_lat_2d = np.meshgrid(modis_lon, modis_lat)
        et_elm_regridded = griddata(
            points, values,
            (modis_lat_2d, modis_lon_2d),
            method='linear'
        )

elif elm_lat_vals.ndim == 1 and et_elm_vals.ndim == 1:
    # Unstructured grid
    print("  E3SM appears to be on an unstructured grid")
    
    if elm_lon_vals.max() > 180:
        elm_lon_vals = np.where(elm_lon_vals > 180, elm_lon_vals - 360, elm_lon_vals)
    
    modis_lon_2d, modis_lat_2d = np.meshgrid(modis_lon, modis_lat)
    
    valid = np.isfinite(et_elm_vals)
    et_elm_regridded = griddata(
        np.column_stack([elm_lat_vals[valid], elm_lon_vals[valid]]),
        et_elm_vals[valid],
        (modis_lat_2d, modis_lon_2d),
        method='linear'
    )
else:
    print(f"  Unexpected grid structure: lat={elm_lat_vals.shape}, lon={elm_lon_vals.shape}, data={et_elm_vals.shape}")
    # Try meshgrid approach
    if elm_lat_vals.ndim == 2:
        points = np.column_stack([elm_lat_vals.ravel(), elm_lon_vals.ravel()])
        values = et_elm_vals.ravel()
        
        if elm_lon_vals.max() > 180:
            points[:, 1] = np.where(points[:, 1] > 180, points[:, 1] - 360, points[:, 1])
        
        modis_lon_2d, modis_lat_2d = np.meshgrid(modis_lon, modis_lat)
        valid = np.isfinite(values)
        et_elm_regridded = griddata(
            points[valid], values[valid],
            (modis_lat_2d, modis_lon_2d),
            method='linear'
        )
    else:
        print("  Cannot handle this grid structure")
        sys.exit(1)

print(f"  E3SM regridded range: {np.nanmin(et_elm_regridded):.4f} to {np.nanmax(et_elm_regridded):.4f} mm/day")

# ============================================================
# Step 5: Compute spatial bias
# ============================================================
print("Computing spatial bias (E3SM - MODIS) ...")

bias = et_elm_regridded - et_modis_mean

# Create a mask where both datasets have valid data
valid_mask = np.isfinite(bias)
print(f"  Valid grid cells: {np.sum(valid_mask)} out of {valid_mask.size}")

# Compute area-weighted global mean bias
# Use cosine of latitude as area weight
modis_lon_2d_grid, modis_lat_2d_grid = np.meshgrid(modis_lon, modis_lat)
cos_lat = np.cos(np.deg2rad(modis_lat_2d_grid))

# Mask invalid values
bias_masked = np.where(valid_mask, bias, np.nan)
cos_lat_masked = np.where(valid_mask, cos_lat, 0)

global_mean_bias = np.nansum(bias_masked * cos_lat_masked) / np.nansum(cos_lat_masked)
print(f"  Global mean bias (area-weighted): {global_mean_bias:.4f} mm/day")

# Simple mean bias
simple_mean_bias = np.nanmean(bias_masked)
print(f"  Global mean bias (simple): {simple_mean_bias:.4f} mm/day")

# Spatial correlation
model_flat = et_elm_regridded[valid_mask]
obs_flat = et_modis_mean[valid_mask]

# Weight by cos(lat) for correlation
weights_flat = cos_lat[valid_mask]

# Unweighted correlation
r_unweighted, p_unweighted = pearsonr(model_flat, obs_flat)
print(f"  Spatial correlation (unweighted): r={r_unweighted:.4f}, p={p_unweighted:.2e}")

# Weighted correlation
def weighted_correlation(x, y, w):
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov_xy = np.sum(w * (x - mx) * (y - my))
    std_x = np.sqrt(np.sum(w * (x - mx)**2))
    std_y = np.sqrt(np.sum(w * (y - my)**2))
    return cov_xy / (std_x * std_y)

r_weighted = weighted_correlation(model_flat, obs_flat, weights_flat)
print(f"  Spatial correlation (area-weighted): r={r_weighted:.4f}")

# RMSE
rmse = np.sqrt(np.nanmean(bias_masked**2))
print(f"  Spatial RMSE: {rmse:.4f} mm/day")

# ============================================================
# Step 6: Save results to CSV
# ============================================================
print("Saving results ...")

results_file = os.path.join(outdir, "et_benchmark_stats.csv")
try:
    import pandas as pd
    stats = pd.DataFrame({
        'metric': [
            'global_mean_bias_mmday_area_weighted',
            'global_mean_bias_mmday_simple',
            'spatial_correlation_unweighted',
            'spatial_correlation_area_weighted',
            'spatial_rmse_mmday',
            'p_value_correlation',
            'n_valid_gridcells'
        ],
        'value': [
            global_mean_bias,
            simple_mean_bias,
            r_unweighted,
            r_weighted,
            rmse,
            p_unweighted,
            int(np.sum(valid_mask))
        ]
    })
    stats.to_csv(results_file, index=False)
    print(f"  Saved stats to {results_file}")
except Exception as e:
    print(f"  Error saving stats: {e}")

# Save bias field as NetCDF
bias_nc_file = os.path.join(outdir, "et_bias_field.nc")
try:
    ds_bias = xr.Dataset({
        'et_bias': xr.DataArray(
            bias_masked, dims=['lat', 'lon'],
            coords={'lat': modis_lat, 'lon': modis_lon},
            attrs={'units': 'mm/day', 'long_name': 'ET bias (E3SM - MODIS)'}
        ),
        'et_e3sm': xr.DataArray(
            et_elm_regridded, dims=['lat', 'lon'],
            coords={'lat': modis_lat, 'lon': modis_lon},
            attrs={'units': 'mm/day', 'long_name': 'E3SM ET climatological mean (1985-1989)'}
        ),
        'et_modis': xr.DataArray(
            et_modis_mean, dims=['lat', 'lon'],
            coords={'lat': modis_lat, 'lon': modis_lon},
            attrs={'units': 'mm/day', 'long_name': 'MODIS ET climatological mean'}
        )
    })
    ds_bias.attrs['description'] = 'E3SM vs MODIS ET benchmark'
    ds_bias.attrs['e3sm_period'] = '1985-1989'
    ds_bias.to_netcdf(bias_nc_file)
    print(f"  Saved bias field to {bias_nc_file}")
except Exception as e:
    print(f"  Error saving bias NetCDF: {e}")

# ============================================================
# Step 7: Produce maps
# ============================================================
print("Generating maps ...")

# Bias map
try:
    fig, axes = plt.subplots(3, 1, figsize=(14, 18),
                             subplot_kw={'projection': ccrs.Robinson()})
    
    # Panel 1: E3SM ET
    ax = axes[0]
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    
    vmax_et = np.nanpercentile(et_modis_mean[np.isfinite(et_modis_mean)], 95)
    im1 = ax.pcolormesh(modis_lon, modis_lat, et_elm_regridded,
                        transform=ccrs.PlateCarree(),
                        cmap='YlGnBu', vmin=0, vmax=vmax_et,
                        shading='auto')
    cb1 = plt.colorbar(im1, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cb1.set_label('ET (mm/day)')
    ax.set_title('E3SM ET Climatology (1985-1989)', fontsize=14)
    
    # Panel 2: MODIS ET
    ax = axes[1]
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    
    im2 = ax.pcolormesh(modis_lon, modis_lat, et_modis_mean,
                        transform=ccrs.PlateCarree(),
                        cmap='YlGnBu', vmin=0, vmax=vmax_et,
                        shading='auto')
    cb2 = plt.colorbar(im2, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cb2.set_label('ET (mm/day)')
    ax.set_title('MODIS ET Climatology', fontsize=14)
    
    # Panel 3: Bias
    ax = axes[2]
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    
    bias_abs_max = np.nanpercentile(np.abs(bias_masked[np.isfinite(bias_masked)]), 95)
    im3 = ax.pcolormesh(modis_lon, modis_lat, bias_masked,
                        transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', vmin=-bias_abs_max, vmax=bias_abs_max,
                        shading='auto')
    cb3 = plt.colorbar(im3, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cb3.set_label('ET Bias (mm/day)')
    ax.set_title(f'ET Bias (E3SM - MODIS)\nGlobal Mean Bias: {global_mean_bias:.3f} mm/day | '
                 f'Spatial Correlation: {r_weighted:.3f} | RMSE: {rmse:.3f} mm/day',
                 fontsize=13)
    
    plt.tight_layout()
    fig_file = os.path.join(outdir, "et_bias_map.png")
    plt.savefig(fig_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved map to {fig_file}")
except Exception as e:
    print(f"  Error creating map: {e}")
    import traceback
    traceback.print_exc()

# Scatter plot of E3SM vs MODIS
try:
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Subsample for plotting
    n_valid = len(model_flat)
    if n_valid > 10000:
        idx = np.random.choice(n_valid, 10000, replace=False)
        x_plot = obs_flat[idx]
        y_plot = model_flat[idx]
    else:
        x_plot = obs_flat
        y_plot = model_flat
    
    ax.scatter(x_plot, y_plot, alpha=0.1, s=2, c='steelblue')
    
    max_val = max(np.nanmax(x_plot), np.nanmax(y_plot)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    
    # Add regression line
    m, b = np.polyfit(obs_flat, model_flat, 1)
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, m * x_line + b, 'r-', linewidth=1.5,
            label=f'Regression (slope={m:.2f})')
    
    ax.set_xlabel('MODIS ET (mm/day)', fontsize=13)
    ax.set_ylabel('E3SM ET (mm/day)', fontsize=13)
    ax.set_title(f'E3SM vs MODIS ET\nr={r_weighted:.3f} (area-weighted), '
                 f'bias={global_mean_bias:.3f} mm/day', fontsize=14)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    scatter_file = os.path.join(outdir, "et_scatter.png")
    plt.savefig(scatter_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot to {scatter_file}")
except Exception as e:
    print(f"  Error creating scatter plot: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ET BENCHMARK SUMMARY")
print("=" * 60)
print(f"  E3SM period:                     1985-1989")
print(f"  Observation:                     MODIS ET")
print(f"  Global mean bias (area-weighted): {global_mean_bias:.4f} mm/day")
print(f"  Global mean bias (simple):        {simple_mean_bias:.4f} mm/day")
print(f"  Spatial RMSE:                     {rmse:.4f} mm/day")
print(f"  Spatial correlation (unweighted): {r_unweighted:.4f}")
print(f"  Spatial correlation (weighted):   {r_weighted:.4f}")
print(f"  Valid grid cells:                 {int(np.sum(valid_mask))}")
print(f"  Output directory:                 {outdir}")
print("=" * 60)
print("Done!")