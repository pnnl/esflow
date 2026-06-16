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
from scipy import stats
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Output directory
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_03_et_benchmark/run2_output"
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
    print("MODIS ET file already exists locally.")

# ============================================================
# Step 2: Load E3SM ELM output for 1985-1989
# ============================================================
print("Loading E3SM ELM output...")
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"

elm_files = []
for year in range(1985, 1990):
    for month in range(1, 13):
        fname = os.path.join(elm_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fname):
            elm_files.append(fname)
        else:
            print(f"  Warning: missing file {fname}")

elm_files.sort()
print(f"  Found {len(elm_files)} ELM files.")

if len(elm_files) == 0:
    print("ERROR: No ELM files found. Exiting.")
    sys.exit(1)

try:
    ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
    print(f"  ELM dataset dimensions: {dict(ds_elm.dims)}")
    print(f"  ELM dataset variables: {list(ds_elm.data_vars)[:20]}")
except Exception as e:
    print(f"Error loading ELM files: {e}")
    # Try with decode_times=False
    try:
        ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=False)
        print(f"  Loaded with decode_times=False. Dims: {dict(ds_elm.dims)}")
    except Exception as e2:
        print(f"Still failed: {e2}")
        sys.exit(1)

# Extract ET components and compute total ET
print("Computing E3SM ET = QVEGE + QVEGT + QSOIL ...")
try:
    qvege = ds_elm['QVEGE']  # mm/s
    qvegt = ds_elm['QVEGT']  # mm/s
    qsoil = ds_elm['QSOIL']  # mm/s
    et_elm = qvege + qvegt + qsoil  # mm/s
    et_elm.name = 'ET'
    print(f"  ET shape: {et_elm.shape}")
except Exception as e:
    print(f"Error computing ET: {e}")
    print(f"  Available variables: {list(ds_elm.data_vars)}")
    sys.exit(1)

# Compute time-mean climatology (1985-1989)
print("Computing E3SM time-mean ET...")
et_elm_mean = et_elm.mean(dim='time')

# Convert from mm/s to kg/m2/s (they are the same since 1 mm/s water = 1 kg/m2/s)
# But we'll convert to a common unit. Let's use kg/m2/s for comparison.
# mm/s = kg/m2/s (density of water = 1000 kg/m3, 1mm = 0.001m, so flux = 0.001 * 1000 = 1 kg/m2/s per mm/s)
# Actually mm/s: 1 mm/s of water depth per second = 1e-3 m/s * 1000 kg/m3 = 1 kg/m2/s
# So mm/s IS kg/m2/s

# Get lat/lon from ELM
elm_lat = ds_elm['lat'].values
elm_lon = ds_elm['lon'].values
print(f"  ELM lat shape: {elm_lat.shape}, lon shape: {elm_lon.shape}")

# ============================================================
# Step 3: Load MODIS ET observation
# ============================================================
print("Loading MODIS ET observations...")
try:
    ds_modis = xr.open_dataset(modis_local)
    print(f"  MODIS dataset dimensions: {dict(ds_modis.dims)}")
    print(f"  MODIS dataset variables: {list(ds_modis.data_vars)}")
    print(f"  MODIS coordinate names: {list(ds_modis.coords)}")
except Exception as e:
    print(f"Error loading MODIS dataset: {e}")
    sys.exit(1)

# Extract ET variable
try:
    et_modis = ds_modis['et']
    print(f"  MODIS ET shape: {et_modis.shape}")
    if hasattr(et_modis, 'units'):
        print(f"  MODIS ET units: {et_modis.units}")
    if hasattr(et_modis, 'long_name'):
        print(f"  MODIS ET long_name: {et_modis.long_name}")
except Exception as e:
    print(f"Error extracting MODIS ET variable: {e}")
    # Try other variable names
    for vname in ds_modis.data_vars:
        print(f"  Variable: {vname}, shape: {ds_modis[vname].shape}")
    sys.exit(1)

# Compute time-mean of MODIS ET
# Check if there is time overlap with 1985-1989
print("  MODIS time range:", end=" ")
try:
    modis_times = ds_modis['time'].values
    print(f"{modis_times[0]} to {modis_times[-1]}")
except:
    print("Could not determine time range")

# Just compute full time mean of MODIS (it's a climatological dataset)
et_modis_mean = et_modis.mean(dim='time')
print(f"  MODIS mean ET shape: {et_modis_mean.shape}")

# Determine MODIS lon/lat
modis_lon_name = None
modis_lat_name = None
for cname in ds_modis.coords:
    if 'lon' in cname.lower():
        modis_lon_name = cname
    if 'lat' in cname.lower():
        modis_lat_name = cname

if modis_lon_name is None or modis_lat_name is None:
    # Try dimension names
    for dname in et_modis_mean.dims:
        if 'lon' in dname.lower():
            modis_lon_name = dname
        if 'lat' in dname.lower():
            modis_lat_name = dname

print(f"  MODIS lon variable: {modis_lon_name}, lat variable: {modis_lat_name}")
modis_lon = ds_modis[modis_lon_name].values
modis_lat = ds_modis[modis_lat_name].values
print(f"  MODIS lon range: [{modis_lon.min():.1f}, {modis_lon.max():.1f}], lat range: [{modis_lat.min():.1f}, {modis_lat.max():.1f}]")

# ============================================================
# Step 4: Unit conversion
# ============================================================
# MODIS ET units: likely kg/m2/s or mm/day or similar
# E3SM ET units: mm/s = kg/m2/s
# We need to check and harmonize
modis_et_units = getattr(et_modis, 'units', 'unknown')
print(f"\nUnit conversion:")
print(f"  E3SM ET units: mm/s (= kg/m2/s)")
print(f"  MODIS ET units: {modis_et_units}")

et_modis_mean_vals = et_modis_mean.values.copy()
et_elm_mean_vals = et_elm_mean.values.copy()

# Convert E3SM from mm/s to kg/m2/s (they are equivalent)
# Convert both to common units: kg/m2/s
# If MODIS is in kg/m2/s already, great
# If MODIS is in g/m2/day, need to convert: g/m2/day -> kg/m2/s: divide by 1e3 * 86400
# If MODIS is in mm/day, convert: mm/day -> kg/m2/s: divide by 86400

if 'g' in str(modis_et_units).lower() and 'day' in str(modis_et_units).lower():
    print("  Converting MODIS from g/m2/day to kg/m2/s")
    et_modis_mean_vals = et_modis_mean_vals / (1e3 * 86400.0)
elif 'mm' in str(modis_et_units).lower() and 'day' in str(modis_et_units).lower():
    print("  Converting MODIS from mm/day to kg/m2/s")
    et_modis_mean_vals = et_modis_mean_vals / 86400.0
elif 'kg' in str(modis_et_units).lower() and 's' in str(modis_et_units).lower():
    print("  MODIS already in kg/m2/s, no conversion needed")
else:
    print(f"  Unknown MODIS units '{modis_et_units}', assuming kg/m2/s")

# For display, convert everything to mm/day
# 1 kg/m2/s = 86400 mm/day
elm_mmday = et_elm_mean_vals * 86400.0  # mm/s -> mm/day
modis_mmday = et_modis_mean_vals * 86400.0  # kg/m2/s -> mm/day

print(f"  E3SM ET range (mm/day): [{np.nanmin(elm_mmday):.4f}, {np.nanmax(elm_mmday):.4f}]")
print(f"  MODIS ET range (mm/day): [{np.nanmin(modis_mmday):.4f}, {np.nanmax(modis_mmday):.4f}]")

# ============================================================
# Step 5: Regrid E3SM to MODIS grid
# ============================================================
print("\nRegridding E3SM to MODIS grid...")

# Determine if ELM data is on a regular grid or unstructured
is_regular = len(elm_lat.shape) == 1 and len(elm_lon.shape) == 1
print(f"  ELM grid is {'regular' if is_regular else 'unstructured'}")

# Ensure consistent longitude convention
# Convert MODIS from 0-360 to -180 to 180 if needed
if modis_lon.max() > 180:
    print("  Converting MODIS lon from 0-360 to -180-180")
    modis_lon_converted = np.where(modis_lon > 180, modis_lon - 360, modis_lon)
    sort_idx = np.argsort(modis_lon_converted)
    modis_lon_converted = modis_lon_converted[sort_idx]
    modis_mmday = modis_mmday[:, sort_idx] if modis_mmday.ndim == 2 else modis_mmday
    modis_lon = modis_lon_converted

# Convert ELM lon to -180 to 180 if needed
if is_regular:
    if elm_lon.max() > 180:
        print("  Converting ELM lon from 0-360 to -180-180")
        elm_lon_converted = np.where(elm_lon > 180, elm_lon - 360, elm_lon)
        sort_idx = np.argsort(elm_lon_converted)
        elm_lon_converted = elm_lon_converted[sort_idx]
        elm_mmday = elm_mmday[:, sort_idx] if elm_mmday.ndim == 2 else elm_mmday
        elm_lon = elm_lon_converted
    
    # Use xarray for regridding via interpolation
    elm_da = xr.DataArray(
        elm_mmday,
        dims=['lat', 'lon'],
        coords={'lat': elm_lat, 'lon': elm_lon}
    )
    
    # Interpolate ELM onto MODIS grid
    try:
        elm_on_modis = elm_da.interp(lat=modis_lat, lon=modis_lon, method='linear')
        elm_regridded = elm_on_modis.values
        print(f"  Regridded E3SM shape: {elm_regridded.shape}")
    except Exception as e:
        print(f"  xarray interp failed: {e}, trying scipy griddata")
        elm_lon2d, elm_lat2d = np.meshgrid(elm_lon, elm_lat)
        modis_lon2d, modis_lat2d = np.meshgrid(modis_lon, modis_lat)
        
        valid = np.isfinite(elm_mmday)
        points = np.column_stack([elm_lat2d[valid], elm_lon2d[valid]])
        values = elm_mmday[valid]
        
        elm_regridded = griddata(
            points, values,
            (modis_lat2d, modis_lon2d),
            method='linear'
        )
        print(f"  Regridded E3SM shape: {elm_regridded.shape}")
else:
    # Unstructured grid - use griddata
    if elm_lon.max() > 180:
        elm_lon = np.where(elm_lon > 180, elm_lon - 360, elm_lon)
    
    modis_lon2d, modis_lat2d = np.meshgrid(modis_lon, modis_lat)
    
    valid = np.isfinite(elm_mmday.ravel())
    if elm_lat.shape == elm_lon.shape:
        points = np.column_stack([elm_lat.ravel()[valid], elm_lon.ravel()[valid]])
    else:
        elm_lon2d_src, elm_lat2d_src = np.meshgrid(elm_lon, elm_lat)
        points = np.column_stack([elm_lat2d_src.ravel()[valid], elm_lon2d_src.ravel()[valid]])
    
    values = elm_mmday.ravel()[valid]
    
    elm_regridded = griddata(
        points, values,
        (modis_lat2d, modis_lon2d),
        method='linear'
    )
    print(f"  Regridded E3SM shape: {elm_regridded.shape}")

# ============================================================
# Step 6: Compute bias and statistics
# ============================================================
print("\nComputing bias and statistics...")

bias = elm_regridded - modis_mmday

# Create weight array based on latitude (area weighting)
modis_lon2d, modis_lat2d = np.meshgrid(modis_lon, modis_lat)
weights = np.cos(np.deg2rad(modis_lat2d))

# Mask where either is NaN
valid_mask = np.isfinite(bias) & np.isfinite(modis_mmday) & np.isfinite(elm_regridded)
bias_masked = np.where(valid_mask, bias, np.nan)
weights_masked = np.where(valid_mask, weights, 0)

# Global mean bias (area-weighted)
if np.sum(weights_masked) > 0:
    global_mean_bias = np.nansum(bias_masked * weights_masked) / np.nansum(weights_masked)
else:
    global_mean_bias = np.nanmean(bias_masked)

# Global mean model and obs
global_mean_model = np.nansum(np.where(valid_mask, elm_regridded, 0) * weights_masked) / np.nansum(weights_masked)
global_mean_obs = np.nansum(np.where(valid_mask, modis_mmday, 0) * weights_masked) / np.nansum(weights_masked)

# Spatial correlation
valid_pts = valid_mask.ravel()
if np.sum(valid_pts) > 10:
    r_value, p_value = stats.pearsonr(
        elm_regridded.ravel()[valid_pts],
        modis_mmday.ravel()[valid_pts]
    )
else:
    r_value, p_value = np.nan, np.nan

# RMSE
rmse = np.sqrt(np.nansum(bias_masked**2 * weights_masked) / np.nansum(weights_masked))

# Relative bias
if global_mean_obs != 0:
    relative_bias = (global_mean_bias / global_mean_obs) * 100
else:
    relative_bias = np.nan

print(f"  Global mean E3SM ET:    {global_mean_model:.4f} mm/day")
print(f"  Global mean MODIS ET:   {global_mean_obs:.4f} mm/day")
print(f"  Global mean bias:       {global_mean_bias:.4f} mm/day")
print(f"  Relative bias:          {relative_bias:.2f} %")
print(f"  Spatial correlation:    {r_value:.4f} (p={p_value:.2e})")
print(f"  RMSE:                   {rmse:.4f} mm/day")
print(f"  Valid grid cells:       {np.sum(valid_mask)}")

# ============================================================
# Step 7: Save statistics to CSV
# ============================================================
stats_file = os.path.join(outdir, "et_benchmark_statistics.csv")
try:
    with open(stats_file, 'w') as f:
        f.write("metric,value,unit\n")
        f.write(f"global_mean_model_ET,{global_mean_model:.6f},mm/day\n")
        f.write(f"global_mean_obs_ET,{global_mean_obs:.6f},mm/day\n")
        f.write(f"global_mean_bias,{global_mean_bias:.6f},mm/day\n")
        f.write(f"relative_bias,{relative_bias:.2f},%\n")
        f.write(f"spatial_correlation,{r_value:.6f},dimensionless\n")
        f.write(f"correlation_p_value,{p_value:.2e},dimensionless\n")
        f.write(f"RMSE,{rmse:.6f},mm/day\n")
        f.write(f"valid_grid_cells,{int(np.sum(valid_mask))},count\n")
        f.write(f"model_period,1985-1989,years\n")
        f.write(f"obs_dataset,MODIS,name\n")
    print(f"\nStatistics saved to {stats_file}")
except Exception as e:
    print(f"Error saving statistics: {e}")

# ============================================================
# Step 8: Save bias field as NetCDF
# ============================================================
bias_nc_file = os.path.join(outdir, "et_bias_field.nc")
try:
    ds_out = xr.Dataset({
        'bias': (['lat', 'lon'], bias_masked),
        'model_et': (['lat', 'lon'], elm_regridded),
        'obs_et': (['lat', 'lon'], modis_mmday),
    }, coords={
        'lat': modis_lat,
        'lon': modis_lon,
    })
    ds_out['bias'].attrs = {'units': 'mm/day', 'long_name': 'ET bias (E3SM - MODIS)'}
    ds_out['model_et'].attrs = {'units': 'mm/day', 'long_name': 'E3SM ET climatology 1985-1989'}
    ds_out['obs_et'].attrs = {'units': 'mm/day', 'long_name': 'MODIS ET climatology'}
    ds_out.attrs['description'] = 'ET benchmark: E3SM vs MODIS'
    ds_out.to_netcdf(bias_nc_file)
    print(f"Bias field saved to {bias_nc_file}")
except Exception as e:
    print(f"Error saving bias NetCDF: {e}")

# ============================================================
# Step 9: Create maps
# ============================================================
print("\nCreating maps...")

try:
    fig, axes = plt.subplots(3, 1, figsize=(14, 18),
                              subplot_kw={'projection': ccrs.Robinson()})
    
    # Determine common color range for model and obs
    vmin_et = 0
    vmax_et = max(np.nanpercentile(elm_regridded[valid_mask], 95),
                  np.nanpercentile(modis_mmday[valid_mask], 95))
    
    # Panel 1: E3SM ET
    ax = axes[0]
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    im1 = ax.pcolormesh(modis_lon, modis_lat, elm_regridded,
                         transform=ccrs.PlateCarree(),
                         cmap='YlGnBu', vmin=vmin_et, vmax=vmax_et,
                         shading='auto')
    cb1 = plt.colorbar(im1, ax=ax, orientation='horizontal', pad=0.05,
                        shrink=0.7, label='ET (mm/day)')
    ax.set_title(f'E3SM ET Climatology (1985-1989)\nGlobal Mean: {global_mean_model:.3f} mm/day',
                 fontsize=13)
    
    # Panel 2: MODIS ET
    ax = axes[1]
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    im2 = ax.pcolormesh(modis_lon, modis_lat, modis_mmday,
                         transform=ccrs.PlateCarree(),
                         cmap='YlGnBu', vmin=vmin_et, vmax=vmax_et,
                         shading='auto')
    cb2 = plt.colorbar(im2, ax=ax, orientation='horizontal', pad=0.05,
                        shrink=0.7, label='ET (mm/day)')
    ax.set_title(f'MODIS ET Climatology\nGlobal Mean: {global_mean_obs:.3f} mm/day',
                 fontsize=13)
    
    # Panel 3: Bias
    ax = axes[2]
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    vmax_bias = np.nanpercentile(np.abs(bias_masked[valid_mask]), 95)
    im3 = ax.pcolormesh(modis_lon, modis_lat, bias_masked,
                         transform=ccrs.PlateCarree(),
                         cmap='RdBu_r', vmin=-vmax_bias, vmax=vmax_bias,
                         shading='auto')
    cb3 = plt.colorbar(im3, ax=ax, orientation='horizontal', pad=0.05,
                        shrink=0.7, label='Bias (mm/day)')
    ax.set_title(f'ET Bias (E3SM − MODIS)\nMean Bias: {global_mean_bias:.3f} mm/day | '
                 f'Spatial Corr: {r_value:.3f} | RMSE: {rmse:.3f} mm/day',
                 fontsize=13)
    
    plt.tight_layout()
    map_file = os.path.join(outdir, "et_bias_map.png")
    plt.savefig(map_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Map saved to {map_file}")
except Exception as e:
    print(f"Error creating maps: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Step 10: Create scatter plot
# ============================================================
try:
    fig, ax = plt.subplots(figsize=(8, 8))
    
    valid_pts = valid_mask.ravel()
    model_vals = elm_regridded.ravel()[valid_pts]
    obs_vals = modis_mmday.ravel()[valid_pts]
    
    # Subsample for plotting if too many points
    n_pts = len(model_vals)
    if n_pts > 50000:
        idx = np.random.choice(n_pts, 50000, replace=False)
        model_plot = model_vals[idx]
        obs_plot = obs_vals[idx]
    else:
        model_plot = model_vals
        obs_plot = obs_vals
    
    ax.scatter(obs_plot, model_plot, s=1, alpha=0.1, c='steelblue')
    
    # 1:1 line
    max_val = max(np.nanmax(model_plot), np.nanmax(obs_plot))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    
    # Linear fit
    slope, intercept, r_val, p_val, std_err = stats.linregress(obs_vals, model_vals)
    x_fit = np.linspace(0, max_val, 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5,
            label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    
    ax.set_xlabel('MODIS ET (mm/day)', fontsize=12)
    ax.set_ylabel('E3SM ET (mm/day)', fontsize=12)
    ax.set_title(f'ET Scatter: E3SM vs MODIS\nr={r_value:.3f}, bias={global_mean_bias:.3f} mm/day',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    scatter_file = os.path.join(outdir, "et_scatter_plot.png")
    plt.savefig(scatter_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {scatter_file}")
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ET BENCHMARK SUMMARY")
print("=" * 60)
print(f"  Model:              E3SM v3 LR Historical")
print(f"  Period:             1985-1989 (5-year climatology)")
print(f"  Observation:        MODIS satellite ET")
print(f"  Global Mean Model:  {global_mean_model:.4f} mm/day")
print(f"  Global Mean Obs:    {global_mean_obs:.4f} mm/day")
print(f"  Global Mean Bias:   {global_mean_bias:.4f} mm/day ({relative_bias:.1f}%)")
print(f"  Spatial Correlation:{r_value:.4f}")
print(f"  RMSE:               {rmse:.4f} mm/day")
print("=" * 60)
print(f"\nAll outputs saved to: {outdir}")
print("  - et_benchmark_statistics.csv")
print("  - et_bias_field.nc")
print("  - et_bias_map.png")
print("  - et_scatter_plot.png")
print("\nDone.")