import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from scipy.interpolate import griddata
import urllib.request

# Output directory
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_03_et_benchmark/run3_output"
os.makedirs(outdir, exist_ok=True)

# ============================================================
# 1. Fetch MODIS ET dataset from ILAMB
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
        raise
else:
    print("MODIS ET file already exists locally.")

# ============================================================
# 2. Load E3SM ELM output for 1985-1989
# ============================================================
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"

elm_files = []
for year in range(1985, 1990):
    for month in range(1, 13):
        fname = os.path.join(elm_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fname):
            elm_files.append(fname)
        else:
            print(f"Warning: missing file {fname}")

elm_files.sort()
print(f"Found {len(elm_files)} ELM files for 1985-1989.")

# Load ELM data - need QVEGE, QVEGT, QSOIL
print("Loading ELM data...")
try:
    ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', data_vars='minimal',
                                coords='minimal', compat='override')
except Exception as e:
    print(f"Error with open_mfdataset: {e}")
    print("Trying to load files individually...")
    datasets = []
    for f in elm_files:
        try:
            ds_tmp = xr.open_dataset(f)
            datasets.append(ds_tmp)
        except Exception as e2:
            print(f"  Skipping {f}: {e2}")
    ds_elm = xr.concat(datasets, dim='time')

print("ELM dataset loaded.")
print(f"  Variables available: {list(ds_elm.data_vars)[:20]}...")

# Check which ET components are available
for v in ['QVEGE', 'QVEGT', 'QSOIL']:
    if v in ds_elm:
        print(f"  {v}: shape={ds_elm[v].shape}, units={ds_elm[v].attrs.get('units','?')}")
    else:
        print(f"  WARNING: {v} not found in ELM output!")

# Compute ET = QVEGE + QVEGT + QSOIL (mm/s)
# Convert to kg/m2/s which equals mm/s for water
et_elm = ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']
et_elm.attrs['units'] = 'mm/s'
et_elm.attrs['long_name'] = 'Evapotranspiration (QVEGE+QVEGT+QSOIL)'

# Compute time-mean climatology
et_elm_mean = et_elm.mean(dim='time')

# Convert from mm/s to kg/m2/s (same numerically) then to mm/day for easier interpretation
# mm/s * 86400 = mm/day
et_elm_mean_mmday = et_elm_mean * 86400.0
print(f"E3SM ET mean range: {float(et_elm_mean_mmday.min()):.4f} to {float(et_elm_mean_mmday.max()):.4f} mm/day")

# Get ELM grid coordinates
if 'lat' in ds_elm.coords:
    elm_lat = ds_elm['lat']
    elm_lon = ds_elm['lon']
elif 'lat' in ds_elm:
    elm_lat = ds_elm['lat']
    elm_lon = ds_elm['lon']
else:
    print("Looking for coordinate variables...")
    for v in ds_elm.coords:
        print(f"  coord: {v}, shape={ds_elm[v].shape}")

print(f"ELM lat shape: {elm_lat.shape}, lon shape: {elm_lon.shape}")
print(f"ELM ET mean shape: {et_elm_mean_mmday.shape}")

# ============================================================
# 3. Load MODIS observation data
# ============================================================
print("\nLoading MODIS ET data...")
ds_modis = xr.open_dataset(modis_local)
print(f"MODIS variables: {list(ds_modis.data_vars)}")
print(f"MODIS coords: {list(ds_modis.coords)}")

# The variable inside is 'et'
modis_et = ds_modis['et']
print(f"MODIS ET shape: {modis_et.shape}")
print(f"MODIS ET units: {modis_et.attrs.get('units', 'unknown')}")

# Get MODIS time range
modis_time = ds_modis['time']
print(f"MODIS time range: {modis_time.values[0]} to {modis_time.values[-1]}")

# Select overlapping time period if possible, otherwise use full climatology
try:
    modis_et_sel = modis_et.sel(time=slice('1985-01-01', '1989-12-31'))
    if len(modis_et_sel.time) == 0:
        print("No MODIS data in 1985-1989 range. Using full MODIS climatology.")
        modis_et_sel = modis_et
    else:
        print(f"Selected MODIS data for 1985-1989: {len(modis_et_sel.time)} time steps")
except Exception:
    print("Could not subset MODIS by time. Using full climatology.")
    modis_et_sel = modis_et

# Compute time-mean
modis_et_mean = modis_et_sel.mean(dim='time')

# Check if MODIS ET needs unit conversion
# MODIS ET from ILAMB is typically in kg/m2/s
modis_units = modis_et.attrs.get('units', '')
print(f"MODIS ET units: {modis_units}")

# Convert MODIS to mm/day
# If kg/m2/s: multiply by 86400
# If already mm/day or similar, adjust accordingly
if 'kg' in modis_units.lower() or 'mm/s' in modis_units.lower() or modis_units == 'kg m-2 s-1':
    modis_et_mean_mmday = modis_et_mean * 86400.0
    print("Converted MODIS ET from kg/m2/s to mm/day")
elif 'day' in modis_units.lower():
    modis_et_mean_mmday = modis_et_mean
    print("MODIS ET already in mm/day (or similar)")
else:
    # Assume kg/m2/s
    modis_et_mean_mmday = modis_et_mean * 86400.0
    print(f"Assuming MODIS ET in kg/m2/s, converting to mm/day")

print(f"MODIS ET mean range: {float(modis_et_mean_mmday.min()):.4f} to {float(modis_et_mean_mmday.max()):.4f} mm/day")

# Get MODIS grid
modis_lat = ds_modis['lat'].values
modis_lon = ds_modis['lon'].values

# Handle 0-360 vs -180-180 longitude convention
if modis_lon.max() > 180:
    print("Converting MODIS longitude from 0-360 to -180-180")
    modis_lon_shifted = np.where(modis_lon > 180, modis_lon - 360, modis_lon)
    sort_idx = np.argsort(modis_lon_shifted)
    modis_lon = modis_lon_shifted[sort_idx]
    modis_et_mean_mmday = modis_et_mean_mmday.values[:, sort_idx] if modis_et_mean_mmday.ndim == 2 else modis_et_mean_mmday
    if isinstance(modis_et_mean_mmday, xr.DataArray):
        modis_et_mean_mmday = modis_et_mean_mmday.values[:, sort_idx]
    else:
        modis_et_mean_mmday = modis_et_mean_mmday[:, sort_idx]
    modis_et_shifted = True
else:
    modis_et_shifted = False
    if isinstance(modis_et_mean_mmday, xr.DataArray):
        modis_et_mean_mmday_vals = modis_et_mean_mmday.values
    else:
        modis_et_mean_mmday_vals = modis_et_mean_mmday

if isinstance(modis_et_mean_mmday, xr.DataArray):
    modis_et_mean_mmday_vals = modis_et_mean_mmday.values
else:
    modis_et_mean_mmday_vals = modis_et_mean_mmday

print(f"MODIS grid: lat ({modis_lat.min():.1f} to {modis_lat.max():.1f}), lon ({modis_lon.min():.1f} to {modis_lon.max():.1f})")

# ============================================================
# 4. Regrid E3SM to MODIS grid for comparison
# ============================================================
print("\nRegridding E3SM data to MODIS grid...")

# Check if ELM data is on a regular grid or unstructured
elm_lat_vals = elm_lat.values
elm_lon_vals = elm_lon.values

print(f"ELM lat dims: {elm_lat.dims}, shape: {elm_lat.shape}")
print(f"ELM lon dims: {elm_lon.dims}, shape: {elm_lon.shape}")

et_elm_vals = et_elm_mean_mmday.values
print(f"ELM ET values shape: {et_elm_vals.shape}")

# Determine if the grid is structured (2D lat/lon) or unstructured
if elm_lat.ndim == 1 and elm_lon.ndim == 1 and et_elm_vals.ndim == 2:
    # Regular grid - use xarray interp
    print("ELM data appears to be on a regular grid")
    
    # Handle longitude convention
    elm_lon_vals_adj = elm_lon_vals.copy()
    if elm_lon_vals.max() > 180:
        elm_lon_vals_adj = np.where(elm_lon_vals > 180, elm_lon_vals - 360, elm_lon_vals)
        sort_idx_elm = np.argsort(elm_lon_vals_adj)
        elm_lon_vals_adj = elm_lon_vals_adj[sort_idx_elm]
        et_elm_vals = et_elm_vals[:, sort_idx_elm]
    
    # Create xarray DataArray for interpolation
    et_elm_da = xr.DataArray(
        et_elm_vals,
        dims=['lat', 'lon'],
        coords={'lat': elm_lat_vals if elm_lon_vals.max() <= 180 else elm_lat_vals,
                'lon': elm_lon_vals_adj}
    )
    
    # Interpolate to MODIS grid
    et_elm_on_modis = et_elm_da.interp(lat=modis_lat, lon=modis_lon, method='linear')
    et_elm_regridded = et_elm_on_modis.values
    
elif elm_lat.ndim == 1 and et_elm_vals.ndim == 1:
    # Unstructured grid - use scipy griddata
    print("ELM data appears to be on an unstructured grid")
    
    elm_lon_vals_adj = elm_lon_vals.copy()
    if elm_lon_vals.max() > 180:
        elm_lon_vals_adj = np.where(elm_lon_vals > 180, elm_lon_vals - 360, elm_lon_vals)
    
    # Create target meshgrid
    modis_lon_2d, modis_lat_2d = np.meshgrid(modis_lon, modis_lat)
    
    # Remove NaN values from source
    valid = ~np.isnan(et_elm_vals)
    
    et_elm_regridded = griddata(
        (elm_lon_vals_adj[valid], elm_lat_vals[valid]),
        et_elm_vals[valid],
        (modis_lon_2d, modis_lat_2d),
        method='linear'
    )
else:
    print(f"Unexpected grid structure: lat ndim={elm_lat.ndim}, data ndim={et_elm_vals.ndim}")
    # Try treating as 2D structured grid
    if elm_lat.ndim == 2:
        elm_lon_vals_adj = elm_lon_vals.copy()
        if elm_lon_vals.max() > 180:
            elm_lon_vals_adj = np.where(elm_lon_vals > 180, elm_lon_vals - 360, elm_lon_vals)
        
        modis_lon_2d, modis_lat_2d = np.meshgrid(modis_lon, modis_lat)
        valid = ~np.isnan(et_elm_vals.ravel())
        
        et_elm_regridded = griddata(
            (elm_lon_vals_adj.ravel()[valid], elm_lat_vals.ravel()[valid]),
            et_elm_vals.ravel()[valid],
            (modis_lon_2d, modis_lat_2d),
            method='linear'
        )

print(f"Regridded E3SM ET shape: {et_elm_regridded.shape}")
print(f"MODIS ET shape: {modis_et_mean_mmday_vals.shape}")

# ============================================================
# 5. Compute bias and statistics
# ============================================================
print("\nComputing bias statistics...")

bias = et_elm_regridded - modis_et_mean_mmday_vals

# Create lat weights for area-weighted statistics
lat_weights = np.cos(np.deg2rad(modis_lat))
lat_weights_2d = np.tile(lat_weights[:, np.newaxis], (1, len(modis_lon)))

# Mask where either is NaN
valid_mask = ~np.isnan(bias) & ~np.isnan(modis_et_mean_mmday_vals) & ~np.isnan(et_elm_regridded)

# Global mean bias (area-weighted)
weights_valid = np.where(valid_mask, lat_weights_2d, 0)
if weights_valid.sum() > 0:
    global_mean_bias = np.nansum(bias * weights_valid) / np.nansum(weights_valid)
else:
    global_mean_bias = np.nanmean(bias)

print(f"Global mean bias (E3SM - MODIS): {global_mean_bias:.4f} mm/day")

# Simple global mean bias (unweighted)
simple_mean_bias = np.nanmean(bias)
print(f"Simple mean bias: {simple_mean_bias:.4f} mm/day")

# Global mean values
elm_global_mean = np.nansum(et_elm_regridded * weights_valid) / np.nansum(weights_valid)
modis_global_mean = np.nansum(modis_et_mean_mmday_vals * weights_valid) / np.nansum(weights_valid)
print(f"E3SM global mean ET: {elm_global_mean:.4f} mm/day")
print(f"MODIS global mean ET: {modis_global_mean:.4f} mm/day")

# Spatial correlation
elm_flat = et_elm_regridded[valid_mask]
modis_flat = modis_et_mean_mmday_vals[valid_mask]
spatial_corr, p_value = stats.pearsonr(elm_flat, modis_flat)
print(f"Spatial correlation (Pearson r): {spatial_corr:.4f} (p-value: {p_value:.2e})")

# RMSE
rmse = np.sqrt(np.nanmean(bias[valid_mask]**2))
print(f"Spatial RMSE: {rmse:.4f} mm/day")

# Relative bias
rel_bias = global_mean_bias / modis_global_mean * 100
print(f"Relative bias: {rel_bias:.2f}%")

# ============================================================
# 6. Save statistics to CSV
# ============================================================
try:
    stats_file = os.path.join(outdir, "et_benchmark_statistics.csv")
    with open(stats_file, 'w') as f:
        f.write("metric,value,units\n")
        f.write(f"global_mean_bias_area_weighted,{global_mean_bias:.6f},mm/day\n")
        f.write(f"global_mean_bias_simple,{simple_mean_bias:.6f},mm/day\n")
        f.write(f"e3sm_global_mean_et,{elm_global_mean:.6f},mm/day\n")
        f.write(f"modis_global_mean_et,{modis_global_mean:.6f},mm/day\n")
        f.write(f"spatial_correlation,{spatial_corr:.6f},dimensionless\n")
        f.write(f"spatial_correlation_pvalue,{p_value:.6e},dimensionless\n")
        f.write(f"spatial_rmse,{rmse:.6f},mm/day\n")
        f.write(f"relative_bias,{rel_bias:.4f},percent\n")
        f.write(f"n_valid_gridcells,{int(valid_mask.sum())},count\n")
    print(f"\nStatistics saved to {stats_file}")
except Exception as e:
    print(f"Error saving statistics CSV: {e}")

# ============================================================
# 7. Save bias field as NetCDF
# ============================================================
try:
    bias_ds = xr.Dataset({
        'et_bias': (['lat', 'lon'], bias, {'units': 'mm/day', 'long_name': 'ET bias (E3SM - MODIS)'}),
        'et_e3sm': (['lat', 'lon'], et_elm_regridded, {'units': 'mm/day', 'long_name': 'E3SM ET climatology'}),
        'et_modis': (['lat', 'lon'], modis_et_mean_mmday_vals, {'units': 'mm/day', 'long_name': 'MODIS ET climatology'}),
    }, coords={
        'lat': ('lat', modis_lat, {'units': 'degrees_north'}),
        'lon': ('lon', modis_lon, {'units': 'degrees_east'}),
    })
    bias_ds.attrs['description'] = 'E3SM vs MODIS ET benchmark: 5-year climatology (1985-1989)'
    bias_ds.attrs['global_mean_bias_mm_per_day'] = global_mean_bias
    bias_ds.attrs['spatial_correlation'] = spatial_corr
    
    nc_file = os.path.join(outdir, "et_bias_field.nc")
    bias_ds.to_netcdf(nc_file)
    print(f"Bias field saved to {nc_file}")
except Exception as e:
    print(f"Error saving NetCDF: {e}")

# ============================================================
# 8. Produce maps
# ============================================================
print("\nGenerating maps...")

try:
    fig = plt.figure(figsize=(18, 14))
    
    # --- E3SM ET map ---
    ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.Robinson())
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    vmax_et = max(np.nanpercentile(et_elm_regridded, 95), np.nanpercentile(modis_et_mean_mmday_vals, 95))
    
    lon_plot, lat_plot = np.meshgrid(modis_lon, modis_lat)
    im1 = ax1.pcolormesh(lon_plot, lat_plot, et_elm_regridded,
                          transform=ccrs.PlateCarree(),
                          cmap='YlGnBu', vmin=0, vmax=vmax_et, shading='auto')
    cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7)
    cb1.set_label('ET (mm/day)')
    ax1.set_title(f'E3SM ET Climatology (1985-1989)\nGlobal Mean: {elm_global_mean:.3f} mm/day')
    
    # --- MODIS ET map ---
    ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.Robinson())
    ax2.set_global()
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    im2 = ax2.pcolormesh(lon_plot, lat_plot, modis_et_mean_mmday_vals,
                          transform=ccrs.PlateCarree(),
                          cmap='YlGnBu', vmin=0, vmax=vmax_et, shading='auto')
    cb2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7)
    cb2.set_label('ET (mm/day)')
    ax2.set_title(f'MODIS ET Climatology\nGlobal Mean: {modis_global_mean:.3f} mm/day')
    
    # --- Bias map ---
    ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.Robinson())
    ax3.set_global()
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    bias_limit = np.nanpercentile(np.abs(bias[valid_mask]), 95)
    
    im3 = ax3.pcolormesh(lon_plot, lat_plot, bias,
                          transform=ccrs.PlateCarree(),
                          cmap='RdBu_r', vmin=-bias_limit, vmax=bias_limit, shading='auto')
    cb3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7)
    cb3.set_label('ET Bias (mm/day)')
    ax3.set_title(f'ET Bias (E3SM - MODIS)\nMean Bias: {global_mean_bias:.3f} mm/day | Correlation: {spatial_corr:.3f} | RMSE: {rmse:.3f} mm/day')
    
    plt.tight_layout()
    map_file = os.path.join(outdir, "et_benchmark_maps.png")
    plt.savefig(map_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Maps saved to {map_file}")
except Exception as e:
    print(f"Error generating maps: {e}")
    import traceback
    traceback.print_exc()

# Also make a standalone bias-only map
try:
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
    
    im = ax.pcolormesh(lon_plot, lat_plot, bias,
                        transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', vmin=-bias_limit, vmax=bias_limit, shading='auto')
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
    cb.set_label('ET Bias (mm/day)')
    ax.set_title(f'ET Bias: E3SM minus MODIS (1985-1989 Climatology)\n'
                 f'Global Mean Bias: {global_mean_bias:.3f} mm/day | '
                 f'Spatial Correlation: {spatial_corr:.3f} | RMSE: {rmse:.3f} mm/day')
    
    plt.tight_layout()
    bias_map_file = os.path.join(outdir, "et_bias_map.png")
    plt.savefig(bias_map_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bias map saved to {bias_map_file}")
except Exception as e:
    print(f"Error generating bias map: {e}")
    import traceback
    traceback.print_exc()

# Scatter plot
try:
    fig3, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Subsample for plotting
    n_points = min(10000, len(elm_flat))
    idx = np.random.choice(len(elm_flat), n_points, replace=False)
    
    ax.scatter(modis_flat[idx], elm_flat[idx], alpha=0.1, s=2, color='steelblue')
    
    max_val = max(np.nanpercentile(elm_flat, 99), np.nanpercentile(modis_flat, 99))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    
    # Linear fit
    slope, intercept, r_val, p_val, std_err = stats.linregress(modis_flat[idx], elm_flat[idx])
    x_fit = np.linspace(0, max_val, 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1,
            label=f'Linear fit (r={r_val:.3f})')
    
    ax.set_xlabel('MODIS ET (mm/day)')
    ax.set_ylabel('E3SM ET (mm/day)')
    ax.set_title(f'E3SM vs MODIS ET Scatter\nr={spatial_corr:.3f}, bias={global_mean_bias:.3f} mm/day')
    ax.legend()
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    scatter_file = os.path.join(outdir, "et_scatter_plot.png")
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {scatter_file}")
except Exception as e:
    print(f"Error generating scatter plot: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("ET BENCHMARK SUMMARY")
print("="*60)
print(f"Period: 1985-1989 climatology")
print(f"E3SM global mean ET:  {elm_global_mean:.4f} mm/day")
print(f"MODIS global mean ET: {modis_global_mean:.4f} mm/day")
print(f"Global mean bias:     {global_mean_bias:.4f} mm/day ({rel_bias:.2f}%)")
print(f"Spatial correlation:  {spatial_corr:.4f}")
print(f"Spatial RMSE:         {rmse:.4f} mm/day")
print(f"Valid grid cells:     {int(valid_mask.sum())}")
print(f"\nAll outputs saved to: {outdir}")
print("="*60)