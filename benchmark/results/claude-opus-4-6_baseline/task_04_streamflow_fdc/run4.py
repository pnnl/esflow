import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import wasserstein_distance

# Output directory
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_04_streamflow_fdc/run4_output"
os.makedirs(outdir, exist_ok=True)

# Paths
case_name = "sample.v3.LR.historical"
rof_dir = "./data/sample/e3sm/rof"
gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
obs_dir = "./data/sample/obs/streamflow"

# ─── 1. Load gauge metadata ────────────────────────────────────────────────
try:
    gauge_meta = pd.read_csv(gauge_meta_path)
    print(f"Loaded gauge metadata: {len(gauge_meta)} gauges")
    print(gauge_meta.head())
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    raise

# ─── 2. Load MOSART daily files for 1985-1989 ──────────────────────────────
daily_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h1.*.nc")
all_daily_files = sorted(glob.glob(daily_pattern))
print(f"Found {len(all_daily_files)} daily MOSART files total")

# Filter to 1985-1989
daily_files_period = []
for f in all_daily_files:
    basename = os.path.basename(f)
    # Extract date part: sample.v3.LR.historical.mosart.h1.YYYY-MM-DD-00000.nc
    parts = basename.replace(".nc", "").split(".")
    date_part = parts[-1]  # e.g., "1985-01-02-00000"
    year = int(date_part.split("-")[0])
    if 1985 <= year <= 1989:
        daily_files_period.append(f)

print(f"Found {len(daily_files_period)} daily MOSART files for 1985-1989")

# If no daily files, try monthly
use_monthly = False
if len(daily_files_period) == 0:
    print("No daily files found for 1985-1989. Trying monthly files...")
    monthly_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h0.*.nc")
    all_monthly_files = sorted(glob.glob(monthly_pattern))
    monthly_files_period = []
    for f in all_monthly_files:
        basename = os.path.basename(f)
        parts = basename.replace(".nc", "").split(".")
        date_part = parts[-1]  # e.g., "1985-01"
        year = int(date_part.split("-")[0])
        if 1985 <= year <= 1989:
            monthly_files_period.append(f)
    print(f"Found {len(monthly_files_period)} monthly MOSART files for 1985-1989")
    if len(monthly_files_period) == 0:
        raise FileNotFoundError("No MOSART files found for 1985-1989")
    daily_files_period = monthly_files_period
    use_monthly = True

# Load a sample file to get grid info
try:
    ds_sample = xr.open_dataset(daily_files_period[0])
    print(f"Sample file variables: {list(ds_sample.data_vars)}")
    print(f"Sample file dims: {dict(ds_sample.dims)}")
    print(f"Sample file coords: {list(ds_sample.coords)}")
    
    # Get grid coordinates
    if 'lat' in ds_sample:
        grid_lat = ds_sample['lat'].values
        grid_lon = ds_sample['lon'].values
    elif 'latlat' in ds_sample:
        grid_lat = ds_sample['latlat'].values
        grid_lon = ds_sample['lonlon'].values
    else:
        # Check for coordinate variables
        for vname in ds_sample.data_vars:
            if 'lat' in vname.lower():
                print(f"  Potential lat var: {vname}")
            if 'lon' in vname.lower():
                print(f"  Potential lon var: {vname}")
        grid_lat = ds_sample['lat'].values
        grid_lon = ds_sample['lon'].values
    
    print(f"Grid shape: lat={grid_lat.shape}, lon={grid_lon.shape}")
    ds_sample.close()
except Exception as e:
    print(f"Error loading sample file: {e}")
    raise

# ─── 3. Match gauges to model grid ─────────────────────────────────────────
def find_nearest_grid_index(glat, glon, grid_lat, grid_lon):
    """Find the nearest grid cell index for a given lat/lon."""
    # Handle case where grid is 1D (regular grid) vs unstructured
    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        # Could be structured or unstructured with single dim
        if grid_lat.shape == grid_lon.shape:
            # Unstructured grid - single dimension
            dist = np.sqrt((grid_lat - glat)**2 + (grid_lon - glon)**2)
            idx = np.argmin(dist)
            return idx
        else:
            # 2D regular grid
            lat_idx = np.argmin(np.abs(grid_lat - glat))
            lon_idx = np.argmin(np.abs(grid_lon - glon))
            return (lat_idx, lon_idx)
    elif grid_lat.ndim == 2:
        dist = np.sqrt((grid_lat - glat)**2 + (grid_lon - glon)**2)
        idx = np.unravel_index(np.argmin(dist), grid_lat.shape)
        return idx
    else:
        # Assume 1D unstructured
        dist = np.sqrt((grid_lat - glat)**2 + (grid_lon - glon)**2)
        idx = np.argmin(dist)
        return idx

gauge_grid_indices = {}
for _, row in gauge_meta.iterrows():
    gid = str(int(row['gauge_id']))
    glat = row['lat']
    glon = row['lon']
    # Adjust lon to match model convention if needed
    glon_model = glon
    if grid_lon.max() > 180 and glon < 0:
        glon_model = glon + 360
    elif grid_lon.min() < 0 and glon > 180:
        glon_model = glon - 360
    
    idx = find_nearest_grid_index(glat, glon_model, grid_lat, grid_lon)
    gauge_grid_indices[gid] = {'idx': idx, 'lat': glat, 'lon': glon, 'lon_model': glon_model}
    
    # Report distance
    if isinstance(idx, tuple):
        matched_lat = grid_lat[idx[0]]
        matched_lon = grid_lon[idx[1]]
    else:
        matched_lat = grid_lat[idx]
        matched_lon = grid_lon[idx]
    dist_deg = np.sqrt((matched_lat - glat)**2 + (matched_lon - glon_model)**2)
    print(f"  Gauge {gid}: ({glat:.2f}, {glon:.2f}) -> grid idx {idx}, dist={dist_deg:.3f} deg")

# ─── 4. Extract model discharge time series for all gauges ──────────────────
print("\nLoading model discharge data...")
var_name = "RIVER_DISCHARGE_OVER_LAND_LIQ"

# Load all files at once using open_mfdataset
try:
    ds = xr.open_mfdataset(daily_files_period, combine='nested', concat_dim='time',
                           data_vars='minimal', coords='minimal', compat='override')
    print(f"Combined dataset time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"Number of time steps: {len(ds.time)}")
except Exception as e:
    print(f"Error with open_mfdataset: {e}")
    print("Trying to load files one by one...")
    datasets = []
    for f in daily_files_period:
        try:
            d = xr.open_dataset(f)
            datasets.append(d)
        except Exception as ex:
            print(f"  Skipping {f}: {ex}")
    ds = xr.concat(datasets, dim='time')

# Extract discharge for each gauge
model_discharge = {}
for gid, info in gauge_grid_indices.items():
    idx = info['idx']
    try:
        if isinstance(idx, tuple):
            ts = ds[var_name][:, idx[0], idx[1]]
        else:
            # Check dimensions
            dims = ds[var_name].dims
            # Find the spatial dimension
            spatial_dim = [d for d in dims if d != 'time'][0]
            ts = ds[var_name].isel({spatial_dim: idx})
        
        model_discharge[gid] = pd.Series(
            ts.values.flatten(),
            index=pd.to_datetime(ds.time.values)
        )
    except Exception as e:
        print(f"  Error extracting model data for gauge {gid}: {e}")

ds.close()
print(f"Extracted model discharge for {len(model_discharge)} gauges")

# ─── 5. Load observation discharge data ─────────────────────────────────────
print("\nLoading observation data...")
obs_discharge = {}
for _, row in gauge_meta.iterrows():
    gid = str(int(row['gauge_id']))
    obs_file = os.path.join(obs_dir, f"{gid}.csv")
    try:
        obs_df = pd.read_csv(obs_file, parse_dates=['date'])
        # Filter to 1985-1989
        mask = (obs_df['date'].dt.year >= 1985) & (obs_df['date'].dt.year <= 1989)
        obs_filtered = obs_df[mask].copy()
        if len(obs_filtered) > 0:
            obs_discharge[gid] = pd.Series(
                obs_filtered['discharge_m3s'].values,
                index=obs_filtered['date'].values
            )
            print(f"  Gauge {gid}: {len(obs_filtered)} obs days")
        else:
            print(f"  Gauge {gid}: no data in 1985-1989")
    except Exception as e:
        print(f"  Error loading obs for gauge {gid}: {e}")

print(f"Loaded observations for {len(obs_discharge)} gauges")

# ─── 6. Compute FDC and distributional metrics ─────────────────────────────
def compute_fdc(discharge, n_percentiles=100):
    """Compute flow duration curve as exceedance probabilities."""
    valid = discharge[~np.isnan(discharge)]
    if len(valid) == 0:
        return np.full(n_percentiles, np.nan), np.linspace(0, 100, n_percentiles)
    sorted_q = np.sort(valid)[::-1]  # descending
    exceedance = np.linspace(0, 100, len(sorted_q))
    # Interpolate to standard percentiles
    target_pct = np.linspace(0, 100, n_percentiles)
    fdc_values = np.interp(target_pct, exceedance, sorted_q)
    return fdc_values, target_pct

def compute_quantile(discharge, exceedance_pct):
    """Compute flow value at given exceedance percentage."""
    valid = discharge[~np.isnan(discharge)]
    if len(valid) == 0:
        return np.nan
    # Q10 means flow exceeded 10% of the time (high flow)
    # Q90 means flow exceeded 90% of the time (low flow)
    return np.percentile(valid, 100 - exceedance_pct)

print("\nComputing FDC metrics...")
metrics_list = []
fdc_data = {}

# Find common gauges
common_gauges = set(model_discharge.keys()) & set(obs_discharge.keys())
print(f"Common gauges for analysis: {len(common_gauges)}")

for gid in sorted(common_gauges):
    mod_ts = model_discharge[gid]
    obs_ts = obs_discharge[gid]
    
    # Align to common dates if using daily data, or just use all available
    if use_monthly:
        # For monthly data, use all values directly
        mod_vals = mod_ts.dropna().values
        obs_vals = obs_ts.dropna().values
    else:
        # Align by date
        mod_idx = pd.DatetimeIndex(mod_ts.index)
        obs_idx = pd.DatetimeIndex(obs_ts.index)
        common_dates = mod_idx.intersection(obs_idx)
        if len(common_dates) == 0:
            # Try aligning by normalizing dates
            mod_ts.index = pd.to_datetime(mod_ts.index).normalize()
            obs_ts.index = pd.to_datetime(obs_ts.index).normalize()
            common_dates = mod_ts.index.intersection(obs_ts.index)
        
        if len(common_dates) > 0:
            mod_vals = mod_ts.loc[common_dates].dropna().values
            obs_vals = obs_ts.loc[common_dates].dropna().values
        else:
            mod_vals = mod_ts.dropna().values
            obs_vals = obs_ts.dropna().values
    
    if len(mod_vals) == 0 or len(obs_vals) == 0:
        print(f"  Gauge {gid}: insufficient data, skipping")
        continue
    
    # Ensure non-negative
    mod_vals = np.maximum(mod_vals, 0)
    obs_vals = np.maximum(obs_vals, 0)
    
    # Compute FDC
    mod_fdc, pct = compute_fdc(mod_vals, 101)
    obs_fdc, _ = compute_fdc(obs_vals, 101)
    
    # Quantile values
    mod_q10 = compute_quantile(mod_vals, 10)
    mod_q50 = compute_quantile(mod_vals, 50)
    mod_q90 = compute_quantile(mod_vals, 90)
    obs_q10 = compute_quantile(obs_vals, 10)
    obs_q50 = compute_quantile(obs_vals, 50)
    obs_q90 = compute_quantile(obs_vals, 90)
    
    # Quantile ratios (model/obs)
    q10_ratio = mod_q10 / obs_q10 if obs_q10 > 0 else np.nan
    q50_ratio = mod_q50 / obs_q50 if obs_q50 > 0 else np.nan
    q90_ratio = mod_q90 / obs_q90 if obs_q90 > 0 else np.nan
    
    # Volume bias
    vol_bias = (np.mean(mod_vals) - np.mean(obs_vals)) / np.mean(obs_vals) if np.mean(obs_vals) > 0 else np.nan
    
    # Wasserstein distance (Earth Mover's Distance)
    w_dist = wasserstein_distance(mod_vals, obs_vals)
    
    # Normalized Wasserstein distance (by mean obs)
    w_dist_norm = w_dist / np.mean(obs_vals) if np.mean(obs_vals) > 0 else np.nan
    
    # Get gauge info
    gauge_row = gauge_meta[gauge_meta['gauge_id'].astype(str) == gid]
    river_name = gauge_row['river_name'].values[0] if len(gauge_row) > 0 else "Unknown"
    lat = gauge_row['lat'].values[0] if len(gauge_row) > 0 else np.nan
    lon = gauge_row['lon'].values[0] if len(gauge_row) > 0 else np.nan
    
    metrics_list.append({
        'gauge_id': gid,
        'river_name': river_name,
        'lat': lat,
        'lon': lon,
        'n_model_days': len(mod_vals),
        'n_obs_days': len(obs_vals),
        'mean_model_m3s': np.mean(mod_vals),
        'mean_obs_m3s': np.mean(obs_vals),
        'volume_bias': vol_bias,
        'wasserstein_distance': w_dist,
        'wasserstein_distance_normalized': w_dist_norm,
        'model_Q10': mod_q10,
        'model_Q50': mod_q50,
        'model_Q90': mod_q90,
        'obs_Q10': obs_q10,
        'obs_Q50': obs_q50,
        'obs_Q90': obs_q90,
        'Q10_ratio': q10_ratio,
        'Q50_ratio': q50_ratio,
        'Q90_ratio': q90_ratio,
    })
    
    fdc_data[gid] = {
        'exceedance_pct': pct,
        'model_fdc': mod_fdc,
        'obs_fdc': obs_fdc,
    }
    
    print(f"  Gauge {gid} ({river_name}): vol_bias={vol_bias:.3f}, W_dist={w_dist:.1f}, "
          f"Q10_ratio={q10_ratio:.3f}, Q50_ratio={q50_ratio:.3f}, Q90_ratio={q90_ratio:.3f}")

# ─── 7. Save metrics CSV ───────────────────────────────────────────────────
try:
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv = os.path.join(outdir, "per_gauge_fdc_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, float_format='%.4f')
    print(f"\nSaved per-gauge metrics to {metrics_csv}")
except Exception as e:
    print(f"Error saving metrics CSV: {e}")

# ─── 8. Save FDC percentile data CSV ───────────────────────────────────────
try:
    fdc_rows = []
    for gid, data in fdc_data.items():
        for i, pct in enumerate(data['exceedance_pct']):
            fdc_rows.append({
                'gauge_id': gid,
                'exceedance_pct': pct,
                'model_discharge_m3s': data['model_fdc'][i],
                'obs_discharge_m3s': data['obs_fdc'][i],
            })
    fdc_df = pd.DataFrame(fdc_rows)
    fdc_csv = os.path.join(outdir, "fdc_percentile_data.csv")
    fdc_df.to_csv(fdc_csv, index=False, float_format='%.4f')
    print(f"Saved FDC percentile data to {fdc_csv}")
except Exception as e:
    print(f"Error saving FDC CSV: {e}")

# ─── 9. Create figure ──────────────────────────────────────────────────────
highlight_gauges = {
    '3629000': 'Amazon',
    '4121801': 'Missouri',
    '4115200': 'Columbia',
    '6742900': 'Danube',
    '2969100': 'Mekong',
    '1159100': 'Orange',
}

try:
    fig = plt.figure(figsize=(20, 14))
    
    # Layout: top row = map (spans full width), bottom 2 rows = 6 FDC panels (3x2)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                           height_ratios=[1.2, 1, 1])
    
    # ── Map panel ──
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax_map.set_title('Wasserstein Distance: E3SM vs Observed River Discharge (1985-1989)',
                     fontsize=14, fontweight='bold')
    
    # Plot all gauges colored by Wasserstein distance
    if len(metrics_df) > 0:
        lats = metrics_df['lat'].values
        lons = metrics_df['lon'].values
        w_dists = metrics_df['wasserstein_distance_normalized'].values
        
        # Use log scale for better color distribution
        w_plot = np.where(w_dists > 0, w_dists, np.nan)
        
        sc = ax_map.scatter(lons, lats, c=w_plot, cmap='YlOrRd',
                           s=60, edgecolors='black', linewidths=0.5,
                           transform=ccrs.PlateCarree(), zorder=5,
                           norm=plt.matplotlib.colors.LogNorm(
                               vmin=max(np.nanmin(w_plot), 0.01),
                               vmax=np.nanmax(w_plot)) if np.nanmax(w_plot) > 0 else None)
        
        cb = plt.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.05,
                          shrink=0.5, label='Normalized Wasserstein Distance')
        
        # Highlight the 6 gauges
        for gid, name in highlight_gauges.items():
            row = metrics_df[metrics_df['gauge_id'] == gid]
            if len(row) > 0:
                ax_map.plot(row['lon'].values[0], row['lat'].values[0],
                           marker='*', markersize=18, color='blue',
                           markeredgecolor='white', markeredgewidth=1.0,
                           transform=ccrs.PlateCarree(), zorder=10)
                ax_map.text(row['lon'].values[0] + 3, row['lat'].values[0] + 2,
                           name, fontsize=9, fontweight='bold', color='blue',
                           transform=ccrs.PlateCarree(), zorder=10,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    alpha=0.8, edgecolor='blue'))
    
    # ── FDC panels for 6 highlighted gauges ──
    panel_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    gauge_ids_ordered = ['3629000', '4121801', '4115200', '6742900', '2969100', '1159100']
    
    for i, gid in enumerate(gauge_ids_ordered):
        row_idx, col_idx = panel_positions[i]
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        name = highlight_gauges[gid]
        
        if gid in fdc_data:
            data = fdc_data[gid]
            pct = data['exceedance_pct']
            mod_fdc = data['model_fdc']
            obs_fdc = data['obs_fdc']
            
            ax.semilogy(pct, obs_fdc, 'b-', linewidth=2, label='Observed', alpha=0.8)
            ax.semilogy(pct, mod_fdc, 'r-', linewidth=2, label='E3SM', alpha=0.8)
            
            # Add quantile reference lines
            for q_exc, q_label in [(10, 'Q10'), (50, 'Q50'), (90, 'Q90')]:
                ax.axvline(x=q_exc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Add metrics text
            m_row = metrics_df[metrics_df['gauge_id'] == gid]
            if len(m_row) > 0:
                vb = m_row['volume_bias'].values[0]
                wd = m_row['wasserstein_distance'].values[0]
                q50r = m_row['Q50_ratio'].values[0]
                stats_text = f'Bias={vb:+.2f}\nW_dist={wd:.0f}\nQ50 ratio={q50r:.2f}'
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                       fontsize=7, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.legend(fontsize=8, loc='lower left')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
        
        ax.set_title(f'{name} ({gid})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Exceedance Probability (%)', fontsize=9)
        ax.set_ylabel('Discharge (m³/s)', fontsize=9)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    fig_path = os.path.join(outdir, "fdc_comparison_map_and_panels.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved figure to {fig_path}")
    
except Exception as e:
    print(f"Error creating figure: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Task complete ===")
print(f"Output directory: {outdir}")
print(f"Files created:")
for f in sorted(os.listdir(outdir)):
    fpath = os.path.join(outdir, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f} ({size_kb:.1f} KB)")