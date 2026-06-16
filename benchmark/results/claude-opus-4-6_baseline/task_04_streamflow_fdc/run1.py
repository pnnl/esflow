import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import wasserstein_distance
from glob import glob

# Output directory
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_04_streamflow_fdc/run1_output"
os.makedirs(output_dir, exist_ok=True)

# Paths
case_name = "sample.v3.LR.historical"
rof_dir = "./data/sample/e3sm/rof"
obs_dir = "./data/sample/obs/streamflow"
gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"

# Date range
start_date = "1985-01-01"
end_date = "1989-12-31"

# Featured gauges
featured_gauges = {
    3629000: "Amazon",
    4121801: "Missouri",
    4115200: "Columbia",
    6742900: "Danube",
    2969100: "Mekong",
    1159100: "Orange"
}

# ---- Load gauge metadata ----
try:
    gauge_meta = pd.read_csv(gauge_meta_path)
    print(f"Loaded gauge metadata: {len(gauge_meta)} gauges")
    print(f"Columns: {list(gauge_meta.columns)}")
    print(gauge_meta.head())
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    raise

# ---- Load MOSART daily files (h1) ----
print("\nLoading MOSART daily h1 files for 1985-1989...")
h1_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h1.*.nc")
h1_files = sorted(glob(h1_pattern))
print(f"Found {len(h1_files)} h1 files total")

# Filter to 1985-1989
h1_files_period = []
for f in h1_files:
    basename = os.path.basename(f)
    # Extract date from filename like sample.v3.LR.historical.mosart.h1.1985-01-02-00000.nc
    parts = basename.replace(".nc", "").split(".")
    date_part = parts[-1]  # e.g., "1985-01-02-00000"
    year_str = date_part.split("-")[0]
    try:
        year = int(year_str)
        if 1985 <= year <= 1989:
            h1_files_period.append(f)
    except ValueError:
        continue

print(f"Found {len(h1_files_period)} h1 files for 1985-1989")

# If no h1 files found, try monthly h0 files
use_monthly = False
if len(h1_files_period) == 0:
    print("No daily h1 files found, falling back to monthly h0 files...")
    h0_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h0.*.nc")
    h0_files = sorted(glob(h0_pattern))
    h0_files_period = []
    for f in h0_files:
        basename = os.path.basename(f)
        parts = basename.replace(".nc", "").split(".")
        date_part = parts[-1]  # e.g., "1985-01"
        year_str = date_part.split("-")[0]
        try:
            year = int(year_str)
            if 1985 <= year <= 1989:
                h0_files_period.append(f)
        except ValueError:
            continue
    print(f"Found {len(h0_files_period)} h0 files for 1985-1989")
    use_monthly = True
    files_to_load = h0_files_period
else:
    files_to_load = h1_files_period

# Load model data
try:
    ds_model = xr.open_mfdataset(files_to_load, combine='nested', concat_dim='time',
                                  data_vars='minimal', coords='minimal', compat='override')
    print(f"Model dataset loaded: {ds_model.dims}")
    print(f"Variables: {list(ds_model.data_vars)}")
    
    # Get discharge variable
    var_name = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    if var_name not in ds_model.data_vars:
        # Try alternative names
        for v in ds_model.data_vars:
            if "DISCHARGE" in v.upper() or "RIVER" in v.upper():
                var_name = v
                break
    print(f"Using discharge variable: {var_name}")
    
    # Get model grid coordinates
    model_lats = ds_model['lat'].values
    model_lons = ds_model['lon'].values
    print(f"Model grid: lat range [{model_lats.min():.2f}, {model_lats.max():.2f}], "
          f"lon range [{model_lons.min():.2f}, {model_lons.max():.2f}]")
    
except Exception as e:
    print(f"Error loading model data: {e}")
    raise

# ---- Helper function: find nearest grid cell ----
def find_nearest_grid_cell(lat, lon, model_lats, model_lons):
    """Find index of nearest grid cell to given lat/lon."""
    # Handle longitude convention (model might use 0-360)
    if model_lons.max() > 180 and lon < 0:
        lon_adj = lon + 360.0
    elif model_lons.min() < 0 and lon > 180:
        lon_adj = lon - 360.0
    else:
        lon_adj = lon
    
    # For 1D grids
    if model_lats.ndim == 1 and model_lons.ndim == 1:
        # MOSART typically has a single spatial dimension
        dist = np.sqrt((model_lats - lat)**2 + (model_lons - lon_adj)**2)
        idx = np.argmin(dist)
        return idx
    else:
        dist = np.sqrt((model_lats - lat)**2 + (model_lons - lon_adj)**2)
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx

# ---- Match gauges to model grid and extract time series ----
print("\nMatching gauges to model grid and extracting data...")

# Determine the spatial dimension name
spatial_dims = [d for d in ds_model[var_name].dims if d != 'time']
print(f"Spatial dimensions: {spatial_dims}")

results = {}
fdc_data = {}

for _, row in gauge_meta.iterrows():
    gauge_id = int(row['gauge_id'])
    gauge_lat = row['lat']
    gauge_lon = row['lon']
    
    try:
        # Load observation data
        obs_file = os.path.join(obs_dir, f"{gauge_id}.csv")
        if not os.path.exists(obs_file):
            print(f"  Gauge {gauge_id}: no observation file found, skipping")
            continue
        
        obs_df = pd.read_csv(obs_file, parse_dates=['date'])
        obs_df = obs_df[(obs_df['date'] >= start_date) & (obs_df['date'] <= end_date)]
        obs_df = obs_df.dropna(subset=['discharge_m3s'])
        
        if len(obs_df) < 30:
            print(f"  Gauge {gauge_id}: insufficient obs data ({len(obs_df)} days), skipping")
            continue
        
        # Find nearest model grid cell
        idx = find_nearest_grid_cell(gauge_lat, gauge_lon, model_lats, model_lons)
        
        # Extract model time series at this grid cell
        if len(spatial_dims) == 1:
            model_ts = ds_model[var_name][:, idx].values
        else:
            model_ts = ds_model[var_name][:, idx[0], idx[1]].values
        
        model_time = pd.to_datetime(ds_model['time'].values)
        model_df = pd.DataFrame({'date': model_time, 'sim_discharge': model_ts})
        model_df = model_df[(model_df['date'] >= start_date) & (model_df['date'] <= end_date)]
        
        # If monthly data, resample obs to monthly for comparison
        if use_monthly:
            obs_monthly = obs_df.set_index('date').resample('MS').mean().reset_index()
            obs_monthly.rename(columns={'discharge_m3s': 'obs_discharge'}, inplace=True)
            model_df['date'] = model_df['date'].dt.to_period('M').dt.to_timestamp()
            merged = pd.merge(model_df, obs_monthly, on='date', how='inner')
        else:
            obs_df_renamed = obs_df.rename(columns={'discharge_m3s': 'obs_discharge'})
            model_df['date'] = model_df['date'].dt.normalize()
            obs_df_renamed['date'] = obs_df_renamed['date'].dt.normalize()
            merged = pd.merge(model_df, obs_df_renamed, on='date', how='inner')
        
        merged = merged.dropna()
        
        if len(merged) < 30:
            print(f"  Gauge {gauge_id}: insufficient matched data ({len(merged)} points), skipping")
            continue
        
        sim = merged['sim_discharge'].values
        obs = merged['obs_discharge'].values
        
        # Compute FDC percentiles
        percentiles = np.arange(0, 100.1, 1.0)
        sim_sorted = np.sort(sim)[::-1]
        obs_sorted = np.sort(obs)[::-1]
        
        sim_fdc = np.percentile(sim, 100 - percentiles)
        obs_fdc = np.percentile(obs, 100 - percentiles)
        
        # Volume bias
        vol_bias = (np.sum(sim) - np.sum(obs)) / np.sum(obs) if np.sum(obs) > 0 else np.nan
        
        # Wasserstein distance (Earth Mover's Distance)
        # Normalize to avoid scale issues for the metric
        emd = wasserstein_distance(sim, obs)
        
        # Quantile ratios: Q10 (high flow), Q50 (median), Q90 (low flow)
        # Q10 = flow exceeded 10% of the time = 90th percentile
        sim_q10 = np.percentile(sim, 90)
        obs_q10 = np.percentile(obs, 90)
        sim_q50 = np.percentile(sim, 50)
        obs_q50 = np.percentile(obs, 50)
        sim_q90 = np.percentile(sim, 10)
        obs_q90 = np.percentile(obs, 10)
        
        q10_ratio = sim_q10 / obs_q10 if obs_q10 > 0 else np.nan
        q50_ratio = sim_q50 / obs_q50 if obs_q50 > 0 else np.nan
        q90_ratio = sim_q90 / obs_q90 if obs_q90 > 0 else np.nan
        
        results[gauge_id] = {
            'gauge_id': gauge_id,
            'lat': gauge_lat,
            'lon': gauge_lon,
            'river_name': row.get('river_name', ''),
            'n_matched_points': len(merged),
            'volume_bias': vol_bias,
            'wasserstein_distance': emd,
            'sim_Q10': sim_q10,
            'obs_Q10': obs_q10,
            'Q10_ratio': q10_ratio,
            'sim_Q50': sim_q50,
            'obs_Q50': obs_q50,
            'Q50_ratio': q50_ratio,
            'sim_Q90': sim_q90,
            'obs_Q90': obs_q90,
            'Q90_ratio': q90_ratio,
            'sim_mean': np.mean(sim),
            'obs_mean': np.mean(obs)
        }
        
        fdc_data[gauge_id] = {
            'percentiles': percentiles,
            'sim_fdc': sim_fdc,
            'obs_fdc': obs_fdc
        }
        
        print(f"  Gauge {gauge_id} ({row.get('river_name', 'N/A')}): "
              f"bias={vol_bias:.3f}, EMD={emd:.1f}, "
              f"Q10r={q10_ratio:.2f}, Q50r={q50_ratio:.2f}, Q90r={q90_ratio:.2f}")
        
    except Exception as e:
        print(f"  Gauge {gauge_id}: error - {e}")
        continue

# Close model dataset
ds_model.close()

print(f"\nSuccessfully processed {len(results)} gauges")

# ---- Save per-gauge metrics CSV ----
try:
    metrics_df = pd.DataFrame(list(results.values()))
    metrics_csv = os.path.join(output_dir, "per_gauge_fdc_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, float_format='%.6f')
    print(f"Saved per-gauge metrics to {metrics_csv}")
except Exception as e:
    print(f"Error saving metrics CSV: {e}")

# ---- Save FDC percentile data CSV ----
try:
    fdc_rows = []
    for gid, fdc in fdc_data.items():
        for i, pct in enumerate(fdc['percentiles']):
            fdc_rows.append({
                'gauge_id': gid,
                'exceedance_pct': pct,
                'sim_discharge_m3s': fdc['sim_fdc'][i],
                'obs_discharge_m3s': fdc['obs_fdc'][i]
            })
    fdc_df = pd.DataFrame(fdc_rows)
    fdc_csv = os.path.join(output_dir, "fdc_percentile_data.csv")
    fdc_df.to_csv(fdc_csv, index=False, float_format='%.4f')
    print(f"Saved FDC percentile data to {fdc_csv}")
except Exception as e:
    print(f"Error saving FDC CSV: {e}")

# ---- Create figure ----
try:
    print("\nCreating figure...")
    
    # Determine which featured gauges are available
    available_featured = {gid: name for gid, name in featured_gauges.items() if gid in results}
    print(f"Available featured gauges: {available_featured}")
    
    n_panels = len(available_featured)
    
    # Figure layout: top = map, bottom = FDC panels (up to 6)
    fig = plt.figure(figsize=(20, 14))
    
    if n_panels > 0:
        # Create grid: map on top spanning full width, FDC panels below
        ncols = min(3, n_panels)
        nrows_fdc = int(np.ceil(n_panels / ncols))
        gs = gridspec.GridSpec(1 + nrows_fdc, ncols, figure=fig, 
                               height_ratios=[1.5] + [1.0]*nrows_fdc,
                               hspace=0.35, wspace=0.3)
        
        # Map panel
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    else:
        ax_map = fig.add_subplot(111, projection=ccrs.Robinson())
    
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    
    # Plot all gauges colored by Wasserstein distance
    if len(results) > 0:
        all_lats = [r['lat'] for r in results.values()]
        all_lons = [r['lon'] for r in results.values()]
        all_emd = [r['wasserstein_distance'] for r in results.values()]
        
        # Use log scale for EMD coloring
        all_emd_arr = np.array(all_emd)
        valid_emd = all_emd_arr[all_emd_arr > 0]
        if len(valid_emd) > 0:
            vmin = np.percentile(valid_emd, 5)
            vmax = np.percentile(valid_emd, 95)
        else:
            vmin, vmax = 1, 1000
        
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-1), vmax=vmax)
        
        sc = ax_map.scatter(all_lons, all_lats, c=all_emd, cmap='RdYlGn_r',
                           norm=norm, s=40, edgecolors='k', linewidths=0.5,
                           transform=ccrs.PlateCarree(), zorder=5)
        
        # Highlight featured gauges
        for gid, name in available_featured.items():
            r = results[gid]
            ax_map.scatter(r['lon'], r['lat'], s=150, marker='*', 
                          c='none', edgecolors='blue', linewidths=2,
                          transform=ccrs.PlateCarree(), zorder=10)
            ax_map.text(r['lon'] + 2, r['lat'] + 2, name, fontsize=8,
                       fontweight='bold', color='blue',
                       transform=ccrs.PlateCarree(), zorder=10)
        
        cb = plt.colorbar(sc, ax=ax_map, orientation='vertical', shrink=0.7, pad=0.02)
        cb.set_label('Wasserstein Distance (m³/s)', fontsize=10)
    
    ax_map.set_title('Flow Duration Curve Analysis: Wasserstein Distance\n'
                     'E3SM vs Observations (1985-1989)', fontsize=14, fontweight='bold')
    
    # FDC comparison panels for featured gauges
    if n_panels > 0:
        for i, (gid, name) in enumerate(available_featured.items()):
            row_idx = 1 + i // ncols
            col_idx = i % ncols
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            fdc = fdc_data[gid]
            r = results[gid]
            
            ax.semilogy(fdc['percentiles'], fdc['obs_fdc'], 'b-', linewidth=2, label='Observed')
            ax.semilogy(fdc['percentiles'], fdc['sim_fdc'], 'r-', linewidth=2, label='E3SM')
            
            # Mark Q10, Q50, Q90
            for q_pct, q_label in [(10, 'Q10'), (50, 'Q50'), (90, 'Q90')]:
                ax.axvline(q_pct, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            
            ax.set_xlabel('Exceedance Probability (%)', fontsize=9)
            ax.set_ylabel('Discharge (m³/s)', fontsize=9)
            ax.set_title(f'{name} ({gid})\nBias={r["volume_bias"]:.2f}, '
                        f'EMD={r["wasserstein_distance"]:.0f}', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add quantile ratio annotations
            text_str = (f'Q10r={r["Q10_ratio"]:.2f}\n'
                       f'Q50r={r["Q50_ratio"]:.2f}\n'
                       f'Q90r={r["Q90_ratio"]:.2f}')
            ax.text(0.02, 0.02, text_str, transform=ax.transAxes,
                   fontsize=7, verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    fig_path = os.path.join(output_dir, "streamflow_fdc_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

except Exception as e:
    print(f"Error creating figure: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")