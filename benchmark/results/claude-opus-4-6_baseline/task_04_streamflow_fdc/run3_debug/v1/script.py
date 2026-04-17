import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wasserstein_distance
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Output directory
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_04_streamflow_fdc/run3_debug/v1/output"
os.makedirs(output_dir, exist_ok=True)

# Paths
case_name = "sample.v3.LR.historical"
rof_dir = "./data/sample/e3sm/rof"
obs_dir = "./data/sample/obs/streamflow"
meta_path = "./data/sample/obs/gauge_metadata.csv"

# Date range
start_date = "1985-01-01"
end_date = "1989-12-31"

# Highlighted gauges
highlight_gauges = {
    3629000: "Amazon",
    4121801: "Missouri",
    4115200: "Columbia",
    6742900: "Danube",
    2969100: "Mekong",
    1159100: "Orange"
}

# ---- Load gauge metadata ----
print("Loading gauge metadata...")
try:
    meta = pd.read_csv(meta_path)
    print(f"  Found {len(meta)} gauges in metadata")
    print(f"  Columns: {list(meta.columns)}")
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    raise

# ---- Load MOSART daily (h1) files ----
print("Loading MOSART daily output files...")
h1_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h1.*.nc")
h1_files = sorted(glob.glob(h1_pattern))
print(f"  Found {len(h1_files)} h1 files total")

# Filter files to 1985-1989 range
h1_files_filtered = []
for f in h1_files:
    basename = os.path.basename(f)
    parts = basename.replace(".nc", "").split(".")
    date_str = parts[-1]  # YYYY-MM-DD-00000
    year = int(date_str.split("-")[0])
    if 1985 <= year <= 1989:
        h1_files_filtered.append(f)

print(f"  Filtered to {len(h1_files_filtered)} h1 files for 1985-1989")

# If no h1 files, try using monthly h0 files instead
use_monthly = False
if len(h1_files_filtered) == 0:
    print("  No daily h1 files found for 1985-1989, trying monthly h0 files...")
    h0_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h0.*.nc")
    h0_files = sorted(glob.glob(h0_pattern))
    h0_files_filtered = []
    for f in h0_files:
        basename = os.path.basename(f)
        parts = basename.replace(".nc", "").split(".")
        date_str = parts[-1]  # YYYY-MM
        year = int(date_str.split("-")[0])
        if 1985 <= year <= 1989:
            h0_files_filtered.append(f)
    print(f"  Found {len(h0_files_filtered)} h0 files for 1985-1989")
    use_monthly = True
    files_to_load = h0_files_filtered
else:
    files_to_load = h1_files_filtered

# Load MOSART data
print("  Opening MOSART dataset...")
try:
    ds_mosart = xr.open_mfdataset(files_to_load, combine='nested', concat_dim='time',
                                   data_vars='minimal', coords='minimal', compat='override')
    print(f"  MOSART time range: {ds_mosart.time.values[0]} to {ds_mosart.time.values[-1]}")
    print(f"  MOSART variables: {list(ds_mosart.data_vars)}")

    var_name = "RIVER_DISCHARGE_OVER_LAND_LIQ"
    if var_name not in ds_mosart.data_vars:
        for v in ds_mosart.data_vars:
            if "DISCHARGE" in v.upper() or "FLOW" in v.upper():
                var_name = v
                break
    print(f"  Using variable: {var_name}")

    # Convert cftime to pandas-compatible datetime
    # Handle cftime.DatetimeNoLeap
    raw_times = ds_mosart.time.values
    try:
        time_values = pd.DatetimeIndex(raw_times)
    except TypeError:
        # cftime dates - convert manually
        import cftime
        time_values = pd.DatetimeIndex([
            pd.Timestamp(year=t.year, month=t.month, day=t.day,
                         hour=t.hour, minute=t.minute, second=t.second)
            for t in raw_times
        ])
        print(f"  Converted cftime dates to pandas DatetimeIndex")

    print(f"  Time values: {time_values[0]} to {time_values[-1]}, {len(time_values)} steps")

    # Get grid coordinates
    if 'lat' in ds_mosart.coords or 'lat' in ds_mosart:
        grid_lat = ds_mosart['lat'].values
        grid_lon = ds_mosart['lon'].values
    elif 'latixy' in ds_mosart:
        grid_lat = ds_mosart['latixy'].values
        grid_lon = ds_mosart['longxy'].values
    else:
        print(f"  Available coords: {list(ds_mosart.coords)}")
        raise ValueError("Cannot find lat/lon coordinates in MOSART output")

    print(f"  Grid shape: lat={grid_lat.shape}, lon={grid_lon.shape}")

    # Handle 360 lon convention
    if np.any(grid_lon > 180):
        grid_lon_adjusted = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    else:
        grid_lon_adjusted = grid_lon.copy()

except Exception as e:
    print(f"Error loading MOSART data: {e}")
    import traceback
    traceback.print_exc()
    raise

# ---- Function to find nearest grid cell ----
def find_nearest_cell(lat, lon, grid_lat, grid_lon):
    """Find index of nearest grid cell to given lat/lon."""
    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        # 1D coordinates (regular grid)
        lat_idx = np.argmin(np.abs(grid_lat - lat))
        lon_idx = np.argmin(np.abs(grid_lon - lon))
        return (lat_idx, lon_idx)
    elif grid_lat.ndim == 1:
        # 1D unstructured
        dist = np.sqrt((grid_lat - lat)**2 + (grid_lon - lon)**2)
        idx = np.argmin(dist)
        return idx
    else:
        # 2D coordinates
        dist = np.sqrt((grid_lat - lat)**2 + (grid_lon - lon)**2)
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx

# ---- Match gauges and extract data ----
print("\nMatching gauges to model grid and extracting data...")

discharge_var = ds_mosart[var_name]

# Determine dimension names
dims = discharge_var.dims
print(f"  Discharge variable dims: {dims}")
print(f"  Discharge variable shape: {discharge_var.shape}")

results = []
fdc_data_all = {}

for _, row in meta.iterrows():
    gauge_id = int(row['gauge_id'])
    glat = row['lat']
    glon = row['lon']

    print(f"\n  Processing gauge {gauge_id} ({row.get('river_name', 'Unknown')}) at ({glat:.2f}, {glon:.2f})...")

    # Find nearest grid cell
    try:
        idx = find_nearest_cell(glat, glon, grid_lat, grid_lon_adjusted)

        if isinstance(idx, tuple) and len(idx) == 2:
            # Regular 2D grid with (lat_idx, lon_idx)
            sim_ts = discharge_var[:, idx[0], idx[1]].values
            cell_lat = float(grid_lat[idx[0]])
            cell_lon = float(grid_lon_adjusted[idx[1]])
        elif isinstance(idx, tuple):
            sim_ts = discharge_var[:, idx[0], idx[1]].values
            cell_lat = float(grid_lat[idx[0], idx[1]])
            cell_lon = float(grid_lon_adjusted[idx[0], idx[1]])
        else:
            # 1D grid index
            if len(dims) == 2:
                spatial_dim = [d for d in dims if d != 'time'][0]
                sim_ts = discharge_var.isel({spatial_dim: int(idx)}).values
            else:
                sim_ts = discharge_var[:, int(idx)].values
            cell_lat = float(grid_lat[idx])
            cell_lon = float(grid_lon_adjusted[idx])

        dist_km = np.sqrt((glat - cell_lat)**2 + (glon - cell_lon)**2) * 111.0
        print(f"    Nearest grid cell: ({cell_lat:.2f}, {cell_lon:.2f}), distance: {dist_km:.1f} km")

        # Convert to pandas Series
        sim_series = pd.Series(sim_ts, index=time_values)
        # Filter to 1985-1989
        mask = (sim_series.index >= start_date) & (sim_series.index <= end_date)
        sim_series = sim_series.loc[mask]
        # Remove negative values
        sim_series = sim_series.clip(lower=0)

    except Exception as e:
        print(f"    Error extracting model data: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Load observations
    obs_file = os.path.join(obs_dir, f"{gauge_id}.csv")
    try:
        obs_df = pd.read_csv(obs_file, parse_dates=['date'])
        obs_df = obs_df.set_index('date')
        obs_series = obs_df['discharge_m3s'].loc[start_date:end_date]
        obs_series = obs_series.dropna()
        obs_series = obs_series.clip(lower=0)
        print(f"    Obs: {len(obs_series)} daily values, Sim: {len(sim_series)} values")
    except Exception as e:
        print(f"    Error loading observations: {e}")
        continue

    if len(obs_series) == 0 or len(sim_series) == 0:
        print(f"    Skipping gauge {gauge_id}: insufficient data")
        continue

    # If using monthly model data, resample obs to monthly for comparison
    if use_monthly:
        obs_monthly = obs_series.resample('MS').mean().dropna()
        # Align time indices - normalize both to month start
        sim_monthly = sim_series.copy()
        sim_monthly.index = sim_monthly.index.to_period('M').to_timestamp()
        obs_monthly.index = obs_monthly.index.to_period('M').to_timestamp()
        common_times = sim_monthly.index.intersection(obs_monthly.index)
        if len(common_times) == 0:
            print(f"    No overlapping months found")
            continue
        sim_aligned = sim_monthly.loc[common_times].values
        obs_aligned = obs_monthly.loc[common_times].values
    else:
        # For daily data, align on common dates
        # Normalize sim_series index to date only (remove time component)
        sim_series.index = sim_series.index.normalize()
        obs_series.index = obs_series.index.normalize()
        common_dates = sim_series.index.intersection(obs_series.index)
        if len(common_dates) == 0:
            print(f"    No overlapping dates found")
            print(f"    Sim range: {sim_series.index.min()} to {sim_series.index.max()}")
            print(f"    Obs range: {obs_series.index.min()} to {obs_series.index.max()}")
            continue
        sim_aligned = sim_series.loc[common_dates].values
        obs_aligned = obs_series.loc[common_dates].values

    # Remove NaN
    valid = np.isfinite(sim_aligned) & np.isfinite(obs_aligned)
    sim_valid = sim_aligned[valid]
    obs_valid = obs_aligned[valid]

    if len(sim_valid) < 10:
        print(f"    Skipping gauge {gauge_id}: only {len(sim_valid)} valid overlapping values")
        continue

    print(f"    {len(sim_valid)} overlapping time steps for analysis")

    # ---- Compute FDC percentiles ----
    percentiles = np.arange(0, 100.1, 1)

    sim_fdc = np.percentile(sim_valid, 100 - percentiles)  # exceedance
    obs_fdc = np.percentile(obs_valid, 100 - percentiles)

    fdc_data_all[gauge_id] = {
        'percentiles': percentiles,
        'sim_fdc': sim_fdc,
        'obs_fdc': obs_fdc,
        'sim_values': sim_valid,
        'obs_values': obs_valid
    }

    # ---- Compute distributional metrics ----
    # Volume bias
    vol_bias = (np.sum(sim_valid) - np.sum(obs_valid)) / np.sum(obs_valid) * 100

    # Wasserstein distance (Earth Mover's Distance)
    emd = wasserstein_distance(sim_valid, obs_valid)

    # Quantile ratios: Q10, Q50, Q90 (exceedance probabilities)
    sim_q10 = np.percentile(sim_valid, 90)  # Q10 exceedance = 90th percentile
    sim_q50 = np.percentile(sim_valid, 50)
    sim_q90 = np.percentile(sim_valid, 10)  # Q90 exceedance = 10th percentile

    obs_q10 = np.percentile(obs_valid, 90)
    obs_q50 = np.percentile(obs_valid, 50)
    obs_q90 = np.percentile(obs_valid, 10)

    q10_ratio = sim_q10 / obs_q10 if obs_q10 > 0 else np.nan
    q50_ratio = sim_q50 / obs_q50 if obs_q50 > 0 else np.nan
    q90_ratio = sim_q90 / obs_q90 if obs_q90 > 0 else np.nan

    mean_sim = np.mean(sim_valid)
    mean_obs = np.mean(obs_valid)

    result = {
        'gauge_id': gauge_id,
        'river_name': row.get('river_name', 'Unknown'),
        'lat': glat,
        'lon': glon,
        'grid_lat': cell_lat,
        'grid_lon': cell_lon,
        'grid_distance_km': dist_km,
        'n_timesteps': len(sim_valid),
        'mean_sim_m3s': mean_sim,
        'mean_obs_m3s': mean_obs,
        'volume_bias_pct': vol_bias,
        'wasserstein_distance': emd,
        'sim_Q10': sim_q10,
        'obs_Q10': obs_q10,
        'Q10_ratio': q10_ratio,
        'sim_Q50': sim_q50,
        'obs_Q50': obs_q50,
        'Q50_ratio': q50_ratio,
        'sim_Q90': sim_q90,
        'obs_Q90': obs_q90,
        'Q90_ratio': q90_ratio
    }
    results.append(result)

    print(f"    Volume Bias: {vol_bias:.1f}%, EMD: {emd:.1f} m³/s, Q50 ratio: {q50_ratio:.2f}")

# ---- Save per-gauge metrics CSV ----
print("\n\nSaving per-gauge metrics...")
try:
    metrics_df = pd.DataFrame(results)
    metrics_path = os.path.join(output_dir, "per_gauge_fdc_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, float_format='%.4f')
    print(f"  Saved metrics to {metrics_path}")
    print(f"\n{metrics_df.to_string()}")
except Exception as e:
    print(f"Error saving metrics: {e}")

# ---- Save FDC percentile data CSV ----
print("\nSaving FDC percentile data...")
try:
    fdc_records = []
    for gauge_id, fdc in fdc_data_all.items():
        for i, pct in enumerate(fdc['percentiles']):
            fdc_records.append({
                'gauge_id': gauge_id,
                'exceedance_pct': pct,
                'sim_discharge_m3s': fdc['sim_fdc'][i],
                'obs_discharge_m3s': fdc['obs_fdc'][i]
            })
    fdc_df = pd.DataFrame(fdc_records)
    fdc_path = os.path.join(output_dir, "fdc_percentile_data.csv")
    fdc_df.to_csv(fdc_path, index=False, float_format='%.4f')
    print(f"  Saved FDC data to {fdc_path}")
except Exception as e:
    print(f"Error saving FDC data: {e}")

# ---- Create figure ----
print("\nCreating figure...")
try:
    fig = plt.figure(figsize=(18, 16))

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                           height_ratios=[1.2, 1, 1])

    # ---- Map panel (top row, spans all 3 columns) ----
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
        ax_map.set_global()
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        ax_map.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor='steelblue', alpha=0.5)
    else:
        ax_map = fig.add_subplot(gs[0, :])

    # Plot all gauges colored by Wasserstein distance
    if len(metrics_df) > 0:
        lats = metrics_df['lat'].values
        lons = metrics_df['lon'].values
        emd_values = metrics_df['wasserstein_distance'].values

        # Clamp minimum for log scale
        emd_min = max(1, np.nanmin(emd_values[emd_values > 0])) if np.any(emd_values > 0) else 1
        emd_max = np.nanmax(emd_values) if np.nanmax(emd_values) > emd_min else emd_min * 10

        if HAS_CARTOPY:
            sc = ax_map.scatter(lons, lats, c=emd_values, cmap='YlOrRd',
                               s=100, edgecolors='black', linewidth=0.8,
                               transform=ccrs.PlateCarree(), zorder=5,
                               norm=plt.matplotlib.colors.LogNorm(
                                   vmin=emd_min, vmax=emd_max))
        else:
            sc = ax_map.scatter(lons, lats, c=emd_values, cmap='YlOrRd',
                               s=100, edgecolors='black', linewidth=0.8, zorder=5,
                               norm=plt.matplotlib.colors.LogNorm(
                                   vmin=emd_min, vmax=emd_max))

        cb = plt.colorbar(sc, ax=ax_map, shrink=0.6, pad=0.02, aspect=20)
        cb.set_label('Wasserstein Distance (m³/s)', fontsize=11)

        # Highlight the 6 specific gauges
        for gid, gname in highlight_gauges.items():
            gauge_row = metrics_df[metrics_df['gauge_id'] == gid]
            if len(gauge_row) > 0:
                glat_h = gauge_row['lat'].values[0]
                glon_h = gauge_row['lon'].values[0]
                if HAS_CARTOPY:
                    ax_map.plot(glon_h, glat_h, 'k*', markersize=16,
                               transform=ccrs.PlateCarree(), zorder=10)
                    ax_map.text(glon_h + 3, glat_h + 3, gname, fontsize=9,
                               fontweight='bold', transform=ccrs.PlateCarree(),
                               zorder=10, ha='left',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                        alpha=0.8, edgecolor='gray'))
                else:
                    ax_map.plot(glon_h, glat_h, 'k*', markersize=16, zorder=10)
                    ax_map.text(glon_h + 3, glat_h + 3, gname, fontsize=9,
                               fontweight='bold', zorder=10)

    ax_map.set_title('Wasserstein Distance: E3SM vs Observed River Discharge (1985–1989)',
                     fontsize=14, fontweight='bold')

    # ---- FDC comparison panels for 6 highlighted gauges ----
    panel_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    gauge_order = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]

    for i, gid in enumerate(gauge_order):
        row_idx, col_idx = panel_positions[i]
        ax = fig.add_subplot(gs[row_idx, col_idx])

        gname = highlight_gauges.get(gid, str(gid))

        if gid in fdc_data_all:
            fdc = fdc_data_all[gid]
            pcts = fdc['percentiles']

            # Use semilogy, but handle zeros
            sim_fdc_plot = np.maximum(fdc['sim_fdc'], 0.01)
            obs_fdc_plot = np.maximum(fdc['obs_fdc'], 0.01)

            ax.semilogy(pcts, obs_fdc_plot, 'b-', linewidth=2, label='Observed', alpha=0.9)
            ax.semilogy(pcts, sim_fdc_plot, 'r-', linewidth=2, label='E3SM', alpha=0.9)

            # Fill between
            ax.fill_between(pcts, obs_fdc_plot, sim_fdc_plot, alpha=0.15, color='purple')

            # Get metrics for annotation
            gauge_metrics = metrics_df[metrics_df['gauge_id'] == gid].iloc[0]

            annotation = (
                f"Vol. Bias: {gauge_metrics['volume_bias_pct']:.1f}%\n"
                f"EMD: {gauge_metrics['wasserstein_distance']:.0f} m³/s\n"
                f"Q10 ratio: {gauge_metrics['Q10_ratio']:.2f}\n"
                f"Q50 ratio: {gauge_metrics['Q50_ratio']:.2f}\n"
                f"Q90 ratio: {gauge_metrics['Q90_ratio']:.2f}"
            )

            ax.text(0.97, 0.97, annotation, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

            # Mark Q10, Q50, Q90 on exceedance axis
            for q_exc in [10, 50, 90]:
                ax.axvline(x=q_exc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

            ax.set_xlabel('Exceedance Probability (%)', fontsize=10)
            ax.set_ylabel('Discharge (m³/s)', fontsize=10)
            ax.legend(loc='lower left', fontsize=9)

            # Set reasonable y limits
            all_vals = np.concatenate([obs_fdc_plot, sim_fdc_plot])
            positive_vals = all_vals[all_vals > 0.01]
            if len(positive_vals) > 0:
                ymin = np.min(positive_vals) * 0.5
                ymax = np.max(positive_vals) * 2
                ax.set_ylim(ymin, ymax)

        else:
            ax.text(0.5, 0.5, f'No data available\nfor gauge {gid}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=11)

        ax.set_title(f'{gname} ({gid})', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, which='both')

    fig_path = os.path.join(output_dir, "streamflow_fdc_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved figure to {fig_path}")

except Exception as e:
    print(f"Error creating figure: {e}")
    import traceback
    traceback.print_exc()

# ---- Summary ----
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if len(results) > 0:
    print(f"Processed {len(results)} gauges successfully")
    print(f"\nMetric statistics across all gauges:")
    for col in ['volume_bias_pct', 'wasserstein_distance', 'Q10_ratio', 'Q50_ratio', 'Q90_ratio']:
        vals = metrics_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:25s}: median={vals.median():.3f}, mean={vals.mean():.3f}, "
                  f"min={vals.min():.3f}, max={vals.max():.3f}")

    print(f"\nHighlighted gauges:")
    for gid, gname in highlight_gauges.items():
        g = metrics_df[metrics_df['gauge_id'] == gid]
        if len(g) > 0:
            g = g.iloc[0]
            print(f"  {gname:12s} ({gid}): Vol.Bias={g['volume_bias_pct']:+7.1f}%, "
                  f"EMD={g['wasserstein_distance']:10.1f} m³/s, "
                  f"Q50_ratio={g['Q50_ratio']:.3f}")
        else:
            print(f"  {gname:12s} ({gid}): NOT FOUND in results")
else:
    print("No gauges were successfully processed.")

print(f"\nOutput files in: {output_dir}")
for f in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, f)
    fsize = os.path.getsize(fpath) / 1024
    print(f"  {f} ({fsize:.1f} KB)")

print("\nDone!")