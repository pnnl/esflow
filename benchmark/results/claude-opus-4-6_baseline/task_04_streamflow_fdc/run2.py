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
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_04_streamflow_fdc/run2_output"
os.makedirs(output_dir, exist_ok=True)

# Paths
case_name = "sample.v3.LR.historical"
rof_dir = "./data/sample/e3sm/rof"
gauge_meta_path = "./data/sample/obs/gauge_metadata.csv"
obs_dir = "./data/sample/obs/streamflow"

# Load gauge metadata
print("Loading gauge metadata...")
gauge_meta = pd.read_csv(gauge_meta_path)
print(f"Found {len(gauge_meta)} gauges")

# Load all MOSART daily h1 files for 1985-1989
print("Loading MOSART daily output files for 1985-1989...")
h1_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h1.*.nc")
all_h1_files = sorted(glob.glob(h1_pattern))

# Filter to 1985-1989
h1_files = []
for f in all_h1_files:
    fname = os.path.basename(f)
    # Extract date from filename like sample.v3.LR.historical.mosart.h1.1985-01-02-00000.nc
    parts = fname.replace(".nc", "").split(".")
    date_part = parts[-1]  # e.g., 1985-01-02-00000
    year = int(date_part.split("-")[0])
    if 1985 <= year <= 1989:
        h1_files.append(f)

print(f"Found {len(h1_files)} daily MOSART files for 1985-1989")

# If no daily h1 files, try monthly h0 files
use_monthly = False
if len(h1_files) == 0:
    print("No daily h1 files found, trying monthly h0 files...")
    h0_pattern = os.path.join(rof_dir, f"{case_name}.mosart.h0.*.nc")
    all_h0_files = sorted(glob.glob(h0_pattern))
    h0_files = []
    for f in all_h0_files:
        fname = os.path.basename(f)
        parts = fname.replace(".nc", "").split(".")
        date_part = parts[-1]  # e.g., 1985-01
        year = int(date_part.split("-")[0])
        if 1985 <= year <= 1989:
            h0_files.append(f)
    print(f"Found {len(h0_files)} monthly MOSART files for 1985-1989")
    use_monthly = True
    h1_files = h0_files

# Load model data
print("Opening model datasets...")
try:
    ds_model = xr.open_mfdataset(h1_files, combine='nested', concat_dim='time',
                                  data_vars='minimal', coords='minimal',
                                  compat='override')
    print(f"Model dataset time range: {ds_model.time.values[0]} to {ds_model.time.values[-1]}")
    print(f"Model variables: {list(ds_model.data_vars)}")
except Exception as e:
    print(f"Error opening model files: {e}")
    # Try loading one by one
    datasets = []
    for f in h1_files:
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except Exception as e2:
            print(f"  Could not open {f}: {e2}")
    if datasets:
        ds_model = xr.concat(datasets, dim='time')
    else:
        raise RuntimeError("Could not load any model files")

# Get discharge variable
var_name = 'RIVER_DISCHARGE_OVER_LAND_LIQ'
if var_name not in ds_model:
    print(f"Variable {var_name} not found. Available: {list(ds_model.data_vars)}")
    # Try alternative names
    for v in ds_model.data_vars:
        if 'DISCHARGE' in v.upper() or 'FLOW' in v.upper():
            var_name = v
            print(f"Using alternative variable: {var_name}")
            break

# Get model grid coordinates
if 'lat' in ds_model.coords:
    model_lat = ds_model['lat'].values
    model_lon = ds_model['lon'].values
elif 'lat' in ds_model:
    model_lat = ds_model['lat'].values
    model_lon = ds_model['lon'].values
else:
    print("Looking for coordinate variables...")
    for coord_name in ['latixy', 'latitude', 'LATIXY']:
        if coord_name in ds_model:
            model_lat = ds_model[coord_name].values
            break
    for coord_name in ['longxy', 'longitude', 'LONGXY']:
        if coord_name in ds_model:
            model_lon = ds_model[coord_name].values
            break

print(f"Model grid: lat shape={model_lat.shape}, lon shape={model_lon.shape}")

# Ensure lon is in -180 to 180 for matching
model_lon_180 = np.where(model_lon > 180, model_lon - 360, model_lon)


def find_nearest_cell(lat, lon, grid_lat, grid_lon):
    """Find nearest grid cell index for a given lat/lon."""
    lon_adjusted = lon
    grid_lon_use = grid_lon.copy()
    # Try both conventions
    dist = np.sqrt((grid_lat - lat)**2 + (grid_lon_use - lon_adjusted)**2)
    idx = np.argmin(dist)
    return idx


def compute_fdc(discharge):
    """Compute flow duration curve - returns exceedance probabilities and sorted flows."""
    sorted_q = np.sort(discharge)[::-1]  # descending
    n = len(sorted_q)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100  # percent
    return exceedance, sorted_q


def compute_fdc_percentiles(discharge, percentiles=None):
    """Compute FDC at specific exceedance percentiles."""
    if percentiles is None:
        percentiles = np.arange(1, 100, 1)
    sorted_q = np.sort(discharge)[::-1]
    n = len(sorted_q)
    exc_probs = np.arange(1, n + 1) / (n + 1) * 100
    # Interpolate to get flows at specific exceedance percentiles
    flows = np.interp(percentiles, exc_probs, sorted_q)
    return percentiles, flows


# Match gauges to model grid and extract data
print("\nMatching gauges to model grid and extracting data...")
results = []
fdc_data_all = {}
highlight_gauges = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
highlight_names = {3629000: 'Amazon', 4121801: 'Missouri', 4115200: 'Columbia',
                   6742900: 'Danube', 2969100: 'Mekong', 1159100: 'Orange'}

fdc_percentiles = np.arange(1, 100, 1)
fdc_records = []

for _, row in gauge_meta.iterrows():
    gauge_id = row['gauge_id']
    glat = row['lat']
    glon = row['lon']
    river_name = row.get('river_name', 'Unknown')

    # Load observation data
    obs_file = os.path.join(obs_dir, f"{gauge_id}.csv")
    if not os.path.exists(obs_file):
        print(f"  No obs file for gauge {gauge_id}, skipping")
        continue

    try:
        obs_df = pd.read_csv(obs_file, parse_dates=['date'])
        # Filter to 1985-1989
        obs_df = obs_df[(obs_df['date'] >= '1985-01-01') & (obs_df['date'] <= '1989-12-31')]
        obs_df = obs_df.dropna(subset=['discharge_m3s'])

        if len(obs_df) < 365:  # At least 1 year of data
            print(f"  Gauge {gauge_id}: insufficient obs data ({len(obs_df)} days), skipping")
            continue

        obs_discharge = obs_df['discharge_m3s'].values
    except Exception as e:
        print(f"  Error loading obs for gauge {gauge_id}: {e}")
        continue

    # Find nearest model grid cell
    try:
        idx = find_nearest_cell(glat, glon, model_lat, model_lon_180)

        # Extract model discharge at this cell
        model_discharge_ts = ds_model[var_name].isel(lndgrid=idx) if 'lndgrid' in ds_model.dims else \
                            ds_model[var_name].isel(gridcell=idx) if 'gridcell' in ds_model.dims else \
                            None

        if model_discharge_ts is None:
            # Try spatial dimension name
            spatial_dims = [d for d in ds_model[var_name].dims if d != 'time']
            if spatial_dims:
                model_discharge_ts = ds_model[var_name].isel({spatial_dims[0]: idx})
            else:
                print(f"  Cannot determine spatial dim for gauge {gauge_id}")
                continue

        sim_discharge = model_discharge_ts.values
        sim_discharge = sim_discharge[~np.isnan(sim_discharge)]

        if len(sim_discharge) < 30:
            print(f"  Gauge {gauge_id}: insufficient sim data ({len(sim_discharge)} steps)")
            continue

    except Exception as e:
        print(f"  Error extracting model data for gauge {gauge_id}: {e}")
        continue

    # If using monthly data, we need to handle comparison differently
    # For FDC, we just compare distributions

    # Make sure discharge values are positive
    obs_pos = obs_discharge[obs_discharge >= 0]
    sim_pos = sim_discharge[sim_discharge >= 0]

    if len(obs_pos) < 30 or len(sim_pos) < 30:
        print(f"  Gauge {gauge_id}: insufficient positive data")
        continue

    # Compute volume bias
    vol_bias = (np.mean(sim_pos) - np.mean(obs_pos)) / np.mean(obs_pos) * 100  # percent

    # Compute Wasserstein distance (Earth Mover's Distance)
    emd = wasserstein_distance(obs_pos, sim_pos)

    # Normalize EMD by mean observed discharge for comparability
    emd_norm = emd / np.mean(obs_pos) if np.mean(obs_pos) > 0 else np.nan

    # Compute quantile ratios (Q10, Q50, Q90 from FDC)
    # Q10 = flow exceeded 10% of time (high flow)
    # Q50 = median flow
    # Q90 = flow exceeded 90% of time (low flow)
    obs_q10 = np.percentile(obs_pos, 90)  # 90th percentile = exceeded 10% of time
    obs_q50 = np.percentile(obs_pos, 50)
    obs_q90 = np.percentile(obs_pos, 10)  # 10th percentile = exceeded 90% of time

    sim_q10 = np.percentile(sim_pos, 90)
    sim_q50 = np.percentile(sim_pos, 50)
    sim_q90 = np.percentile(sim_pos, 10)

    q10_ratio = sim_q10 / obs_q10 if obs_q10 > 0 else np.nan
    q50_ratio = sim_q50 / obs_q50 if obs_q50 > 0 else np.nan
    q90_ratio = sim_q90 / obs_q90 if obs_q90 > 0 else np.nan

    results.append({
        'gauge_id': gauge_id,
        'river_name': river_name,
        'lat': glat,
        'lon': glon,
        'n_obs_days': len(obs_pos),
        'n_sim_steps': len(sim_pos),
        'obs_mean_m3s': np.mean(obs_pos),
        'sim_mean_m3s': np.mean(sim_pos),
        'volume_bias_pct': vol_bias,
        'wasserstein_distance': emd,
        'wasserstein_distance_norm': emd_norm,
        'obs_Q10': obs_q10,
        'sim_Q10': sim_q10,
        'Q10_ratio': q10_ratio,
        'obs_Q50': obs_q50,
        'sim_Q50': sim_q50,
        'Q50_ratio': q50_ratio,
        'obs_Q90': obs_q90,
        'sim_Q90': sim_q90,
        'Q90_ratio': q90_ratio,
    })

    # Compute FDC percentile data
    obs_exc, obs_fdc = compute_fdc_percentiles(obs_pos, fdc_percentiles)
    sim_exc, sim_fdc = compute_fdc_percentiles(sim_pos, fdc_percentiles)

    for i, pct in enumerate(fdc_percentiles):
        fdc_records.append({
            'gauge_id': gauge_id,
            'river_name': river_name,
            'exceedance_pct': pct,
            'obs_discharge_m3s': obs_fdc[i],
            'sim_discharge_m3s': sim_fdc[i],
        })

    # Store FDC data for highlight gauges
    if int(gauge_id) in highlight_gauges:
        fdc_data_all[int(gauge_id)] = {
            'obs_exc': obs_exc, 'obs_fdc': obs_fdc,
            'sim_exc': sim_exc, 'sim_fdc': sim_fdc,
            'obs_raw': obs_pos, 'sim_raw': sim_pos,
        }

    print(f"  Gauge {gauge_id} ({river_name}): bias={vol_bias:.1f}%, EMD={emd:.1f}, "
          f"Q10_ratio={q10_ratio:.2f}, Q50_ratio={q50_ratio:.2f}, Q90_ratio={q90_ratio:.2f}")

# Create results DataFrame
df_results = pd.DataFrame(results)
print(f"\nProcessed {len(df_results)} gauges successfully")

# Save per-gauge metrics
try:
    metrics_path = os.path.join(output_dir, "fdc_metrics_per_gauge.csv")
    df_results.to_csv(metrics_path, index=False, float_format='%.4f')
    print(f"Saved metrics to {metrics_path}")
except Exception as e:
    print(f"Error saving metrics CSV: {e}")

# Save FDC percentile data
try:
    fdc_df = pd.DataFrame(fdc_records)
    fdc_path = os.path.join(output_dir, "fdc_percentile_data.csv")
    fdc_df.to_csv(fdc_path, index=False, float_format='%.4f')
    print(f"Saved FDC data to {fdc_path}")
except Exception as e:
    print(f"Error saving FDC CSV: {e}")

# Create figure
print("\nCreating figure...")
try:
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.94, bottom=0.05)

    # Top panel: Map of Wasserstein distance
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')

    if len(df_results) > 0:
        # Plot all gauges colored by normalized Wasserstein distance
        lons = df_results['lon'].values
        lats = df_results['lat'].values
        emd_vals = df_results['wasserstein_distance_norm'].values

        # Clip for colorbar
        vmin, vmax = 0, np.nanpercentile(emd_vals, 95) if len(emd_vals) > 0 else 1

        sc = ax_map.scatter(lons, lats, c=emd_vals, cmap='YlOrRd',
                           s=60, edgecolors='black', linewidths=0.5,
                           transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=max(vmax, 0.1),
                           zorder=5)

        # Highlight the 6 specific gauges
        for gid in highlight_gauges:
            gid_match = df_results[df_results['gauge_id'] == gid]
            if len(gid_match) > 0:
                ax_map.scatter(gid_match['lon'].values, gid_match['lat'].values,
                              s=150, marker='*', c='blue', edgecolors='white',
                              linewidths=1, transform=ccrs.PlateCarree(), zorder=10)
                name = highlight_names.get(gid, str(gid))
                ax_map.text(gid_match['lon'].values[0] + 3, gid_match['lat'].values[0] + 3,
                           name, fontsize=8, fontweight='bold',
                           transform=ccrs.PlateCarree(), zorder=10,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        cb = plt.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.05,
                         fraction=0.04, aspect=40)
        cb.set_label('Normalized Wasserstein Distance (EMD / mean obs Q)', fontsize=10)

    ax_map.set_title('Wasserstein Distance: E3SM vs Observed River Discharge (1985-1989)',
                     fontsize=14, fontweight='bold')

    # Bottom 2 rows: FDC comparison panels for 6 gauges
    gauge_order = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
    panel_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    for i, gid in enumerate(gauge_order):
        row, col = panel_positions[i]
        ax = fig.add_subplot(gs[row, col])

        name = highlight_names.get(gid, str(gid))

        if gid in fdc_data_all:
            data = fdc_data_all[gid]

            # Plot FDC
            ax.semilogy(data['obs_exc'], data['obs_fdc'], 'b-', linewidth=2,
                       label='Observed', alpha=0.8)
            ax.semilogy(data['sim_exc'], data['sim_fdc'], 'r-', linewidth=2,
                       label='E3SM', alpha=0.8)

            # Add metrics text
            gid_row = df_results[df_results['gauge_id'] == gid]
            if len(gid_row) > 0:
                bias = gid_row['volume_bias_pct'].values[0]
                emd_val = gid_row['wasserstein_distance'].values[0]
                q10r = gid_row['Q10_ratio'].values[0]
                q50r = gid_row['Q50_ratio'].values[0]
                q90r = gid_row['Q90_ratio'].values[0]
                text = (f'Bias: {bias:.1f}%\n'
                       f'EMD: {emd_val:.0f} m³/s\n'
                       f'Q10 ratio: {q10r:.2f}\n'
                       f'Q50 ratio: {q50r:.2f}\n'
                       f'Q90 ratio: {q90r:.2f}')
                ax.text(0.98, 0.98, text, transform=ax.transAxes,
                       fontsize=7, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Exceedance Probability (%)', fontsize=9)
            ax.set_ylabel('Discharge (m³/s)', fontsize=9)
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
        else:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')

        ax.set_title(f'{name} ({gid})', fontsize=11, fontweight='bold')

    fig_path = os.path.join(output_dir, "fdc_distribution_comparison.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")

except Exception as e:
    print(f"Error creating figure: {e}")
    import traceback
    traceback.print_exc()

# Close dataset
ds_model.close()

print("\nDone!")
print(f"Output directory: {output_dir}")
print(f"Files generated:")
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size = os.path.getsize(fpath)
    print(f"  {f} ({size/1024:.1f} KB)")