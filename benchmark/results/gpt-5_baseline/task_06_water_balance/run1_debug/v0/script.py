import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader as shpreader

from datetime import datetime

# -------------------------
# Configuration and paths
# -------------------------
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"
basin_geojson = "./data/sample/obs/basin_polygons.geojson"
gauge_meta_csv = "./data/sample/obs/gauge_metadata.csv"

start_year = 1985
end_year = 1989

# Output directory (as specified by the user)
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_06_water_balance/run1_output"
os.makedirs(outdir, exist_ok=True)

# Basins of interest: name -> gauge_id (grdc_no)
basins_target = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}

# -------------------------
# Helper functions
# -------------------------
def find_lat_lon_names(ds):
    lat_name = None
    lon_name = None
    for name in ds.coords:
        lname = name.lower()
        if ('lat' == lname) or ('latitude' == lname) or ('lat' in lname and 'n' not in lname):
            lat_name = name if lat_name is None else lat_name
        if ('lon' == lname) or ('longitude' == lname) or ('lon' in lname and 'g' not in lname):
            lon_name = name if lon_name is None else lon_name
    # Fallback: search in data variables dims
    if lat_name is None or lon_name is None:
        for var in ds.data_vars:
            for dim in ds[var].dims:
                d = dim.lower()
                if lat_name is None and 'lat' in d:
                    lat_name = dim
                if lon_name is None and 'lon' in d:
                    lon_name = dim
    if lat_name is None or lon_name is None:
        raise ValueError("Could not determine latitude/longitude coordinate names.")
    return lat_name, lon_name

def normalize_longitude(ds, lon_name):
    # Convert lon to [-180, 180] and sort
    lon = ds[lon_name]
    lon_vals = lon.values
    lon_norm = ((lon_vals + 180) % 360) - 180
    ds = ds.assign_coords({lon_name: lon_norm})
    try:
        ds = ds.sortby(lon_name)
    except Exception:
        # If sortby fails due to chunking or other issues, use argsort
        order = np.argsort(ds[lon_name].values)
        ds = ds.isel({lon_name: order})
    return ds

def compute_time_weights(time_index):
    # Weight by days in month
    try:
        # pandas datetimeindex
        t = pd.DatetimeIndex(time_index.values)
        days = t.days_in_month.values.astype(float)
    except Exception:
        # Fallback: assume equal weighting
        days = np.ones(len(time_index), dtype=float)
    weights = xr.DataArray(days, coords={time_index.dims[0]: time_index}, dims=(time_index.dims[0],))
    return weights

def area_weights(lat, lon):
    # Compute approximate area weights for regular lat-lon grids
    # lat, lon are 1D arrays
    R = 6371000.0
    lat_rad = np.deg2rad(lat.values)
    # compute lat bounds
    dlat = np.gradient(lat_rad)
    # compute lon bounds in radians
    if lon.ndim == 1:
        lon_rad = np.deg2rad(lon.values)
        dlon = np.gradient(lon_rad)
    else:
        # Shouldn't happen for 1D lon
        lon_rad = np.deg2rad(lon.values)
        dlon = np.gradient(lon_rad, axis=-1)
    # Use outer product to get cell area weights
    # Approximate area weight proportional to cos(lat) * dlat * dlon
    # Produce a 2D array of shape (lat, lon)
    coslat = np.cos(lat_rad)
    w_lat = dlat * coslat
    if lon.ndim == 1:
        w = np.outer(w_lat, dlon)
    else:
        # Fallback
        w = np.outer(w_lat, np.ones(lon.shape[-1]))
    # Ensure positive weights
    w = np.abs(w)
    return xr.DataArray(w, coords={lat.dims[0]: lat, lon.dims[0]: lon}, dims=(lat.dims[0], lon.dims[0]))

def to_2d_lonlat(ds, lat_name, lon_name):
    # Return 2D lon/lat grids
    lat = ds[lat_name]
    lon = ds[lon_name]
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    else:
        lon2d = lon.values
        lat2d = lat.values
    return lon2d, lat2d

def geometry_to_mask(lon2d, lat2d, geom):
    mask = None
    try:
        # Try vectorized shapely if available
        from shapely import geometry as sgeom
        try:
            from shapely import vectorized as svect
            m = svect.contains(geom, lon2d, lat2d)
            mask = m
        except Exception:
            mask = None
    except Exception:
        mask = None

    if mask is None:
        # Fallback using matplotlib.path
        from matplotlib.path import Path
        import numpy as np
        pts = np.vstack([lon2d.ravel(), lat2d.ravel()]).T
        m = np.zeros(pts.shape[0], dtype=bool)
        geoms = []
        try:
            from shapely.geometry import MultiPolygon, Polygon
            if geom.geom_type == 'MultiPolygon':
                geoms = list(geom.geoms)
            elif geom.geom_type == 'Polygon':
                geoms = [geom]
            else:
                # Try to buffer tiny geometries or handle geometry collections
                try:
                    geoms = [geom]
                except Exception:
                    geoms = []
        except Exception:
            geoms = []
        for g in geoms:
            try:
                exterior = np.asarray(g.exterior.coords)
                path_ext = Path(exterior)
                inside = path_ext.contains_points(pts)
                # Remove holes
                if g.interiors:
                    for hole in g.interiors:
                        inter = np.asarray(hole.coords)
                        path_hole = Path(inter)
                        in_hole = path_hole.contains_points(pts)
                        inside[in_hole] = False
                m = m | inside
            except Exception:
                continue
        mask = m.reshape(lon2d.shape)
    return mask

def weighted_mean(da, weights, dim=None):
    if dim is None:
        dim = list(da.dims)
    # Align weights with da
    w = weights
    # Mask weights where da is nan
    w = xr.broadcast(w, da)[0]
    w = w.where(np.isfinite(da))
    total_w = w.sum(dim=dim, skipna=True)
    mean = (da * w).sum(dim=dim, skipna=True) / total_w
    return mean

# -------------------------
# Read ELM data
# -------------------------
print("Collecting ELM files...")
dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
elm_files = []
for t in dates:
    fn = os.path.join(elm_dir, f"{case_name}.elm.h0.{t.strftime('%Y-%m')}.nc")
    if os.path.exists(fn):
        elm_files.append(fn)

if len(elm_files) == 0:
    raise FileNotFoundError("No ELM monthly files found for the specified period.")

# Open dataset with only needed variables
vars_needed = ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]
print(f"Opening {len(elm_files)} ELM files...")
ds = xr.open_mfdataset(elm_files, combine="by_coords", decode_times=True, use_cftime=False)

# Ensure variables exist
missing = [v for v in vars_needed if v not in ds.variables]
if missing:
    raise KeyError(f"Missing variables in dataset: {missing}")

# Identify lat/lon names and normalize longitude to [-180, 180]
lat_name, lon_name = find_lat_lon_names(ds)
ds = normalize_longitude(ds, lon_name)

# Convert flux units to mm/day
# P = (RAIN + SNOW) [kg/m2/s] -> mm/s -> mm/day
# ET = (QVEGE + QVEGT + QSOIL) [mm/s] -> mm/day
# Q = QRUNOFF [mm/s] -> mm/day
sec_per_day = 86400.0
P = (ds["RAIN"] + ds["SNOW"]) * sec_per_day
ET = (ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]) * sec_per_day
Q = (ds["QRUNOFF"]) * sec_per_day

P.name = "P"
ET.name = "ET"
Q.name = "Q"
P.attrs["units"] = "mm/day"
ET.attrs["units"] = "mm/day"
Q.attrs["units"] = "mm/day"

# Compute time-weighted climatological mean (1985-1989)
print("Computing time-weighted climatological means...")
time_dim = [d for d in P.dims if d in ds.dims and 'time' in d][0] if 'time' in ds.dims else 'time'
if time_dim not in P.dims:
    # fallback
    time_dim = 'time'

weights_t = compute_time_weights(ds[time_dim])

P_mean = (P * weights_t).sum(dim=time_dim, skipna=True) / weights_t.sum()
ET_mean = (ET * weights_t).sum(dim=time_dim, skipna=True) / weights_t.sum()
Q_mean = (Q * weights_t).sum(dim=time_dim, skipna=True) / weights_t.sum()
residual_mean = P_mean - ET_mean - Q_mean
residual_mean.name = "Residual"
residual_mean.attrs["units"] = "mm/day"

# Compute area weights for global mean
print("Computing global area-weighted means...")
lat = ds[lat_name]
lon = ds[lon_name]
if lat.ndim == 1 and lon.ndim == 1:
    aw = area_weights(lat, lon)
else:
    # Fallback: cos(lat) weighting
    aw = xr.ufuncs.cos(np.deg2rad(lat))
    # broadcast to 2D if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        aw = aw * xr.ones_like(lon)[None, :]
    aw = aw.where(np.isfinite(P_mean))

# Compute global means (ignore NaNs)
global_P = weighted_mean(P_mean, aw, dim=[lat_name, lon_name]).item()
global_ET = weighted_mean(ET_mean, aw, dim=[lat_name, lon_name]).item()
global_Q = weighted_mean(Q_mean, aw, dim=[lat_name, lon_name]).item()
global_res = weighted_mean(residual_mean, aw, dim=[lat_name, lon_name]).item()

# Save global means to CSV
global_csv = os.path.join(outdir, f"global_means_{start_year}-{end_year}.csv")
try:
    df_global = pd.DataFrame({
        "component": ["P", "ET", "Q", "Residual"],
        "mm_per_day": [global_P, global_ET, global_Q, global_res]
    })
    df_global.to_csv(global_csv, index=False)
    print(f"Saved global means to {global_csv}")
except Exception as e:
    print(f"Error saving global means CSV: {e}")

# Save climatological fields to NetCDF
clim_ds = xr.Dataset(
    {
        "P": P_mean,
        "ET": ET_mean,
        "Q": Q_mean,
        "Residual": residual_mean
    }
)
clim_nc = os.path.join(outdir, f"elm_climatology_{start_year}-{end_year}.nc")
try:
    clim_ds.to_netcdf(clim_nc)
    print(f"Saved climatological fields to {clim_nc}")
except Exception as e:
    print(f"Error saving climatology NetCDF: {e}")

# -------------------------
# Basin clipping and statistics
# -------------------------
print("Reading basin polygons...")
basin_shapes = []
try:
    reader = shpreader.Reader(basin_geojson)
    basin_records = list(reader.records())
except Exception as e:
    print(f"Error reading basin polygons: {e}")
    basin_records = []

# Build a map from grdc_no to geometry and name (if available)
basin_map = {}
for rec in basin_records:
    props = rec.attributes
    geom = rec.geometry
    grdc_no = None
    # property name as specified: "grdc_no"
    for key in props.keys():
        if key.lower() == "grdc_no":
            grdc_no = str(props[key]).strip()
            break
    if grdc_no is None:
        continue
    basin_map[grdc_no] = geom

# Prepare lon/lat grids for masking
lon2d, lat2d = to_2d_lonlat(clim_ds, lat_name, lon_name)

# Compute per-basin area-weighted means
basin_stats = []
for basin_name, gid in basins_target.items():
    geom = basin_map.get(str(gid))
    if geom is None:
        print(f"Warning: Basin polygon for gauge_id {gid} not found in GeoJSON.")
        continue
    print(f"Computing mask for basin {basin_name} (ID: {gid})...")
    mask = geometry_to_mask(lon2d, lat2d, geom)
    if mask is None:
        print(f"Warning: Could not compute mask for basin {basin_name}.")
        continue
    mask_da = xr.DataArray(mask, coords={lat_name: clim_ds[lat_name], lon_name: clim_ds[lon_name]}, dims=(lat_name, lon_name))
    # Masked fields
    P_b = P_mean.where(mask_da)
    ET_b = ET_mean.where(mask_da)
    Q_b = Q_mean.where(mask_da)
    Res_b = residual_mean.where(mask_da)
    # Area weights masked
    if isinstance(aw, xr.DataArray):
        aw_b = aw.where(mask_da)
    else:
        # compute area weights as cos(lat)
        aw_b = xr.ufuncs.cos(np.deg2rad(clim_ds[lat_name]))
        if clim_ds[lat_name].ndim == 1 and clim_ds[lon_name].ndim == 1:
            aw_b = aw_b * xr.ones_like(clim_ds[lon_name])[None, :]
        aw_b = aw_b.where(mask_da)

    # Compute means
    mean_P_b = weighted_mean(P_b, aw_b, dim=[lat_name, lon_name]).item()
    mean_ET_b = weighted_mean(ET_b, aw_b, dim=[lat_name, lon_name]).item()
    mean_Q_b = weighted_mean(Q_b, aw_b, dim=[lat_name, lon_name]).item()
    mean_Res_b = weighted_mean(Res_b, aw_b, dim=[lat_name, lon_name]).item()

    basin_stats.append({
        "basin": basin_name,
        "gauge_id": gid,
        "P_mm_per_day": mean_P_b,
        "ET_mm_per_day": mean_ET_b,
        "Q_mm_per_day": mean_Q_b,
        "Residual_mm_per_day": mean_Res_b
    })

# Save basin stats to CSV
basin_csv = os.path.join(outdir, f"basin_means_{start_year}-{end_year}.csv")
try:
    df_basin = pd.DataFrame(basin_stats)
    df_basin.to_csv(basin_csv, index=False)
    print(f"Saved basin means to {basin_csv}")
except Exception as e:
    print(f"Error saving basin means CSV: {e}")

# -------------------------
# Plot composite figure
# -------------------------
print("Creating composite figure...")
fig = plt.figure(figsize=(14, 12))

# Top panel: global residual map with basin outlines
ax_map = plt.subplot2grid((3, 2), (0, 0), colspan=2, projection=ccrs.Robinson())
ax_map.set_global()
ax_map.coastlines(linewidth=0.7)
ax_map.add_feature(cfeature.BORDERS, linewidth=0.2)

# Plot residual field
try:
    cmap = plt.cm.RdBu_r
    vlim = np.nanmax(np.abs([residual_mean.min().item(), residual_mean.max().item()]))
    vlim = max(vlim, 0.5)  # prevent too small limits
    pcm = ax_map.pcolormesh(clim_ds[lon_name], clim_ds[lat_name], residual_mean,
                            transform=ccrs.PlateCarree(), cmap=cmap, vmin=-vlim, vmax=vlim, shading='auto')
    cb = plt.colorbar(pcm, ax=ax_map, orientation='horizontal', pad=0.05, fraction=0.05)
    cb.set_label("Water balance residual (P - ET - Q) [mm/day]")
except Exception as e:
    print(f"Error plotting residual map: {e}")

# Add basin outlines
try:
    for basin_name, gid in basins_target.items():
        geom = basin_map.get(str(gid))
        if geom is None: 
            continue
        ax_map.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor='none', edgecolor='k', linewidth=1.2, alpha=0.8)
        # Optionally add label near centroid
        try:
            centroid = geom.centroid
            ax_map.text(centroid.x, centroid.y, basin_name, transform=ccrs.PlateCarree(),
                        fontsize=8, weight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.5))
        except Exception:
            pass
except Exception as e:
    print(f"Error adding basin outlines: {e}")

# Bottom panels: per-basin bar charts
# Arrange as 2 columns x 2 rows of subplots? We have 6 basins: we'll do 2 rows x 3 columns
basins_for_plot = [b for b in basins_target.keys() if any(s["basin"] == b for s in basin_stats)]
n_basins = len(basins_for_plot)
ncols = 3
nrows = 2
bar_colors = {'P': '#1f77b4', 'ET': '#2ca02c', 'Q': '#ff7f0e', 'Residual': '#d62728'}

for i, basin_name in enumerate(basins_for_plot):
    row = 1 + i // ncols
    col = i % ncols
    ax = plt.subplot2grid((3, 2), (row, col)) if ncols == 2 else plt.subplot(3, 3, 4 + i)
    # For consistency, we will directly create a grid of 2 rows x 3 columns below map
    # Alternative: simple plt.subplot approach
plt.clf()

# Rebuild figure to ensure correct layout
fig = plt.figure(figsize=(16, 12))
gs_map = plt.GridSpec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.25)

# Map axis
ax_map = fig.add_subplot(gs_map[0, 0], projection=ccrs.Robinson())
ax_map.set_global()
ax_map.coastlines(linewidth=0.7)
ax_map.add_feature(cfeature.BORDERS, linewidth=0.2)
try:
    vlim = np.nanmax(np.abs([residual_mean.min().item(), residual_mean.max().item()]))
    vlim = max(vlim, 0.5)
    pcm = ax_map.pcolormesh(clim_ds[lon_name], clim_ds[lat_name], residual_mean,
                            transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vlim, vmax=vlim, shading='auto')
    cb = plt.colorbar(pcm, ax=ax_map, orientation='horizontal', pad=0.05, fraction=0.05)
    cb.set_label("Water balance residual (P - ET - Q) [mm/day]")
except Exception as e:
    print(f"Error plotting residual map (second attempt): {e}")

try:
    for basin_name, gid in basins_target.items():
        geom = basin_map.get(str(gid))
        if geom is None:
            continue
        ax_map.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor='none', edgecolor='k', linewidth=1.2, alpha=0.8)
        try:
            centroid = geom.centroid
            ax_map.text(centroid.x, centroid.y, basin_name, transform=ccrs.PlateCarree(),
                        fontsize=8, weight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.0))
        except Exception:
            pass
except Exception as e:
    print(f"Error adding basin outlines (second attempt): {e}")

# Bar charts grid 2 rows x 3 columns
gs_bars = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_map[1:, 0], wspace=0.25, hspace=0.35)

# Prepare basin stats dict for quick lookup
stats_lookup = {s["basin"]: s for s in basin_stats}

for idx, basin_name in enumerate(basins_for_plot):
    r = idx // 3
    c = idx % 3
    ax = fig.add_subplot(gs_bars[r, c])
    s = stats_lookup[basin_name]
    components = ["P", "ET", "Q", "Residual"]
    values = [s["P_mm_per_day"], s["ET_mm_per_day"], s["Q_mm_per_day"], s["Residual_mm_per_day"]]
    colors = [bar_colors[k] for k in components]
    x = np.arange(len(components))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_title(basin_name)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("mm/day")
    # Add value labels
    for xi, val in zip(x, values):
        ax.text(xi, val + (0.02 * (max(values) - min(values) + 1e-6)), f"{val:.2f}", ha='center', va='bottom', fontsize=8)

# Save composite figure
figfile = os.path.join(outdir, f"water_balance_composite_{start_year}-{end_year}.png")
try:
    fig.suptitle(f"E3SM ELM Water Balance (1985-1989)\nCase: {case_name}", fontsize=14)
    plt.savefig(figfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved composite figure to {figfile}")
except Exception as e:
    print(f"Error saving composite figure: {e}")

print("Analysis complete.")