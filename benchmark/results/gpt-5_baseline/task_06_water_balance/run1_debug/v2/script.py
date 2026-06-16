import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape as shapely_shape

# -------------------------
# Configuration and paths
# -------------------------
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"
basin_geojson = "./data/sample/obs/basin_polygons.geojson"
gauge_meta_csv = "./data/sample/obs/gauge_metadata.csv"  # not used directly

start_year = 1985
end_year = 1989

# Output directory (as specified by the user)
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_06_water_balance/run1_debug/v2/output"
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
    # Check coords and variables
    for name in list(ds.coords) + list(ds.variables):
        lname = name.lower()
        if lat_name is None and (lname == "lat" or lname == "latitude" or ("lat" in lname and "n" not in lname)):
            lat_name = name
        if lon_name is None and (lname == "lon" or lname == "longitude" or ("lon" in lname and "g" not in lname)):
            lon_name = name
    # Fallback: search dims
    if lat_name is None or lon_name is None:
        for v in ds.data_vars:
            for d in ds[v].dims:
                dl = d.lower()
                if lat_name is None and "lat" in dl:
                    lat_name = d
                if lon_name is None and "lon" in dl:
                    lon_name = d
    if lat_name is None or lon_name is None:
        raise ValueError("Could not determine latitude/longitude coordinate names.")
    return lat_name, lon_name

def normalize_longitude(ds, lon_name):
    try:
        lon = ds[lon_name]
        if lon.ndim == 1:
            lon_vals = lon.values
            lon_norm = ((lon_vals + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: lon_norm})
            try:
                ds = ds.sortby(lon_name)
            except Exception:
                order = np.argsort(ds[lon_name].values)
                ds = ds.isel({lon_name: order})
        else:
            lon_vals = lon.values
            lon_norm = ((lon_vals + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: (lon.dims, lon_norm)})
    except Exception:
        pass
    return ds

def area_weights(lat, lon):
    # Compute approximate area weights for lat-lon grids
    if lat.ndim == 1 and lon.ndim == 1:
        lat_rad = np.deg2rad(lat.values)
        lon_rad = np.deg2rad(lon.values)
        dlat = np.gradient(lat_rad)
        dlon = np.gradient(lon_rad)
        coslat = np.cos(lat_rad)
        w_lat = np.abs(dlat * coslat)
        w = np.outer(w_lat, np.abs(dlon))
        return xr.DataArray(w, coords={lat.dims[0]: lat, lon.dims[0]: lon}, dims=(lat.dims[0], lon.dims[0]))
    else:
        coslat = np.abs(np.cos(np.deg2rad(lat)))
        return xr.DataArray(coslat, coords={d: lat.coords[d] for d in lat.dims}, dims=lat.dims)

def to_2d_lonlat(ds, lat_name, lon_name):
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
        from shapely import vectorized as svect
        mask = svect.contains(geom, lon2d, lat2d)
    except Exception:
        mask = None
    if mask is not None:
        return mask
    # Fallback using Path
    from matplotlib.path import Path
    pts = np.vstack([lon2d.ravel(), lat2d.ravel()]).T
    m = np.zeros(pts.shape[0], dtype=bool)
    try:
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        geoms = []
        if geom.geom_type == 'Polygon':
            geoms = [geom]
        elif geom.geom_type == 'MultiPolygon':
            geoms = list(geom.geoms)
        elif geom.geom_type == 'GeometryCollection':
            geoms = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        else:
            geoms = [geom]
    except Exception:
        geoms = [geom]
    for g in geoms:
        try:
            exterior = np.asarray(g.exterior.coords)
            path_ext = Path(exterior)
            inside = path_ext.contains_points(pts)
            if hasattr(g, "interiors"):
                for hole in g.interiors:
                    inter = np.asarray(hole.coords)
                    path_hole = Path(inter)
                    in_hole = path_hole.contains_points(pts)
                    inside[in_hole] = False
            m = m | inside
        except Exception:
            continue
    return m.reshape(lon2d.shape)

def weighted_mean(da, weights, dim=None):
    if dim is None:
        dim = list(da.dims)
    w = xr.broadcast(weights, da)[0]
    w = w.where(np.isfinite(da))
    total_w = w.sum(dim=dim, skipna=True)
    mean = (da * w).sum(dim=dim, skipna=True) / total_w
    return mean

def get_time_dim(da):
    for d in da.dims:
        if 'time' in d.lower():
            return d
    return None

def load_basin_geojson(path):
    basin_map = {}
    try:
        with open(path, 'r') as f:
            gj = json.load(f)
        features = gj.get("features", [])
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", None)
            if geom is None:
                continue
            grdc_no = None
            for k in props:
                if k.lower() == "grdc_no":
                    grdc_no = str(props[k]).strip()
                    break
            if grdc_no is None:
                continue
            try:
                shp = shapely_shape(geom)
            except Exception:
                continue
            basin_map[grdc_no] = shp
    except Exception as e:
        print(f"Error reading basin polygons from GeoJSON: {e}")
    return basin_map

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

# Open dataset with only needed variables, with decode_times=False to avoid noleap/pandas error
vars_needed = ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]
print(f"Opening {len(elm_files)} ELM files...")
ds = xr.open_mfdataset(elm_files, combine="by_coords", decode_times=False)

# Ensure variables exist
missing = [v for v in vars_needed if v not in ds.variables]
if missing:
    raise KeyError(f"Missing variables in dataset: {missing}")

# Identify lat/lon names and normalize longitude to [-180, 180]
lat_name, lon_name = find_lat_lon_names(ds)
ds = normalize_longitude(ds, lon_name)

# Convert flux units to mm/day
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

# Compute climatological mean (equal-weighted over months to avoid calendar issues)
print("Computing climatological means...")
time_dim = get_time_dim(P)
if time_dim is None:
    raise ValueError("Time dimension not found in variables.")
weights_t = xr.DataArray(np.ones(P.sizes[time_dim], dtype=float), coords={time_dim: P[time_dim]}, dims=(time_dim,))
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
aw = area_weights(lat, lon)

# Compute global means (ignore NaNs)
global_P = float(weighted_mean(P_mean, aw, dim=[lat_name, lon_name]).values)
global_ET = float(weighted_mean(ET_mean, aw, dim=[lat_name, lon_name]).values)
global_Q = float(weighted_mean(Q_mean, aw, dim=[lat_name, lon_name]).values)
global_res = float(weighted_mean(residual_mean, aw, dim=[lat_name, lon_name]).values)

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
basin_map = load_basin_geojson(basin_geojson)
if not basin_map:
    print("Warning: No basin polygons loaded. Proceeding without basin clipping.")

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
    try:
        mask = geometry_to_mask(lon2d, lat2d, geom)
    except Exception as e:
        print(f"Warning: Could not compute mask for basin {basin_name}: {e}")
        continue
    mask_da = xr.DataArray(mask, coords={lat_name: clim_ds[lat_name], lon_name: clim_ds[lon_name]}, dims=(lat_name, lon_name))
    # Masked fields
    P_b = P_mean.where(mask_da)
    ET_b = ET_mean.where(mask_da)
    Q_b = Q_mean.where(mask_da)
    Res_b = residual_mean.where(mask_da)
    # Area weights masked
    aw_b = aw.where(mask_da)

    # Compute means
    try:
        mean_P_b = float(weighted_mean(P_b, aw_b, dim=[lat_name, lon_name]).values)
    except Exception:
        mean_P_b = float('nan')
    try:
        mean_ET_b = float(weighted_mean(ET_b, aw_b, dim=[lat_name, lon_name]).values)
    except Exception:
        mean_ET_b = float('nan')
    try:
        mean_Q_b = float(weighted_mean(Q_b, aw_b, dim=[lat_name, lon_name]).values)
    except Exception:
        mean_Q_b = float('nan')
    try:
        mean_Res_b = float(weighted_mean(Res_b, aw_b, dim=[lat_name, lon_name]).values)
    except Exception:
        mean_Res_b = float('nan')

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
try:
    fig = plt.figure(figsize=(16, 12))
    gs_map = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.3)

    # Top panel: global residual map with basin outlines
    ax_map = fig.add_subplot(gs_map[0, 0], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.coastlines(linewidth=0.7)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.2)

    try:
        res_min = float(residual_mean.min().values)
        res_max = float(residual_mean.max().values)
        vlim = max(abs(res_min), abs(res_max))
        vlim = max(vlim, 0.5)
    except Exception:
        vlim = 2.0

    try:
        pcm = ax_map.pcolormesh(clim_ds[lon_name], clim_ds[lat_name], residual_mean,
                                transform=ccrs.PlateCarree(), cmap='RdBu_r',
                                vmin=-vlim, vmax=vlim, shading='auto')
        cb = plt.colorbar(pcm, ax=ax_map, orientation='horizontal', pad=0.05, fraction=0.05)
        cb.set_label("Water balance residual (P - ET - Q) [mm/day]")
    except Exception as e:
        print(f"Error plotting residual map: {e}")

    # Add basin outlines
    if basin_map:
        try:
            for basin_name, gid in basins_target.items():
                geom = basin_map.get(str(gid))
                if geom is None:
                    continue
                ax_map.add_geometries([geom], crs=ccrs.PlateCarree(),
                                      facecolor='none', edgecolor='k', linewidth=1.0, alpha=0.9)
                try:
                    centroid = geom.centroid
                    ax_map.text(centroid.x, centroid.y, basin_name, transform=ccrs.PlateCarree(),
                                fontsize=8, weight='bold', ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.0))
                except Exception:
                    pass
        except Exception as e:
            print(f"Error adding basin outlines: {e}")

    # Bottom panels: per-basin bar charts (2 rows x 3 columns)
    gs_bars = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_map[1:, 0], wspace=0.3, hspace=0.4)

    # Prepare basin stats dict for plotting
    stats_lookup = {s["basin"]: s for s in basin_stats}
    basins_for_plot = [b for b in basins_target.keys() if b in stats_lookup]

    bar_colors = {'P': '#1f77b4', 'ET': '#2ca02c', 'Q': '#ff7f0e', 'Residual': '#d62728'}

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
        try:
            finite_vals = [v for v in values if np.isfinite(v)]
            ymax = max([abs(v) for v in finite_vals]) if finite_vals else 1.0
            for xi, val in zip(x, values):
                if np.isfinite(val):
                    offset = 0.02 * (ymax if ymax > 0 else 1.0)
                    ax.text(xi, val + np.sign(val if val != 0 else 1) * offset, f"{val:.2f}",
                            ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
        except Exception:
            pass

    fig.suptitle(f"E3SM ELM Water Balance (1985-1989)\nCase: {case_name}", fontsize=14)
    figfile = os.path.join(outdir, f"water_balance_composite_{start_year}-{end_year}.png")
    plt.savefig(figfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved composite figure to {figfile}")
except Exception as e:
    print(f"Error creating/saving composite figure: {e}")

print("Analysis complete.")