import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# Output directory
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_06_water_balance/run2_output"
os.makedirs(output_dir, exist_ok=True)

# Constants
case_name = "sample.v3.LR.historical"
lnd_dir = "./data/sample/e3sm/lnd"
basin_file = "./data/sample/obs/basin_polygons.geojson"
gauge_meta_file = "./data/sample/obs/gauge_metadata.csv"

# Basin definitions: name -> gauge_id
basins = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}

# Conversion: mm/s -> mm/day
MM_S_TO_MM_DAY = 86400.0

# ============================================================
# 1. Load ELM monthly data for 1985-1989
# ============================================================
print("Loading ELM monthly data for 1985-1989...")
files = []
for year in range(1985, 1990):
    for month in range(1, 13):
        fname = os.path.join(lnd_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fname):
            files.append(fname)
        else:
            print(f"  Warning: file not found: {fname}")

print(f"  Found {len(files)} files")

if len(files) == 0:
    raise FileNotFoundError("No ELM files found for 1985-1989")

# Open all files
ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)

# ============================================================
# 2. Compute component fields
# ============================================================
print("Computing water balance components...")

# Precipitation: RAIN + SNOW (kg/m2/s = mm/s)
precip = ds["RAIN"] + ds["SNOW"]

# Evapotranspiration: QVEGE + QVEGT + QSOIL (mm/s)
et = ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]

# Total runoff: QRUNOFF (mm/s)
runoff = ds["QRUNOFF"]

# Compute time-mean fields (still in mm/s)
P_mean = precip.mean(dim="time")
ET_mean = et.mean(dim="time")
Q_mean = runoff.mean(dim="time")

# Convert to mm/day for interpretability
P_mean_mmday = P_mean * MM_S_TO_MM_DAY
ET_mean_mmday = ET_mean * MM_S_TO_MM_DAY
Q_mean_mmday = Q_mean * MM_S_TO_MM_DAY

# Residual: P - ET - Q
residual_mmday = P_mean_mmday - ET_mean_mmday - Q_mean_mmday

# Get lat/lon coordinates
lat = ds["lat"]
lon = ds["lon"]

# Determine grid structure
print(f"  Data dimensions: {P_mean_mmday.dims}")
print(f"  Data shape: {P_mean_mmday.shape}")

# ============================================================
# 3. Compute area-weighted global means
# ============================================================
print("Computing area-weighted global means...")

try:
    # Try to get landfrac and area from the dataset
    if "landfrac" in ds:
        landfrac = ds["landfrac"].isel(time=0) if "time" in ds["landfrac"].dims else ds["landfrac"]
    else:
        landfrac = xr.ones_like(P_mean_mmday)

    if "area" in ds:
        area = ds["area"].isel(time=0) if "time" in ds["area"].dims else ds["area"]
    else:
        # Compute area weights from lat
        if P_mean_mmday.ndim == 2:
            lat_vals = ds["lat"].values
            weights = np.cos(np.deg2rad(lat_vals))
            if weights.ndim == 1:
                weights_2d = np.broadcast_to(weights[:, np.newaxis], P_mean_mmday.shape)
            else:
                weights_2d = weights
            area = xr.DataArray(weights_2d, dims=P_mean_mmday.dims, coords=P_mean_mmday.coords)
        else:
            # 1D grid - use lat values
            lat_vals = lat.values
            weights = np.cos(np.deg2rad(lat_vals))
            area = xr.DataArray(weights, dims=P_mean_mmday.dims, coords=P_mean_mmday.coords)

    # Apply land fraction
    total_weight = (area * landfrac)

    # Mask out zero/nan weights
    valid = np.isfinite(P_mean_mmday.values) & (total_weight.values > 0)
    tw = total_weight.values[valid]

    P_global = np.average(P_mean_mmday.values[valid], weights=tw)
    ET_global = np.average(ET_mean_mmday.values[valid], weights=tw)
    Q_global = np.average(Q_mean_mmday.values[valid], weights=tw)
    R_global = P_global - ET_global - Q_global

    print(f"  Global area-weighted means (mm/day):")
    print(f"    P  = {P_global:.4f}")
    print(f"    ET = {ET_global:.4f}")
    print(f"    Q  = {Q_global:.4f}")
    print(f"    Residual (P-ET-Q) = {R_global:.4f}")

except Exception as e:
    print(f"  Error computing global means: {e}")
    P_global = float(np.nanmean(P_mean_mmday.values))
    ET_global = float(np.nanmean(ET_mean_mmday.values))
    Q_global = float(np.nanmean(Q_mean_mmday.values))
    R_global = P_global - ET_global - Q_global
    print(f"  Falling back to simple means (mm/day):")
    print(f"    P  = {P_global:.4f}")
    print(f"    ET = {ET_global:.4f}")
    print(f"    Q  = {Q_global:.4f}")
    print(f"    Residual = {R_global:.4f}")

# Save global summary
try:
    summary = pd.DataFrame({
        "Component": ["Precipitation (P)", "Evapotranspiration (ET)", "Runoff (Q)", "Residual (P-ET-Q)"],
        "Global_Mean_mm_day": [P_global, ET_global, Q_global, R_global]
    })
    summary.to_csv(os.path.join(output_dir, "global_water_balance_summary.csv"), index=False)
    print("  Saved global summary CSV")
except Exception as e:
    print(f"  Error saving summary CSV: {e}")

# ============================================================
# 4. Load basin polygons and compute basin-averaged values
# ============================================================
print("Loading basin polygons and computing basin averages...")

try:
    with open(basin_file, "r") as f:
        geojson = json.load(f)
except Exception as e:
    print(f"  Error loading basin polygons: {e}")
    geojson = None

# Function to check if a point is inside a polygon using ray-casting
def point_in_polygon(px, py, polygon):
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def get_basin_mask(lon_vals, lat_vals, geometry, grid_is_2d=True):
    """Create a boolean mask for grid cells inside a basin polygon."""
    # Handle MultiPolygon and Polygon
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return None

    if grid_is_2d:
        if lon_vals.ndim == 2:
            mask = np.zeros(lon_vals.shape, dtype=bool)
            for i in range(lon_vals.shape[0]):
                for j in range(lon_vals.shape[1]):
                    px, py = lon_vals[i, j], lat_vals[i, j]
                    for poly_coords in polygons:
                        exterior = poly_coords[0]
                        if point_in_polygon(px, py, exterior):
                            mask[i, j] = True
                            break
        else:
            # 1D lat/lon -> meshgrid
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
            mask = np.zeros(lon2d.shape, dtype=bool)
            for i in range(lon2d.shape[0]):
                for j in range(lon2d.shape[1]):
                    px, py = lon2d[i, j], lat2d[i, j]
                    for poly_coords in polygons:
                        exterior = poly_coords[0]
                        if point_in_polygon(px, py, exterior):
                            mask[i, j] = True
                            break
    else:
        # 1D unstructured grid
        mask = np.zeros(lon_vals.shape, dtype=bool)
        for i in range(len(lon_vals)):
            px, py = lon_vals[i], lat_vals[i]
            for poly_coords in polygons:
                exterior = poly_coords[0]
                if point_in_polygon(px, py, exterior):
                    mask[i] = True
                    break
    return mask


# Get lat/lon values
lon_vals = lon.values
lat_vals = lat.values

# Determine grid type
if P_mean_mmday.ndim == 1:
    grid_is_2d = False
    grid_is_structured = False
elif P_mean_mmday.ndim == 2:
    grid_is_2d = True
    grid_is_structured = True
else:
    grid_is_2d = False
    grid_is_structured = False

print(f"  Grid type: {'2D structured' if grid_is_structured else '1D unstructured'}")

# Process each basin
basin_results = {}

if geojson is not None:
    # Build feature lookup by gauge_id
    features_by_id = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        gid = str(props.get("grdc_no", props.get("gauge_id", "")))
        features_by_id[gid] = feature

    for basin_name, gauge_id in basins.items():
        print(f"  Processing basin: {basin_name} (gauge_id={gauge_id})")
        feature = features_by_id.get(gauge_id)
        if feature is None:
            print(f"    Warning: No polygon found for gauge_id={gauge_id}")
            basin_results[basin_name] = {"P": np.nan, "ET": np.nan, "Q": np.nan, "R": np.nan}
            continue

        geometry = feature["geometry"]

        try:
            # Ensure lon_vals are in same convention as polygon
            # Check polygon longitude range
            if geometry["type"] == "Polygon":
                all_coords = geometry["coordinates"][0]
            elif geometry["type"] == "MultiPolygon":
                all_coords = []
                for poly in geometry["coordinates"]:
                    all_coords.extend(poly[0])

            poly_lons = [c[0] for c in all_coords]
            poly_lon_min = min(poly_lons)
            poly_lon_max = max(poly_lons)

            # Adjust grid longitudes if needed
            lon_use = lon_vals.copy()
            if poly_lon_min < 0 and np.nanmin(lon_use) >= 0:
                # Polygon uses -180 to 180, grid uses 0 to 360
                lon_use = np.where(lon_use > 180, lon_use - 360, lon_use)
            elif poly_lon_min >= 0 and np.nanmin(lon_use) < 0:
                # Polygon uses 0-360, grid uses -180-180
                lon_use = np.where(lon_use < 0, lon_use + 360, lon_use)

            if grid_is_structured:
                if lon_use.ndim == 1 and lat_vals.ndim == 1:
                    mask = get_basin_mask(lon_use, lat_vals, geometry, grid_is_2d=True)
                else:
                    mask = get_basin_mask(lon_use, lat_vals, geometry, grid_is_2d=True)
            else:
                mask = get_basin_mask(lon_use, lat_vals, geometry, grid_is_2d=False)

            n_cells = np.sum(mask)
            print(f"    Found {n_cells} grid cells in basin")

            if n_cells == 0:
                basin_results[basin_name] = {"P": np.nan, "ET": np.nan, "Q": np.nan, "R": np.nan}
                continue

            # Extract basin-averaged values
            P_basin_vals = P_mean_mmday.values[mask]
            ET_basin_vals = ET_mean_mmday.values[mask]
            Q_basin_vals = Q_mean_mmday.values[mask]

            # Area weighting within basin
            if grid_is_structured:
                if lat_vals.ndim == 1:
                    lat2d = np.broadcast_to(lat_vals[:, np.newaxis], mask.shape) if lat_vals.shape[0] == mask.shape[0] else np.broadcast_to(lat_vals[np.newaxis, :], mask.shape)
                    w = np.cos(np.deg2rad(lat2d))[mask]
                else:
                    w = np.cos(np.deg2rad(lat_vals))[mask]
            else:
                w = np.cos(np.deg2rad(lat_vals))[mask]

            valid = np.isfinite(P_basin_vals) & (w > 0)
            if np.sum(valid) > 0:
                P_avg = np.average(P_basin_vals[valid], weights=w[valid])
                ET_avg = np.average(ET_basin_vals[valid], weights=w[valid])
                Q_avg = np.average(Q_basin_vals[valid], weights=w[valid])
            else:
                P_avg = np.nanmean(P_basin_vals)
                ET_avg = np.nanmean(ET_basin_vals)
                Q_avg = np.nanmean(Q_basin_vals)

            R_avg = P_avg - ET_avg - Q_avg
            basin_results[basin_name] = {"P": P_avg, "ET": ET_avg, "Q": Q_avg, "R": R_avg}
            print(f"    P={P_avg:.3f}, ET={ET_avg:.3f}, Q={Q_avg:.3f}, R={R_avg:.4f} mm/day")

        except Exception as e:
            print(f"    Error processing basin {basin_name}: {e}")
            import traceback
            traceback.print_exc()
            basin_results[basin_name] = {"P": np.nan, "ET": np.nan, "Q": np.nan, "R": np.nan}

# Save basin results
try:
    basin_df = pd.DataFrame(basin_results).T
    basin_df.index.name = "Basin"
    basin_df.columns = ["P_mm_day", "ET_mm_day", "Q_mm_day", "Residual_mm_day"]
    basin_df.to_csv(os.path.join(output_dir, "basin_water_balance.csv"))
    print("  Saved basin water balance CSV")
except Exception as e:
    print(f"  Error saving basin CSV: {e}")

# ============================================================
# 5. Save spatial fields as NetCDF
# ============================================================
print("Saving spatial fields as NetCDF...")
try:
    out_ds = xr.Dataset()
    out_ds["P"] = P_mean_mmday.copy()
    out_ds["P"].attrs = {"units": "mm/day", "long_name": "Precipitation (RAIN+SNOW)"}
    out_ds["ET"] = ET_mean_mmday.copy()
    out_ds["ET"].attrs = {"units": "mm/day", "long_name": "Evapotranspiration (QVEGE+QVEGT+QSOIL)"}
    out_ds["Q"] = Q_mean_mmday.copy()
    out_ds["Q"].attrs = {"units": "mm/day", "long_name": "Total Runoff (QRUNOFF)"}
    out_ds["residual"] = residual_mmday.copy()
    out_ds["residual"].attrs = {"units": "mm/day", "long_name": "Water Balance Residual (P-ET-Q)"}
    out_ds.attrs["description"] = "E3SM ELM water balance closure 1985-1989 climatological mean"
    out_ds.to_netcdf(os.path.join(output_dir, "water_balance_fields.nc"))
    print("  Saved water_balance_fields.nc")
except Exception as e:
    print(f"  Error saving NetCDF: {e}")

# ============================================================
# 6. Create composite figure
# ============================================================
print("Creating composite figure...")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    has_cartopy = True
except ImportError:
    has_cartopy = False
    print("  Warning: cartopy not available, using simple plot")

try:
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.35)

    # ---- Top panel: Global residual map ----
    if has_cartopy:
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    else:
        ax_map = fig.add_subplot(gs[0, :])

    # Prepare residual data for plotting
    res_data = residual_mmday.values.copy()

    if grid_is_structured:
        if lon_vals.ndim == 1 and lat_vals.ndim == 1:
            plot_lon = lon_vals.copy()
            plot_lat = lat_vals.copy()

            # Convert 0-360 to -180-180 for plotting
            if np.nanmax(plot_lon) > 180:
                shift_idx = np.argmax(plot_lon > 180)
                plot_lon = np.where(plot_lon > 180, plot_lon - 360, plot_lon)
                sort_idx = np.argsort(plot_lon)
                plot_lon = plot_lon[sort_idx]
                res_data = res_data[:, sort_idx]
                P_plot = P_mean_mmday.values[:, sort_idx]
                ET_plot = ET_mean_mmday.values[:, sort_idx]
                Q_plot = Q_mean_mmday.values[:, sort_idx]
            else:
                P_plot = P_mean_mmday.values
                ET_plot = ET_mean_mmday.values
                Q_plot = Q_mean_mmday.values

            # Determine colorbar range
            vmax = max(abs(np.nanpercentile(res_data, 2)), abs(np.nanpercentile(res_data, 98)))
            vmax = min(vmax, 2.0)  # cap at 2 mm/day
            if vmax == 0:
                vmax = 0.1
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            if has_cartopy:
                im = ax_map.pcolormesh(plot_lon, plot_lat, res_data,
                                       transform=ccrs.PlateCarree(),
                                       cmap="RdBu", norm=norm, shading="auto")
                ax_map.coastlines(linewidth=0.5)
                ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
                ax_map.set_global()
            else:
                im = ax_map.pcolormesh(plot_lon, plot_lat, res_data,
                                       cmap="RdBu", norm=norm, shading="auto")
        else:
            # 2D lat/lon arrays
            plot_lon = lon_vals.copy()
            plot_lat = lat_vals.copy()
            if np.nanmax(plot_lon) > 180:
                plot_lon = np.where(plot_lon > 180, plot_lon - 360, plot_lon)

            vmax = max(abs(np.nanpercentile(res_data, 2)), abs(np.nanpercentile(res_data, 98)))
            vmax = min(vmax, 2.0)
            if vmax == 0:
                vmax = 0.1
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            if has_cartopy:
                im = ax_map.pcolormesh(plot_lon, plot_lat, res_data,
                                       transform=ccrs.PlateCarree(),
                                       cmap="RdBu", norm=norm, shading="auto")
                ax_map.coastlines(linewidth=0.5)
                ax_map.set_global()
            else:
                im = ax_map.pcolormesh(plot_lon, plot_lat, res_data,
                                       cmap="RdBu", norm=norm, shading="auto")
    else:
        # Unstructured 1D grid - scatter plot
        plot_lon = lon_vals.copy()
        plot_lat = lat_vals.copy()
        if np.nanmax(plot_lon) > 180:
            plot_lon = np.where(plot_lon > 180, plot_lon - 360, plot_lon)

        vmax = max(abs(np.nanpercentile(res_data, 2)), abs(np.nanpercentile(res_data, 98)))
        vmax = min(vmax, 2.0)
        if vmax == 0:
            vmax = 0.1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        if has_cartopy:
            im = ax_map.scatter(plot_lon, plot_lat, c=res_data, s=1,
                                transform=ccrs.PlateCarree(),
                                cmap="RdBu", norm=norm)
            ax_map.coastlines(linewidth=0.5)
            ax_map.set_global()
        else:
            im = ax_map.scatter(plot_lon, plot_lat, c=res_data, s=1,
                                cmap="RdBu", norm=norm)

    # Add basin outlines to the map
    if geojson is not None:
        basin_colors = {"Amazon": "green", "Missouri": "blue", "Columbia": "purple",
                        "Danube": "orange", "Mekong": "red", "Orange": "brown"}
        for basin_name, gauge_id in basins.items():
            feature = features_by_id.get(gauge_id)
            if feature is None:
                continue
            geometry = feature["geometry"]
            color = basin_colors.get(basin_name, "black")

            if geometry["type"] == "Polygon":
                polys = [geometry["coordinates"]]
            elif geometry["type"] == "MultiPolygon":
                polys = geometry["coordinates"]
            else:
                continue

            for poly_coords in polys:
                exterior = np.array(poly_coords[0])
                ex_lon = exterior[:, 0]
                ex_lat = exterior[:, 1]
                if has_cartopy:
                    ax_map.plot(ex_lon, ex_lat, color=color, linewidth=1.5,
                                transform=ccrs.PlateCarree(), label=basin_name)
                else:
                    ax_map.plot(ex_lon, ex_lat, color=color, linewidth=1.5, label=basin_name)

    cb = fig.colorbar(im, ax=ax_map, orientation="horizontal", pad=0.05, shrink=0.6, aspect=30)
    cb.set_label("Water Balance Residual P-ET-Q (mm/day)", fontsize=11)

    ax_map.set_title("E3SM ELM Water Balance Residual (1985-1989 Climatology)", fontsize=14, fontweight="bold")

    # Add legend for basin outlines (avoid duplicate labels)
    handles_seen = set()
    handles, labels = ax_map.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in handles_seen:
            unique_handles.append(h)
            unique_labels.append(l)
            handles_seen.add(l)
    if unique_handles:
        ax_map.legend(unique_handles, unique_labels, loc="lower left", fontsize=8,
                      framealpha=0.8, ncol=3)

    # ---- Bottom panels: Per-basin bar charts ----
    basin_names_list = list(basins.keys())
    bar_colors = {"P": "#1f77b4", "ET": "#2ca02c", "Q": "#ff7f0e", "R": "#d62728"}

    for idx, basin_name in enumerate(basin_names_list):
        row = 1 + idx // 3
        col = idx % 3
        ax_bar = fig.add_subplot(gs[row, col])

        if basin_name in basin_results:
            vals = basin_results[basin_name]
            components = ["P", "ET", "Q", "R"]
            values = [vals["P"], vals["ET"], vals["Q"], vals["R"]]
            colors = [bar_colors[c] for c in components]

            bars = ax_bar.bar(components, values, color=colors, edgecolor="black", linewidth=0.5)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if np.isfinite(val):
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

            ax_bar.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
            ax_bar.set_title(basin_name, fontsize=12, fontweight="bold")
            ax_bar.set_ylabel("mm/day", fontsize=9)
            ax_bar.tick_params(labelsize=9)
        else:
            ax_bar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_bar.transAxes)
            ax_bar.set_title(basin_name, fontsize=12)

    # Add global means annotation
    fig.text(0.02, 0.98,
             f"Global means (mm/day): P={P_global:.3f}, ET={ET_global:.3f}, Q={Q_global:.3f}, Residual={R_global:.4f}",
             fontsize=10, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    plt.savefig(os.path.join(output_dir, "water_balance_composite.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved water_balance_composite.png")

except Exception as e:
    print(f"  Error creating figure: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 7. Print final summary
# ============================================================
print("\n" + "=" * 60)
print("WATER BALANCE CLOSURE SUMMARY (1985-1989)")
print("=" * 60)
print(f"\nGlobal area-weighted means (mm/day):")
print(f"  Precipitation (P)       = {P_global:.4f}")
print(f"  Evapotranspiration (ET) = {ET_global:.4f}")
print(f"  Total Runoff (Q)        = {Q_global:.4f}")
print(f"  Residual (P - ET - Q)   = {R_global:.4f}")
if P_global > 0:
    print(f"  Residual / P            = {R_global / P_global * 100:.2f}%")

print(f"\nPer-basin water balance (mm/day):")
print(f"  {'Basin':<12s} {'P':>8s} {'ET':>8s} {'Q':>8s} {'Residual':>10s}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for basin_name in basin_names_list:
    if basin_name in basin_results:
        v = basin_results[basin_name]
        print(f"  {basin_name:<12s} {v['P']:8.3f} {v['ET']:8.3f} {v['Q']:8.3f} {v['R']:10.4f}")

print(f"\nOutput files saved to: {output_dir}")
print("Done!")