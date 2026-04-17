import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import shape, Point
from shapely.validation import make_valid
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

# Output directory
outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_06_water_balance/run4_debug/v1/output"
os.makedirs(outdir, exist_ok=True)

# Configuration
case_name = "sample.v3.LR.historical"
elm_dir = "./data/sample/e3sm/lnd"
years = range(1985, 1990)

# Basin definitions
basin_info = {
    "3629000": "Amazon",
    "4121801": "Missouri",
    "4115200": "Columbia",
    "6742900": "Danube",
    "2969100": "Mekong",
    "1159100": "Orange",
}

# ============================================================
# 1. Load monthly ELM data for 1985-1989
# ============================================================
print("Loading ELM monthly data for 1985-1989...")

file_list = []
for year in years:
    for month in range(1, 13):
        fname = os.path.join(elm_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fname):
            file_list.append(fname)
        else:
            print(f"  Warning: missing file {fname}")

print(f"  Found {len(file_list)} files")

try:
    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    print(f"  Dataset dimensions: {dict(ds.dims)}")
    print(f"  Available variables: {list(ds.data_vars)}")
except Exception as e:
    print(f"Error opening ELM files: {e}")
    raise

# ============================================================
# 2. Compute time-mean fields
# ============================================================
print("Computing climatological time-mean fields...")

sec_per_day = 86400.0

P = (ds["RAIN"] + ds["SNOW"]).mean(dim="time") * sec_per_day
P.name = "P"
P.attrs["units"] = "mm/day"
P.attrs["long_name"] = "Precipitation (RAIN+SNOW)"

ET = (ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]).mean(dim="time") * sec_per_day
ET.name = "ET"
ET.attrs["units"] = "mm/day"
ET.attrs["long_name"] = "Evapotranspiration (QVEGE+QVEGT+QSOIL)"

Q = ds["QRUNOFF"].mean(dim="time") * sec_per_day
Q.name = "Q"
Q.attrs["units"] = "mm/day"
Q.attrs["long_name"] = "Total Runoff (QRUNOFF)"

residual = P - ET - Q
residual.name = "residual"
residual.attrs["units"] = "mm/day"
residual.attrs["long_name"] = "Water Balance Residual (P - ET - Q)"

print(f"  P shape: {P.shape}, ET shape: {ET.shape}, Q shape: {Q.shape}")

# ============================================================
# 3. Compute area-weighted global means
# ============================================================
print("Computing area-weighted global means...")

if "area" in ds:
    area = ds["area"].isel(time=0) if "time" in ds["area"].dims else ds["area"]
elif "area" in ds.coords:
    area = ds.coords["area"]
else:
    print("  Warning: 'area' not found")
    area = xr.ones_like(P)

if "landfrac" in ds:
    landfrac = ds["landfrac"].isel(time=0) if "time" in ds["landfrac"].dims else ds["landfrac"]
elif "landfrac" in ds.coords:
    landfrac = ds.coords["landfrac"]
else:
    print("  Warning: 'landfrac' not found, using 1.0")
    landfrac = xr.ones_like(P)

weights = area * landfrac
weights = weights.where(P.notnull())
total_weight = weights.sum()

P_global = float((P * weights).sum() / total_weight)
ET_global = float((ET * weights).sum() / total_weight)
Q_global = float((Q * weights).sum() / total_weight)
res_global = P_global - ET_global - Q_global

print(f"  Global mean P:        {P_global:.4f} mm/day")
print(f"  Global mean ET:       {ET_global:.4f} mm/day")
print(f"  Global mean Q:        {Q_global:.4f} mm/day")
print(f"  Global mean Residual: {res_global:.4f} mm/day")

global_df = pd.DataFrame({
    "Component": ["P (RAIN+SNOW)", "ET (QVEGE+QVEGT+QSOIL)", "Q (QRUNOFF)", "Residual (P-ET-Q)"],
    "Global_Mean_mm_per_day": [P_global, ET_global, Q_global, res_global],
    "Global_Mean_mm_per_year": [P_global * 365.25, ET_global * 365.25, Q_global * 365.25, res_global * 365.25],
})
try:
    global_df.to_csv(os.path.join(outdir, "global_water_balance.csv"), index=False)
    print("  Saved global_water_balance.csv")
except Exception as e:
    print(f"  Error saving global CSV: {e}")

# ============================================================
# 4. Load basin polygons and clip fields to basins
# ============================================================
print("Loading basin polygons...")

try:
    with open("./data/sample/obs/basin_polygons.geojson", "r") as f:
        basins_geojson = json.load(f)
    print(f"  Loaded {len(basins_geojson['features'])} basin features")
except Exception as e:
    print(f"Error loading basin polygons: {e}")
    basins_geojson = {"features": []}

try:
    gauge_meta = pd.read_csv("./data/sample/obs/gauge_metadata.csv")
    print(f"  Gauge metadata: {len(gauge_meta)} gauges")
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    gauge_meta = pd.DataFrame()

# Build basin polygon lookup - fix invalid geometries
basin_polygons = {}
for feature in basins_geojson["features"]:
    gid = str(feature["properties"].get("grdc_no", feature["properties"].get("gauge_id", "")))
    if gid in basin_info:
        try:
            poly = shape(feature["geometry"])
            # Fix invalid geometries
            if not poly.is_valid:
                print(f"  Fixing invalid polygon for {basin_info[gid]} ({gid})")
                try:
                    poly = make_valid(poly)
                except Exception:
                    poly = poly.buffer(0)
            basin_polygons[gid] = poly
            print(f"  Basin {basin_info[gid]} ({gid}): polygon loaded, valid={poly.is_valid}, bounds={poly.bounds}")
        except Exception as e:
            print(f"  Error parsing polygon for {gid}: {e}")

# Get lat/lon arrays
lat_vals = P.coords["lat"].values
lon_vals = P.coords["lon"].values

# Create 2D meshgrid
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

# Normalize longitudes to -180..180 for polygon comparison
lon2d_180 = np.where(lon2d > 180, lon2d - 360, lon2d)

# ============================================================
# 5. Compute basin masks using prepared polygons and vectorized approach
# ============================================================
print("Computing basin masks and clipping fields...")

from shapely.prepared import prep

basin_results = {}

for gid, bname in basin_info.items():
    if gid not in basin_polygons:
        print(f"  Skipping {bname} ({gid}): no polygon found")
        continue

    poly = basin_polygons[gid]
    prepared_poly = prep(poly)
    bounds = poly.bounds  # (minlon, minlat, maxlon, maxlat)

    mask = np.zeros(P.shape, dtype=bool)

    # Prefilter by bounding box with buffer
    lat_mask = (lat2d >= bounds[1] - 1) & (lat2d <= bounds[3] + 1)
    lon_mask = (lon2d_180 >= bounds[0] - 1) & (lon2d_180 <= bounds[2] + 1)
    bbox_mask = lat_mask & lon_mask
    candidate_indices = np.argwhere(bbox_mask)

    print(f"  {bname}: checking {len(candidate_indices)} candidate cells...")

    for idx in candidate_indices:
        i, j = idx[0], idx[1]
        lat_ij = lat2d[i, j]
        lon_ij = lon2d_180[i, j]
        try:
            pt = Point(lon_ij, lat_ij)
            if prepared_poly.contains(pt):
                mask[i, j] = True
        except Exception:
            # Skip cells that cause topology errors
            pass

    mask_da = xr.DataArray(mask, dims=P.dims, coords=P.coords)
    ncells = int(mask_da.sum())
    print(f"  {bname}: {ncells} grid cells in basin")

    if ncells == 0:
        print(f"  Warning: no cells found for {bname}, skipping")
        continue

    basin_weights = weights.where(mask_da)
    bw_sum = float(basin_weights.sum())

    if bw_sum > 0:
        P_basin = float((P.where(mask_da) * basin_weights).sum() / bw_sum)
        ET_basin = float((ET.where(mask_da) * basin_weights).sum() / bw_sum)
        Q_basin = float((Q.where(mask_da) * basin_weights).sum() / bw_sum)
    else:
        P_basin = float(P.where(mask_da).mean())
        ET_basin = float(ET.where(mask_da).mean())
        Q_basin = float(Q.where(mask_da).mean())

    res_basin = P_basin - ET_basin - Q_basin

    basin_results[bname] = {
        "gauge_id": gid,
        "P_mm_day": P_basin,
        "ET_mm_day": ET_basin,
        "Q_mm_day": Q_basin,
        "Residual_mm_day": res_basin,
        "ncells": ncells,
        "mask": mask_da,
    }

    print(f"    P={P_basin:.3f}, ET={ET_basin:.3f}, Q={Q_basin:.3f}, Res={res_basin:.4f} mm/day")

# Save basin results to CSV
basin_rows = []
for bname, bdata in basin_results.items():
    basin_rows.append({
        "Basin": bname,
        "gauge_id": bdata["gauge_id"],
        "P_mm_day": bdata["P_mm_day"],
        "ET_mm_day": bdata["ET_mm_day"],
        "Q_mm_day": bdata["Q_mm_day"],
        "Residual_mm_day": bdata["Residual_mm_day"],
        "P_mm_year": bdata["P_mm_day"] * 365.25,
        "ET_mm_year": bdata["ET_mm_day"] * 365.25,
        "Q_mm_year": bdata["Q_mm_day"] * 365.25,
        "Residual_mm_year": bdata["Residual_mm_day"] * 365.25,
        "ncells": bdata["ncells"],
    })

try:
    basin_df = pd.DataFrame(basin_rows)
    basin_df.to_csv(os.path.join(outdir, "basin_water_balance.csv"), index=False)
    print("  Saved basin_water_balance.csv")
except Exception as e:
    print(f"  Error saving basin CSV: {e}")

# ============================================================
# 6. Save fields to NetCDF
# ============================================================
print("Saving spatial fields to NetCDF...")
try:
    out_ds = xr.Dataset({
        "P": P,
        "ET": ET,
        "Q": Q,
        "residual": residual,
    })
    out_ds.attrs["description"] = "ELM water balance components (1985-1989 climatological mean)"
    out_ds.attrs["units"] = "mm/day"
    out_ds.to_netcdf(os.path.join(outdir, "water_balance_fields.nc"))
    print("  Saved water_balance_fields.nc")
except Exception as e:
    print(f"  Error saving NetCDF: {e}")

# ============================================================
# 7. Create composite figure
# ============================================================
print("Creating composite figure...")

try:
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # ----- Top panel: Global residual map with basin outlines -----
    ax_map = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.coastlines(linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

    # Plot residual field
    plot_lon = lon_vals
    plot_lat = lat_vals
    res_data = residual.values.copy()

    if np.any(plot_lon > 180):
        lon_shifted = np.where(plot_lon > 180, plot_lon - 360, plot_lon)
        sort_idx = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[sort_idx]
        res_data = res_data[:, sort_idx]
    else:
        lon_shifted = plot_lon

    vmax = max(0.1, float(np.nanpercentile(np.abs(res_data[np.isfinite(res_data)]), 95)))
    im = ax_map.pcolormesh(
        lon_shifted, plot_lat, res_data,
        transform=ccrs.PlateCarree(),
        cmap="RdBu",
        vmin=-vmax, vmax=vmax,
        shading="auto",
    )

    cb = plt.colorbar(im, ax=ax_map, orientation="horizontal", pad=0.05, shrink=0.7, aspect=40)
    cb.set_label("Water Balance Residual (P - ET - Q) [mm/day]", fontsize=11)

    # Overlay basin outlines
    basin_colors = {
        "Amazon": "#e41a1c",
        "Missouri": "#377eb8",
        "Columbia": "#4daf4a",
        "Danube": "#984ea3",
        "Mekong": "#ff7f00",
        "Orange": "#a65628",
    }

    for gid, bname in basin_info.items():
        if gid in basin_polygons:
            poly = basin_polygons[gid]
            color = basin_colors.get(bname, "black")

            def plot_polygon_ring(ax, poly_obj, color, label=None):
                if poly_obj.geom_type == "Polygon":
                    coords = np.array(poly_obj.exterior.coords)
                    ax.plot(coords[:, 0], coords[:, 1],
                            transform=ccrs.PlateCarree(), color=color,
                            linewidth=1.5, label=label)
                elif poly_obj.geom_type == "MultiPolygon":
                    for k, p in enumerate(poly_obj.geoms):
                        coords = np.array(p.exterior.coords)
                        ax.plot(coords[:, 0], coords[:, 1],
                                transform=ccrs.PlateCarree(), color=color,
                                linewidth=1.5, label=label if k == 0 else None)
                elif poly_obj.geom_type == "GeometryCollection":
                    first = True
                    for geom in poly_obj.geoms:
                        if geom.geom_type in ("Polygon", "MultiPolygon"):
                            plot_polygon_ring(ax, geom, color, label=label if first else None)
                            first = False

            plot_polygon_ring(ax_map, poly, color, label=bname)

    legend_patches = [
        plt.Line2D([0], [0], color=basin_colors.get(bname, "black"), linewidth=2, label=bname)
        for bname in basin_results.keys()
    ]
    ax_map.legend(handles=legend_patches, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)

    ax_map.set_title(
        f"ELM Water Balance Residual (P - ET - Q)\n"
        f"1985-1989 Climatological Mean | Global mean residual = {res_global:.4f} mm/day",
        fontsize=13, fontweight="bold"
    )

    # ----- Bottom panel: Per-basin bar charts -----
    n_basins = len(basin_results)
    if n_basins > 0:
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, n_basins, subplot_spec=gs[1], wspace=0.35)

        for idx, (bname, bdata) in enumerate(basin_results.items()):
            ax_bar = fig.add_subplot(gs_bottom[idx])

            components = ["P", "ET", "Q", "Residual"]
            values = [bdata["P_mm_day"], bdata["ET_mm_day"], bdata["Q_mm_day"], bdata["Residual_mm_day"]]
            colors_bar = ["#2166ac", "#b2182b", "#1b7837", "#636363"]

            bars = ax_bar.bar(components, values, color=colors_bar, edgecolor="black", linewidth=0.5)

            for bar, val in zip(bars, values):
                ypos = bar.get_height()
                va = "bottom" if ypos >= 0 else "top"
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.3f}",
                    ha="center", va=va, fontsize=7, fontweight="bold"
                )

            ax_bar.axhline(0, color="black", linewidth=0.5)
            ax_bar.set_title(f"{bname}\n(ID: {bdata['gauge_id']})", fontsize=9, fontweight="bold",
                           color=basin_colors.get(bname, "black"))
            ax_bar.set_ylabel("mm/day" if idx == 0 else "", fontsize=9)
            ax_bar.tick_params(axis="x", labelsize=8, rotation=30)
            ax_bar.tick_params(axis="y", labelsize=8)

            ymax = max(values) * 1.3 if max(values) > 0 else 1.0
            ymin = min(0, min(values) * 1.5)
            ax_bar.set_ylim(ymin, ymax)

    plt.savefig(os.path.join(outdir, "water_balance_composite.png"), dpi=200, bbox_inches="tight")
    print("  Saved water_balance_composite.png")
    plt.close()

except Exception as e:
    print(f"Error creating figure: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 8. Summary report
# ============================================================
print("\n" + "=" * 60)
print("WATER BALANCE SUMMARY (1985-1989)")
print("=" * 60)
print(f"\nGlobal Area-Weighted Means (mm/day):")
print(f"  Precipitation (P):        {P_global:.4f}")
print(f"  Evapotranspiration (ET):   {ET_global:.4f}")
print(f"  Total Runoff (Q):          {Q_global:.4f}")
print(f"  Residual (P - ET - Q):     {res_global:.4f}")
print(f"  Residual / P:              {res_global / P_global * 100:.2f}%")
print(f"\nGlobal Area-Weighted Means (mm/year):")
print(f"  Precipitation (P):        {P_global * 365.25:.1f}")
print(f"  Evapotranspiration (ET):   {ET_global * 365.25:.1f}")
print(f"  Total Runoff (Q):          {Q_global * 365.25:.1f}")
print(f"  Residual (P - ET - Q):     {res_global * 365.25:.1f}")

print(f"\nPer-Basin Results (mm/day):")
print(f"  {'Basin':<12s} {'P':>8s} {'ET':>8s} {'Q':>8s} {'Residual':>10s} {'Res/P %':>8s}")
print(f"  {'-' * 56}")
for bname, bdata in basin_results.items():
    pct = bdata["Residual_mm_day"] / bdata["P_mm_day"] * 100 if bdata["P_mm_day"] != 0 else float("inf")
    print(f"  {bname:<12s} {bdata['P_mm_day']:8.3f} {bdata['ET_mm_day']:8.3f} {bdata['Q_mm_day']:8.3f} {bdata['Residual_mm_day']:10.4f} {pct:8.2f}")

print(f"\nOutputs saved to: {outdir}")
print("  - global_water_balance.csv")
print("  - basin_water_balance.csv")
print("  - water_balance_fields.nc")
print("  - water_balance_composite.png")
print("Done.")