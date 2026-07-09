import os
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path

warnings.filterwarnings("ignore")

output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_06_water_balance/run2_debug/v2/output"
os.makedirs(output_dir, exist_ok=True)

case_name = "sample.v3.LR.historical"
lnd_dir = "./data/sample/e3sm/lnd"
basin_file = "./data/sample/obs/basin_polygons.geojson"

basins = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}

MM_S_TO_MM_DAY = 86400.0

# Collect files
files = []
for year in range(1985, 1990):
    for month in range(1, 13):
        fname = os.path.join(lnd_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fname):
            files.append(fname)

print(f"Found {len(files)} ELM files")
if len(files) == 0:
    raise FileNotFoundError("No ELM files found")

# Accumulate means incrementally to avoid memory/time issues
print("Computing time-mean fields incrementally...")
P_sum = None
ET_sum = None
Q_sum = None
count = 0
lat_vals = None
lon_vals = None
area_vals = None
landfrac_vals = None

for i, fname in enumerate(files):
    try:
        with xr.open_dataset(fname, decode_times=False) as d:
            rain = d["RAIN"].values
            snow = d["SNOW"].values
            qvege = d["QVEGE"].values
            qvegt = d["QVEGT"].values
            qsoil = d["QSOIL"].values
            qrunoff = d["QRUNOFF"].values

            # Handle time dim - squeeze if needed
            if rain.ndim > 2:
                rain = rain[0]
                snow = snow[0]
                qvege = qvege[0]
                qvegt = qvegt[0]
                qsoil = qsoil[0]
                qrunoff = qrunoff[0]
            elif rain.ndim == 2 and rain.shape[0] == 1:
                rain = rain[0]
                snow = snow[0]
                qvege = qvege[0]
                qvegt = qvegt[0]
                qsoil = qsoil[0]
                qrunoff = qrunoff[0]

            p = rain + snow
            e = qvege + qvegt + qsoil
            q = qrunoff

            if P_sum is None:
                P_sum = np.zeros_like(p, dtype=np.float64)
                ET_sum = np.zeros_like(p, dtype=np.float64)
                Q_sum = np.zeros_like(p, dtype=np.float64)

            P_sum += np.nan_to_num(p, 0.0)
            ET_sum += np.nan_to_num(e, 0.0)
            Q_sum += np.nan_to_num(q, 0.0)
            count += 1

            if lat_vals is None:
                lat_vals = d["lat"].values
                lon_vals = d["lon"].values
                if "area" in d:
                    av = d["area"].values
                    if av.ndim > lat_vals.ndim:
                        av = av[0] if av.shape[0] == 1 else av
                    area_vals = av
                if "landfrac" in d:
                    lv = d["landfrac"].values
                    if lv.ndim > lat_vals.ndim:
                        lv = lv[0] if lv.shape[0] == 1 else lv
                    landfrac_vals = lv

    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        continue

    if (i + 1) % 12 == 0:
        print(f"  Processed {i+1}/{len(files)} files")

print(f"  Processed {count} files total")

P_vals = (P_sum / count) * MM_S_TO_MM_DAY
ET_vals = (ET_sum / count) * MM_S_TO_MM_DAY
Q_vals = (Q_sum / count) * MM_S_TO_MM_DAY
R_vals = P_vals - ET_vals - Q_vals

data_shape = P_vals.shape
print(f"  Data shape: {data_shape}, lat: {lat_vals.shape}, lon: {lon_vals.shape}")

is_structured = (P_vals.ndim == 2 and lat_vals.ndim == 1 and lon_vals.ndim == 1)

# Global area-weighted means
print("Computing global means...")
if is_structured:
    cos_lat = np.cos(np.deg2rad(lat_vals))
    w = cos_lat[:, np.newaxis] * np.ones((1, len(lon_vals)))
    if landfrac_vals is not None:
        w = w * landfrac_vals
elif area_vals is not None:
    w = area_vals.copy()
    if landfrac_vals is not None:
        w = w * landfrac_vals
else:
    if lat_vals.ndim == 1 and len(lat_vals) == P_vals.shape[0]:
        w = np.cos(np.deg2rad(lat_vals))
    else:
        w = np.ones_like(P_vals)

valid = np.isfinite(P_vals) & (w > 0)
wv = w[valid]
P_global = float(np.average(P_vals[valid], weights=wv))
ET_global = float(np.average(ET_vals[valid], weights=wv))
Q_global = float(np.average(Q_vals[valid], weights=wv))
R_global = P_global - ET_global - Q_global

print(f"  P={P_global:.4f}, ET={ET_global:.4f}, Q={Q_global:.4f}, R={R_global:.4f} mm/day")

pd.DataFrame({
    "Component": ["P", "ET", "Q", "Residual"],
    "Global_Mean_mm_day": [P_global, ET_global, Q_global, R_global]
}).to_csv(os.path.join(output_dir, "global_water_balance_summary.csv"), index=False)

# Flat arrays for basin masking
if is_structured:
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    flat_lon = lon2d.ravel()
    flat_lat = lat2d.ravel()
else:
    flat_lon = lon_vals.ravel()
    flat_lat = lat_vals.ravel()

flat_lon_180 = np.where(flat_lon > 180, flat_lon - 360, flat_lon)
P_flat = P_vals.ravel()
ET_flat = ET_vals.ravel()
Q_flat = Q_vals.ravel()

# Load basins
print("Loading basin polygons...")
geojson = None
features_by_id = {}
try:
    with open(basin_file, "r") as f:
        geojson = json.load(f)
    for feat in geojson["features"]:
        props = feat["properties"]
        gid = str(props.get("grdc_no", props.get("gauge_id", "")))
        features_by_id[gid] = feat
except Exception as e:
    print(f"  Error: {e}")

basin_results = {}
print("Computing basin averages...")
for basin_name, gauge_id in basins.items():
    feat = features_by_id.get(gauge_id)
    if feat is None:
        print(f"  {basin_name}: no polygon")
        basin_results[basin_name] = {"P": np.nan, "ET": np.nan, "Q": np.nan, "R": np.nan}
        continue

    geom = feat["geometry"]
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        basin_results[basin_name] = {"P": np.nan, "ET": np.nan, "Q": np.nan, "R": np.nan}
        continue

    # Detect lon convention
    sample = [c[0] for c in polys[0][0][:5]]
    use_180 = any(s < 0 for s in sample)
    use_lon = flat_lon_180 if use_180 else flat_lon

    mask = np.zeros(len(flat_lon), dtype=bool)
    for poly_c in polys:
        ext = np.array(poly_c[0])
        lon_min, lon_max = ext[:, 0].min() - 1, ext[:, 0].max() + 1
        lat_min, lat_max = ext[:, 1].min() - 1, ext[:, 1].max() + 1
        bbox = (use_lon >= lon_min) & (use_lon <= lon_max) & (flat_lat >= lat_min) & (flat_lat <= lat_max)
        idx = np.where(bbox)[0]
        if len(idx) == 0:
            continue
        path = Path(ext)
        pts = np.column_stack([use_lon[idx], flat_lat[idx]])
        inside = path.contains_points(pts)
        mask[idx[inside]] = True

    nc = mask.sum()
    print(f"  {basin_name}: {nc} cells")
    if nc == 0:
        basin_results[basin_name] = {"P": np.nan, "ET": np.nan, "Q": np.nan, "R": np.nan}
        continue

    wb = np.cos(np.deg2rad(flat_lat[mask]))
    vb = np.isfinite(P_flat[mask]) & (wb > 0)
    if vb.sum() > 0:
        Pa = np.average(P_flat[mask][vb], weights=wb[vb])
        Ea = np.average(ET_flat[mask][vb], weights=wb[vb])
        Qa = np.average(Q_flat[mask][vb], weights=wb[vb])
    else:
        Pa = np.nanmean(P_flat[mask])
        Ea = np.nanmean(ET_flat[mask])
        Qa = np.nanmean(Q_flat[mask])
    Ra = Pa - Ea - Qa
    basin_results[basin_name] = {"P": Pa, "ET": Ea, "Q": Qa, "R": Ra}
    print(f"    P={Pa:.3f}, ET={Ea:.3f}, Q={Qa:.3f}, R={Ra:.4f}")

pd.DataFrame(basin_results).T.rename(columns={"P": "P_mm_day", "ET": "ET_mm_day", "Q": "Q_mm_day", "R": "Residual_mm_day"}).to_csv(
    os.path.join(output_dir, "basin_water_balance.csv"), index_label="Basin")

# Save NetCDF
print("Saving NetCDF...")
try:
    if is_structured:
        coords = {"lat": lat_vals, "lon": lon_vals}
        dims = ["lat", "lon"]
    else:
        dims = list(range(P_vals.ndim))
        coords = {"lat": (dims, lat_vals), "lon": (dims, lon_vals)}

    out_ds = xr.Dataset({
        "P": (["lat", "lon"] if is_structured else dims, P_vals, {"units": "mm/day"}),
        "ET": (["lat", "lon"] if is_structured else dims, ET_vals, {"units": "mm/day"}),
        "Q": (["lat", "lon"] if is_structured else dims, Q_vals, {"units": "mm/day"}),
        "residual": (["lat", "lon"] if is_structured else dims, R_vals, {"units": "mm/day"}),
    }, coords=coords if is_structured else {})
    out_ds.to_netcdf(os.path.join(output_dir, "water_balance_fields.nc"))
    print("  Saved")
except Exception as e:
    print(f"  Error: {e}")

# Plot
print("Creating figure...")
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    has_cartopy = True
except ImportError:
    has_cartopy = False

try:
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.35)

    if has_cartopy:
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    else:
        ax_map = fig.add_subplot(gs[0, :])

    if is_structured:
        plon = lon_vals.copy()
        pdata = R_vals.copy()
        if np.nanmax(plon) > 180:
            plon = np.where(plon > 180, plon - 360, plon)
            si = np.argsort(plon)
            plon = plon[si]
            pdata = pdata[:, si]
        finite = pdata[np.isfinite(pdata)]
        vmax = float(np.percentile(np.abs(finite), 95)) if len(finite) > 0 else 1.0
        vmax = min(max(vmax, 0.01), 2.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        if has_cartopy:
            im = ax_map.pcolormesh(plon, lat_vals, pdata, transform=ccrs.PlateCarree(),
                                   cmap="RdBu", norm=norm, shading="auto")
            ax_map.coastlines(linewidth=0.5)
            ax_map.set_global()
        else:
            im = ax_map.pcolormesh(plon, lat_vals, pdata, cmap="RdBu", norm=norm, shading="auto")
    else:
        finite = R_vals.ravel()[np.isfinite(R_vals.ravel())]
        vmax = float(np.percentile(np.abs(finite), 95)) if len(finite) > 0 else 1.0
        vmax = min(max(vmax, 0.01), 2.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        if has_cartopy:
            im = ax_map.scatter(flat_lon_180, flat_lat, c=R_vals.ravel(), s=0.5,
                                transform=ccrs.PlateCarree(), cmap="RdBu", norm=norm)
            ax_map.coastlines(linewidth=0.5)
            ax_map.set_global()
        else:
            im = ax_map.scatter(flat_lon_180, flat_lat, c=R_vals.ravel(), s=0.5, cmap="RdBu", norm=norm)

    bcols = {"Amazon": "green", "Missouri": "blue", "Columbia": "purple",
             "Danube": "orange", "Mekong": "red", "Orange": "brown"}
    drawn = set()
    for bn, gid in basins.items():
        feat = features_by_id.get(gid)
        if feat is None:
            continue
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            ps = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            ps = geom["coordinates"]
        else:
            continue
        lbl = bn if bn not in drawn else None
        drawn.add(bn)
        for pc in ps:
            ext = np.array(pc[0])
            if has_cartopy:
                ax_map.plot(ext[:, 0], ext[:, 1], color=bcols.get(bn, "k"), lw=1.5,
                            transform=ccrs.PlateCarree(), label=lbl)
            else:
                ax_map.plot(ext[:, 0], ext[:, 1], color=bcols.get(bn, "k"), lw=1.5, label=lbl)
            lbl = None

    cb = fig.colorbar(im, ax=ax_map, orientation="horizontal", pad=0.05, shrink=0.6)
    cb.set_label("Residual P-ET-Q (mm/day)")
    ax_map.set_title("E3SM ELM Water Balance Residual (1985-1989)", fontsize=14, fontweight="bold")
    h, l = ax_map.get_legend_handles_labels()
    if h:
        ax_map.legend(h, l, loc="lower left", fontsize=8, framealpha=0.8, ncol=3)

    bcolors = {"P": "#1f77b4", "ET": "#2ca02c", "Q": "#ff7f0e", "R": "#d62728"}
    for idx, bn in enumerate(basins.keys()):
        ax = fig.add_subplot(gs[1 + idx // 3, idx % 3])
        v = basin_results.get(bn, {})
        comps = ["P", "ET", "Q", "R"]
        vals = [v.get("P", np.nan), v.get("ET", np.nan), v.get("Q", np.nan), v.get("R", np.nan)]
        bars = ax.bar(comps, vals, color=[bcolors[c] for c in comps], edgecolor="k", lw=0.5)
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(bn, fontsize=12, fontweight="bold")
        ax.set_ylabel("mm/day", fontsize=9)

    fig.text(0.02, 0.98,
             f"Global (mm/day): P={P_global:.3f}, ET={ET_global:.3f}, Q={Q_global:.3f}, R={R_global:.4f}",
             fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    plt.savefig(os.path.join(output_dir, "water_balance_composite.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figure")
except Exception as e:
    print(f"  Figure error: {e}")
    import traceback
    traceback.print_exc()

print(f"\nDone. Output in {output_dir}")