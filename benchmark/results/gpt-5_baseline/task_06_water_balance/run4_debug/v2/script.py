import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import shape as shapely_shape, Point as ShapelyPoint
from shapely.ops import unary_union
from shapely.prepared import prep as shapely_prep

def main():
    # User-specified output directory
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_06_water_balance/run4_debug/v2/output"
    os.makedirs(output_dir, exist_ok=True)

    # Paths and constants
    base_lnd = "./data/sample/e3sm/lnd"
    case = "sample.v3.LR.historical"
    start_year, end_year = 1985, 1989
    basin_geojson_path = "./data/sample/obs/basin_polygons.geojson"

    # Target basins: mapping from GRDC id to name
    target_basins = {
        "3629000": "Amazon",
        "4121801": "Missouri",
        "4115200": "Columbia",
        "6742900": "Danube",
        "2969100": "Mekong",
        "1159100": "Orange",
    }

    # 1) Gather ELM files for 1985-1989
    files = []
    for y in range(start_year, end_year + 1):
        pattern = os.path.join(base_lnd, f"{case}.elm.h0.{y:04d}-*.nc")
        files.extend(sorted(glob.glob(pattern)))

    if not files:
        print("No ELM files found for the specified period. Check the data path and filenames.")
        return

    # 2) Open dataset
    try:
        ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)
    except Exception as e:
        print(f"Error opening ELM files: {e}")
        return

    # Helper: find lat/lon coordinates
    def get_coord(ds, names):
        for n in names:
            if n in ds.coords:
                return ds[n]
            if n in ds.variables and np.ndim(ds[n]) >= 1:
                return ds[n]
        return None

    lat_da = get_coord(ds, ["lat", "latitude", "LAT", "nav_lat", "yc", "y", "lsmlat"])
    lon_da = get_coord(ds, ["lon", "longitude", "LON", "nav_lon", "xc", "x", "lsmlon"])

    if lat_da is None or lon_da is None:
        print("Could not find latitude/longitude coordinates in ELM dataset.")
        return

    # Normalize longitudes to [-180, 180] for spatial operations
    def normalize_lon(lon_vals):
        lon = lon_vals.copy()
        lon = np.where(lon > 180, lon - 360, lon)
        return lon

    # Build 2D lat/lon arrays if necessary
    if lat_da.ndim == 1 and lon_da.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_da.values, lat_da.values)
    else:
        lat2d = lat_da.values
        lon2d = lon_da.values

    lon2d_norm = normalize_lon(lon2d)

    # 3) Compute monthly components and climatological means
    # Units:
    #   - RAIN + SNOW: kg/m2/s -> mm/s (1 kg/m2 = 1 mm), then to mm/day by * 86400
    #   - QVEGE/QVEGT/QSOIL: mm/s -> mm/day
    #   - QRUNOFF: mm/s -> mm/day
    def get_var(ds, name):
        if name in ds.variables:
            return ds[name]
        else:
            raise KeyError(f"Variable '{name}' not found in dataset.")

    try:
        RAIN = get_var(ds, "RAIN")
        SNOW = get_var(ds, "SNOW")
        QVEGE = get_var(ds, "QVEGE")
        QVEGT = get_var(ds, "QVEGT")
        QSOIL = get_var(ds, "QSOIL")
        QRUNOFF = get_var(ds, "QRUNOFF")
    except KeyError as e:
        print(str(e))
        return

    seconds_per_day = 86400.0

    # Convert to mm/day
    P_ts = (RAIN + SNOW) * seconds_per_day
    ET_ts = (QVEGE + QVEGT + QSOIL) * seconds_per_day
    Q_ts = QRUNOFF * seconds_per_day

    # Climatological mean over time (1985-1989 monthly)
    P_mean = P_ts.mean(dim="time", skipna=True)
    ET_mean = ET_ts.mean(dim="time", skipna=True)
    Q_mean = Q_ts.mean(dim="time", skipna=True)
    residual = P_mean - ET_mean - Q_mean

    # 4) Compute area weights for global mean
    # Prefer 'area' if available; otherwise use cos(lat) weighting
    area_var = None
    for cand in ["area", "cell_area", "areacella", "area2d"]:
        if cand in ds.variables:
            area_var = ds[cand]
            break

    def to_numpy(da):
        try:
            return da.values
        except Exception:
            return np.array(da)

    # Figure out field dims and coordinate arrays
    if P_mean.ndim != 2:
        print("Unexpected field dimensionality (not 2D).")
        return
    dims = P_mean.dims
    coord0_vals = P_mean[dims[0]].values if dims[0] in P_mean.coords else np.arange(P_mean.sizes[dims[0]])
    coord1_vals = P_mean[dims[1]].values if dims[1] in P_mean.coords else np.arange(P_mean.sizes[dims[1]])

    if area_var is not None:
        weights = area_var
        # Broadcast weights to field shapes if needed
        try:
            weights = weights.broadcast_like(P_mean)
        except Exception:
            # Attempt simple rename based on length match
            wdims = list(weights.dims)
            mapping = {}
            for di in dims:
                for wdi in wdims:
                    if ds.dims.get(wdi, None) == P_mean.sizes[di] or weights.sizes[wdi] == P_mean.sizes[di]:
                        if wdi not in mapping.values():
                            mapping[wdi] = di
                            break
            try:
                weights = weights.rename(mapping).broadcast_like(P_mean)
            except Exception as e:
                print(f"Failed to align area weights to field: {e}")
                # Fallback to cosine weighting
                area_var = None

    if area_var is None:
        # Cosine latitude weighting
        if lat_da.ndim == 1 and lon_da.ndim == 1:
            lat_vals = lat_da.values
            lon_vals = lon_da.values
            # Determine which field dim corresponds to lat/lon
            lat_dim = None
            lon_dim = None
            # Try to match by coordinate equality
            for d in dims:
                if d in P_mean.coords:
                    cv = P_mean[d].values
                    if cv.shape == lat_vals.shape and np.allclose(cv, lat_vals, equal_nan=True):
                        lat_dim = d
                    if cv.shape == lon_vals.shape and np.allclose(cv, lon_vals, equal_nan=True):
                        lon_dim = d
            # Fallback by length if needed
            if lat_dim is None or lon_dim is None:
                if P_mean.sizes[dims[0]] == len(lat_vals) and P_mean.sizes[dims[1]] == len(lon_vals):
                    lat_dim, lon_dim = dims[0], dims[1]
                elif P_mean.sizes[dims[0]] == len(lon_vals) and P_mean.sizes[dims[1]] == len(lat_vals):
                    lat_dim, lon_dim = dims[1], dims[0]
                else:
                    # As a last resort, assume dims[0]=lat, dims[1]=lon
                    lat_dim, lon_dim = dims[0], dims[1]
            # Build weights as 2D array with correct orientation
            lat_axis_len = P_mean.sizes[lat_dim]
            lon_axis_len = P_mean.sizes[lon_dim]
            lat_w = np.cos(np.deg2rad(lat_vals.reshape(-1)))
            if lat_axis_len != len(lat_w):
                # interpolate or broadcast if mismatch; fallback to using field's coordinate values
                lat_coord_vals = P_mean[lat_dim].values
                lat_w = np.cos(np.deg2rad(lat_coord_vals.reshape(-1)))
            w2d = np.broadcast_to(lat_w[:, None], (lat_axis_len, lon_axis_len))
            # Place into DataArray with dims order
            if (lat_dim, lon_dim) == dims:
                w_arr = w2d
            else:
                # transpose if lat is dims[1]
                w_arr = w2d.T
            weights = xr.DataArray(
                w_arr,
                dims=dims,
                coords={dims[0]: (dims[0], coord0_vals), dims[1]: (dims[1], coord1_vals)},
            )
        else:
            # 2D lat
            w = np.cos(np.deg2rad(lat_da.values))
            weights = xr.DataArray(
                w,
                dims=dims,
                coords={dims[0]: (dims[0], coord0_vals), dims[1]: (dims[1], coord1_vals)},
            )

    def weighted_global_mean(field, weights):
        # Align weights to field
        try:
            w = weights.broadcast_like(field)
        except Exception:
            w = weights
        # Mask invalids
        w = w.where(np.isfinite(field))
        num = (field * w).sum(skipna=True)  # sum over all dims
        den = w.sum(skipna=True)
        gm_da = num / den
        gm_da = gm_da.compute()
        gm_vals = gm_da.values
        # Reduce to scalar
        return float(np.asarray(gm_vals).squeeze())

    global_means = {
        "P_mm_day": weighted_global_mean(P_mean, weights),
        "ET_mm_day": weighted_global_mean(ET_mean, weights),
        "Q_mm_day": weighted_global_mean(Q_mean, weights),
        "Residual_mm_day": weighted_global_mean(residual, weights),
    }

    # 5) Read basin polygons and prepare mask function
    try:
        with open(basin_geojson_path, "r") as f:
            geo = json.load(f)
    except Exception as e:
        print(f"Error reading basin polygons: {e}")
        return

    # Map basin id -> unified shapely geometry
    basin_geoms = {}
    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        grdc_no = str(props.get("grdc_no", "")).strip()
        if grdc_no in target_basins:
            try:
                geom = shapely_shape(feat.get("geometry"))
            except Exception:
                continue
            if grdc_no in basin_geoms:
                basin_geoms[grdc_no].append(geom)
            else:
                basin_geoms[grdc_no] = [geom]

    # Union geometries for each basin id
    for k, geolist in list(basin_geoms.items()):
        try:
            basin_geoms[k] = unary_union(geolist)
        except Exception:
            basin_geoms[k] = geolist[0]

    # Helper: compute basin mask based on point-in-polygon at grid-cell centers
    lat2d_vals = np.array(lat2d, dtype=float)
    lon2d_vals = np.array(lon2d_norm, dtype=float)

    # Build an xarray dataset for climatology fields for saving and convenience
    clim_ds = xr.Dataset(
        data_vars=dict(
            P=(dims, P_mean.values),
            ET=(dims, ET_mean.values),
            Q=(dims, Q_mean.values),
            residual=(dims, residual.values),
        ),
        coords={dims[0]: (dims[0], coord0_vals), dims[1]: (dims[1], coord1_vals)}
    )
    clim_ds["P"].attrs["units"] = "mm/day"
    clim_ds["ET"].attrs["units"] = "mm/day"
    clim_ds["Q"].attrs["units"] = "mm/day"
    clim_ds["residual"].attrs["units"] = "mm/day"

    # Save climatology to NetCDF
    nc_out = os.path.join(output_dir, f"elm_climatology_{start_year}-{end_year}.nc")
    try:
        clim_ds.to_netcdf(nc_out)
        print(f"Saved climatology to {nc_out}")
    except Exception as e:
        print(f"Error saving climatology NetCDF: {e}")

    # Save global means to CSV
    global_csv = os.path.join(output_dir, f"global_means_{start_year}-{end_year}.csv")
    try:
        pd.DataFrame([global_means]).to_csv(global_csv, index=False)
        print(f"Saved global means to {global_csv}")
    except Exception as e:
        print(f"Error saving global means CSV: {e}")

    # 6) Compute basin area-weighted means for P, ET, Q, residual
    def basin_mask_from_geom(geom):
        prep_geom = shapely_prep(geom)
        minx, miny, maxx, maxy = geom.bounds
        in_bbox = (lon2d_vals >= minx) & (lon2d_vals <= maxx) & (lat2d_vals >= miny) & (lat2d_vals <= maxy)
        mask = np.zeros(lon2d_vals.shape, dtype=bool)
        idxs = np.argwhere(in_bbox)
        for ii, jj in idxs:
            p = ShapelyPoint(float(lon2d_vals[ii, jj]), float(lat2d_vals[ii, jj]))
            if prep_geom.contains(p) or prep_geom.intersects(p):
                mask[ii, jj] = True
        return mask

    # Convert fields to numpy
    P_np = P_mean.values
    ET_np = ET_mean.values
    Q_np = Q_mean.values
    R_np = residual.values
    # Weights to numpy
    weights_aligned = weights
    try:
        weights_aligned = weights_aligned.broadcast_like(P_mean)
    except Exception:
        pass
    weights_np = weights_aligned.values

    basin_rows = []
    for bid, bname in target_basins.items():
        if bid not in basin_geoms:
            print(f"Warning: Basin geometry for {bname} (id={bid}) not found in GeoJSON.")
            row = dict(basin_id=bid, basin_name=bname,
                       P_mm_day=np.nan, ET_mm_day=np.nan, Q_mm_day=np.nan, Residual_mm_day=np.nan)
            basin_rows.append(row)
            continue

        geom = basin_geoms[bid]
        try:
            mask = basin_mask_from_geom(geom)
        except Exception as e:
            print(f"Error computing mask for basin {bname} ({bid}): {e}")
            mask = None

        if mask is None or not np.any(mask):
            print(f"Warning: Empty mask for basin {bname} ({bid}).")
            row = dict(basin_id=bid, basin_name=bname,
                       P_mm_day=np.nan, ET_mm_day=np.nan, Q_mm_day=np.nan, Residual_mm_day=np.nan)
            basin_rows.append(row)
            continue

        valid = mask & np.isfinite(P_np) & np.isfinite(weights_np)
        wmask = np.where(valid, weights_np, 0.0)
        wsum = wmask.sum()
        if wsum == 0:
            p_mean_b = et_mean_b = q_mean_b = r_mean_b = np.nan
        else:
            p_mean_b = np.nansum(P_np * wmask) / wsum
            et_mean_b = np.nansum(ET_np * wmask) / wsum
            q_mean_b = np.nansum(Q_np * wmask) / wsum
            r_mean_b = np.nansum(R_np * wmask) / wsum

        row = dict(basin_id=bid, basin_name=bname,
                   P_mm_day=float(p_mean_b), ET_mm_day=float(et_mean_b),
                   Q_mm_day=float(q_mean_b), Residual_mm_day=float(r_mean_b))
        basin_rows.append(row)

    basin_df = pd.DataFrame(basin_rows)
    basin_csv = os.path.join(output_dir, f"basin_means_{start_year}-{end_year}.csv")
    try:
        basin_df.to_csv(basin_csv, index=False)
        print(f"Saved basin means to {basin_csv}")
    except Exception as e:
        print(f"Error saving basin means CSV: {e}")

    # 7) Plot composite figure
    fig_path = os.path.join(output_dir, f"water_balance_closure_{start_year}-{end_year}.png")
    try:
        resid_vals = R_np
        finite_vals = resid_vals[np.isfinite(resid_vals)]
        if finite_vals.size > 0:
            vmax = np.nanpercentile(np.abs(finite_vals), 98)
            if vmax == 0 or np.isnan(vmax):
                vmax = np.nanmax(np.abs(finite_vals)) if finite_vals.size > 0 else 1.0
        else:
            vmax = 1.0
        vmin = -vmax

        plt.close("all")
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[2.2, 1, 1], hspace=0.35, wspace=0.25)

        # Top panel: global residual map
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.coastlines(linewidth=0.7)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax_map.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5)

        pc = ax_map.pcolormesh(lon2d_norm, lat2d_vals, resid_vals,
                               transform=ccrs.PlateCarree(),
                               cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        cb = fig.colorbar(pc, ax=ax_map, orientation="horizontal", pad=0.02, fraction=0.05)
        cb.set_label("Residual (P - ET - Q) [mm/day]")

        # Draw basin outlines
        for bid, geom in basin_geoms.items():
            try:
                ax_map.add_geometries([geom], crs=ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="k", linewidth=1.0, alpha=0.8)
            except Exception:
                continue

        ax_map.set_title(f"E3SM ELM Water Balance Residual (1985-1989 mean)")

        # Bottom panels: per-basin bar charts (6 basins across two rows)
        ordered_rows = [r for bid in target_basins.keys() for r in basin_rows if r["basin_id"] == bid]
        bar_labels = ["P", "ET", "Q", "Residual"]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        for idx, row in enumerate(ordered_rows):
            r = idx // 3 + 1  # rows 1 and 2 in gridspec (indices 1 and 2)
            c = idx % 3
            ax = fig.add_subplot(gs[r, c])
            values = [row["P_mm_day"], row["ET_mm_day"], row["Q_mm_day"], row["Residual_mm_day"]]
            ax.bar(bar_labels, values, color=colors)
            ax.set_ylabel("mm/day")
            ax.set_title(f"{row['basin_name']} ({row['basin_id']})")
            ax.axhline(0, color="k", linewidth=0.5)
            ax.tick_params(axis='x', rotation=0)

        gm_text = f"Global means (mm/day): P={global_means['P_mm_day']:.3f}, ET={global_means['ET_mm_day']:.3f}, Q={global_means['Q_mm_day']:.3f}, Resid={global_means['Residual_mm_day']:.3f}"
        fig.suptitle(gm_text, y=0.98, fontsize=12)

        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved composite figure to {fig_path}")
    except Exception as e:
        print(f"Error creating/saving figure: {e}")

    print("Analysis complete.")

if __name__ == "__main__":
    main()