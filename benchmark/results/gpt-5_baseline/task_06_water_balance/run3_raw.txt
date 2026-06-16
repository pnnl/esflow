import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def main():
    # Configuration
    case_name = "sample.v3.LR.historical"
    data_dir = "./data/sample/e3sm"
    lnd_dir = os.path.join(data_dir, "lnd")
    basin_geojson_path = "./data/sample/obs/basin_polygons.geojson"
    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"  # not strictly required but kept for completeness
    start_year, end_year = 1985, 1989
    # Output directory (as requested by user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_06_water_balance/run3_output"
    os.makedirs(output_dir, exist_ok=True)

    # Basins of interest: {name: grdc_no}
    basins = {
        "Amazon": 3629000,
        "Missouri": 4121801,
        "Columbia": 4115200,
        "Danube": 6742900,
        "Mekong": 2969100,
        "Orange": 1159100,
    }

    # 1) Gather ELM files for 1985-1989 monthly
    file_pattern = os.path.join(
        lnd_dir, f"{case_name}.elm.h0.*.nc"
    )
    all_files = sorted(glob.glob(file_pattern))
    sel_files = []
    for fp in all_files:
        # parse year and month from filename
        # expected format: {case}.elm.h0.YYYY-MM.nc
        base = os.path.basename(fp)
        parts = base.split(".")
        # find chunk containing YYYY-MM
        ym = None
        for p in parts:
            if "-" in p and len(p) == 7:
                ym = p
                break
        if ym is None:
            continue
        try:
            year = int(ym.split("-")[0])
            if start_year <= year <= end_year:
                sel_files.append(fp)
        except Exception:
            continue

    if len(sel_files) == 0:
        print("No ELM files found for the specified period.")
        return

    print(f"Found {len(sel_files)} ELM monthly files for {start_year}-{end_year}")

    # 2) Open dataset
    try:
        ds = xr.open_mfdataset(sel_files, combine="by_coords", decode_times=True)
    except Exception as e:
        print(f"Failed to open dataset: {e}")
        return

    # 3) Ensure time slicing
    try:
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    except Exception as e:
        print(f"Failed to slice time: {e}")
        return

    # 4) Compute P, ET, Q in mm/day
    # Variables:
    # P = RAIN + SNOW (kg/m2/s -> mm/s)
    # ET = QVEGE + QVEGT + QSOIL (mm/s)
    # Q = QRUNOFF (mm/s)
    sec_per_day = 86400.0

    def get_var_safe(dset, varname):
        if varname in dset:
            return dset[varname]
        # try case-insensitive
        for k in dset.data_vars:
            if k.lower() == varname.lower():
                return dset[k]
        raise KeyError(f"Variable {varname} not found in dataset")

    try:
        RAIN = get_var_safe(ds, "RAIN")
        SNOW = get_var_safe(ds, "SNOW")
        QVEGE = get_var_safe(ds, "QVEGE")
        QVEGT = get_var_safe(ds, "QVEGT")
        QSOIL = get_var_safe(ds, "QSOIL")
        QRUNOFF = get_var_safe(ds, "QRUNOFF")
    except KeyError as e:
        print(str(e))
        return

    # Convert and sum
    P = (RAIN + SNOW) * sec_per_day  # mm/day
    ET = (QVEGE + QVEGT + QSOIL) * sec_per_day  # mm/day
    Q = QRUNOFF * sec_per_day  # mm/day

    # 5) Time mean over the 5-year period
    P_mean = P.mean(dim="time", skipna=True)
    ET_mean = ET.mean(dim="time", skipna=True)
    Q_mean = Q.mean(dim="time", skipna=True)
    RES_mean = P_mean - ET_mean - Q_mean

    # 6) Get lat/lon and area weights
    # Try to find lat/lon variables and area
    def get_lat_lon(dset, da):
        # Identify spatial dims (exclude 'time' and any 'ht' or 'lev' if present)
        spatial_dims = [d for d in da.dims if d not in ("time", "lev", "z", "z_t", "z_w")]
        lat = None
        lon = None

        # Direct coordinates
        for cand in ["lat", "latitude", "LAT", "Latitude"]:
            if cand in dset.coords or cand in dset.variables:
                lat = dset[cand]
                break
        for cand in ["lon", "longitude", "LON", "Longitude"]:
            if cand in dset.coords or cand in dset.variables:
                lon = dset[cand]
                break

        # Special case: unstructured grid with ncol
        if lat is None and "ncol" in dset.dims and "lat" in dset:
            lat = dset["lat"]
        if lon is None and "ncol" in dset.dims and "lon" in dset:
            lon = dset["lon"]

        # Try to align shapes to data
        # If lat/lon are 1D and match one of the dims
        if lat is not None and lon is not None:
            # Ensure lat/lon in the same shape as data (broadcast if necessary)
            try:
                lat_b = xr.broadcast(lat, da.isel({dim: 0 for dim in da.dims if dim != lat.dims[0]}) if len(lat.dims)==1 else lat)[0]
            except Exception:
                lat_b = lat
            try:
                lon_b = xr.broadcast(lon, da.isel({dim: 0 for dim in da.dims if dim != lon.dims[0]}) if len(lon.dims)==1 else lon)[0]
            except Exception:
                lon_b = lon
            # Try simple broadcasting to da
            try:
                lat2d, lon2d = xr.broadcast(lat_b, lon_b)
            except Exception:
                # fallback: broadcast to DA
                lat2d, lon2d = xr.broadcast(da, lat)[1], xr.broadcast(da, lon)[1]
            # Finally align to da
            lat2d, lon2d = xr.align(lat2d, da, join="override")[0], xr.align(lon2d, da, join="override")[0]
            return lat2d, lon2d

        # Fallback: try to find in dataarray's coords
        for c in da.coords:
            if c.lower() in ("lat", "latitude"):
                lat = da[c]
            if c.lower() in ("lon", "longitude"):
                lon = da[c]
        if lat is not None and lon is not None:
            lat2d, lon2d = xr.broadcast(lat, lon)
            lat2d, lon2d = xr.align(lat2d, da, join="override")[0], xr.align(lon2d, da, join="override")[0]
            return lat2d, lon2d

        raise ValueError("Could not determine latitude/longitude coordinates")

    try:
        lat2d, lon2d = get_lat_lon(ds, P_mean)
    except Exception as e:
        print(f"Failed to get lat/lon: {e}")
        return

    # Normalize longitudes to [-180, 180]
    def wrap_lon(lon):
        arr = lon.copy()
        arr = ((arr + 180) % 360) - 180
        return arr

    try:
        lon2d_wrapped = xr.apply_ufunc(wrap_lon, lon2d)
    except Exception:
        lon2d_wrapped = (((lon2d + 180) % 360) - 180)

    # Compute area weights
    def compute_area_from_coords(lat, lon):
        # If lat, lon are 1D: compute using edges
        if (lat.ndim == 1) and (lon.ndim == 1):
            lat1d = lat.values
            lon1d = lon.values
            # Compute edges
            dlat = np.diff(lat1d).mean() if len(lat1d) > 1 else 1.0
            dlon = np.diff(lon1d).mean() if len(lon1d) > 1 else 1.0
            lat_edges = np.concatenate([
                [lat1d[0] - dlat/2],
                (lat1d[:-1] + lat1d[1:])/2,
                [lat1d[-1] + dlat/2]
            ])
            lon_edges = np.concatenate([
                [lon1d[0] - dlon/2],
                (lon1d[:-1] + lon1d[1:])/2,
                [lon1d[-1] + dlon/2]
            ])
            # Area per cell
            Re = 6371000.0
            # Convert to radians
            lat_edges_rad = np.deg2rad(lat_edges)
            lon_edges_rad = np.deg2rad(lon_edges)
            # 2D arrays of edges
            dlon2d = np.diff(lon_edges_rad)
            dlat2d = np.diff(lat_edges_rad)
            # Compute area via spherical trapezoid formula:
            # area = R^2 * dlon * (sin(lat2) - sin(lat1))
            area = np.outer(np.ones(len(lat1d)), dlon2d) * 0.0
            area = np.zeros((len(lat1d), len(lon1d)))
            for i in range(len(lat1d)):
                sin_lat2 = np.sin(lat_edges_rad[i+1])
                sin_lat1 = np.sin(lat_edges_rad[i])
                area[i, :] = (Re**2) * (sin_lat2 - sin_lat1) * dlon2d
            return area  # in m^2
        else:
            # For 2D or unstructured: approximate using cos(lat)
            Re = 6371000.0
            total_area = 4 * np.pi * Re**2
            w = np.cos(np.deg2rad(lat.values))
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.maximum(0.0, w)
            w_sum = np.nansum(w)
            if w_sum == 0:
                return np.ones_like(lat.values)
            area = w / w_sum * total_area
            return area

    # Try to read 'area' variable
    area_weights = None
    for vname in ["area", "AREA", "landarea", "land_area", "gridcell_area"]:
        if vname in ds.variables:
            try:
                area_var = ds[vname]
                area_weights = area_var
                # Ensure units in m^2
                units = area_var.attrs.get("units", "").lower()
                if "km" in units:
                    area_weights = area_var * 1e6
                elif units in ["m2", "m^2", "meter2", "meters2"]:
                    area_weights = area_var
                elif units == "" and np.nanmax(area_var.values) < 1e5:
                    # If suspiciously small, assume it's in km^2
                    area_weights = area_var * 1e6
                else:
                    area_weights = area_var
                # Broadcast to P_mean shape if necessary
                try:
                    area_weights, _ = xr.broadcast(area_weights, P_mean)
                except Exception:
                    pass
                break
            except Exception:
                area_weights = None

    if area_weights is None:
        # compute from coords
        try:
            # If lat/lon are 1D coordinate variables
            lat_coords = None
            lon_coords = None
            if "lat" in ds.coords and ds["lat"].ndim == 1 and "lon" in ds.coords and ds["lon"].ndim == 1:
                lat_coords = ds["lat"]
                lon_coords = ds["lon"]
            elif "latitude" in ds.coords and ds["latitude"].ndim == 1 and "longitude" in ds.coords and ds["longitude"].ndim == 1:
                lat_coords = ds["latitude"]
                lon_coords = ds["longitude"]
            if lat_coords is not None and lon_coords is not None:
                area_array = compute_area_from_coords(lat_coords, lon_coords)
                area_weights = xr.DataArray(
                    area_array,
                    coords={lat_coords.dims[0]: lat_coords, lon_coords.dims[0]: lon_coords},
                    dims=(lat_coords.dims[0], lon_coords.dims[0]),
                    name="area_weights",
                    attrs={"units": "m2"},
                )
                # Align to P_mean
                area_weights, _ = xr.broadcast(area_weights, P_mean)
            else:
                # 2D or unstructured: approximate
                area_array = compute_area_from_coords(lat2d, lon2d)
                area_weights = xr.DataArray(
                    area_array,
                    coords=P_mean.coords,
                    dims=P_mean.dims,
                    name="area_weights",
                    attrs={"units": "m2"},
                )
        except Exception as e:
            print(f"Failed to compute area weights, using unit weights: {e}")
            area_weights = xr.ones_like(P_mean)

    # Ensure lons wrapped for plotting and clipping
    lon2d_wrapped_np = np.array(lon2d_wrapped.values)
    lat2d_np = np.array(lat2d.values)

    # 7) Compute area-weighted global means
    def area_weighted_mean(field, area):
        f = field.where(np.isfinite(field))
        a = area.where(np.isfinite(field))
        num = (f * a).sum(skipna=True)
        den = a.sum(skipna=True)
        return float((num / den).values)

    gm_P = area_weighted_mean(P_mean, area_weights)
    gm_ET = area_weighted_mean(ET_mean, area_weights)
    gm_Q = area_weighted_mean(Q_mean, area_weights)
    gm_RES = area_weighted_mean(RES_mean, area_weights)

    global_means_df = pd.DataFrame(
        {
            "component": ["P_mm_day", "ET_mm_day", "Q_mm_day", "Residual_mm_day"],
            "global_mean": [gm_P, gm_ET, gm_Q, gm_RES],
        }
    )

    # 8) Save fields to NetCDF
    try:
        out_ds = xr.Dataset(
            {
                "P_mm_day": P_mean,
                "ET_mm_day": ET_mean,
                "Q_mm_day": Q_mean,
                "Residual_mm_day": RES_mean,
            }
        )
        out_ds["lon"] = lon2d
        out_ds["lat"] = lat2d
        out_nc_path = os.path.join(output_dir, f"{case_name}_ELM_water_balance_1985-1989_timemean.nc")
        out_ds.to_netcdf(out_nc_path)
        print(f"Saved fields to {out_nc_path}")
    except Exception as e:
        print(f"Failed to save NetCDF: {e}")

    # 9) Load basin polygons and create masks
    try:
        with open(basin_geojson_path, "r") as f:
            geojson = json.load(f)
    except Exception as e:
        print(f"Failed to read basin GeoJSON: {e}")
        return

    # Build mapping from grdc_no to list of polygon rings (each polygon can contain outer and holes)
    # We will store as list of polygons; each polygon is dict with 'outer': list[(lon,lat)], 'holes': list of lists[(lon,lat)]
    def extract_polygons(feature):
        geom = feature.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        polygons = []
        if gtype == "Polygon":
            # coords: list of linear rings: [ [ [x,y], ... ], [hole1...], ...]
            if len(coords) > 0:
                outer = coords[0]
                holes = coords[1:] if len(coords) > 1 else []
                polygons.append({"outer": outer, "holes": holes})
        elif gtype == "MultiPolygon":
            for poly in coords:
                if len(poly) > 0:
                    outer = poly[0]
                    holes = poly[1:] if len(poly) > 1 else []
                    polygons.append({"outer": outer, "holes": holes})
        return polygons

    # Create grid points for point-in-polygon tests
    points = np.column_stack([lon2d_wrapped_np.ravel(), lat2d_np.ravel()])

    # Prepare masks dict
    basin_masks = {}
    basin_polylines = {}  # for plotting outlines
    target_ids = set([int(v) for v in basins.values()])

    features = geojson.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        grdc_no = props.get("grdc_no")
        try:
            grdc_no_int = int(grdc_no)
        except Exception:
            continue
        if grdc_no_int not in target_ids:
            continue
        polys = extract_polygons(feat)
        # Build mask
        mask = np.zeros(points.shape[0], dtype=bool)
        outlines = []  # collect outer boundaries for plotting
        for poly in polys:
            outer = np.array(poly["outer"])
            # Wrap longitudes to [-180, 180]
            outer_wrapped = outer.copy()
            outer_wrapped[:, 0] = ((outer_wrapped[:, 0] + 180) % 360) - 180
            path_outer = MplPath(outer_wrapped)
            inside = path_outer.contains_points(points)
            # Subtract holes
            for hole in poly["holes"]:
                hole_arr = np.array(hole)
                hole_arr[:, 0] = ((hole_arr[:, 0] + 180) % 360) - 180
                path_hole = MplPath(hole_arr)
                inside_hole = path_hole.contains_points(points)
                inside = np.logical_and(inside, np.logical_not(inside_hole))
            # Accumulate across polygons
            mask = np.logical_or(mask, inside)
            # Save outline for plotting (just outer ring)
            outlines.append(outer_wrapped)
        basin_masks[grdc_no_int] = mask.reshape(lat2d_np.shape)
        basin_polylines[grdc_no_int] = outlines

    # 10) Compute per-basin area-weighted means for P, ET, Q, RES
    basin_results = []
    for bname, gid in basins.items():
        mask = basin_masks.get(gid, None)
        if mask is None:
            print(f"Warning: No mask found for basin {bname} ({gid})")
            bm_P = np.nan
            bm_ET = np.nan
            bm_Q = np.nan
            bm_RES = np.nan
        else:
            mask_da = xr.DataArray(mask, coords=P_mean.coords, dims=P_mean.dims)
            # Multiply by area weights
            area_in = area_weights.where(mask_da)
            def awm(field):
                num = (field.where(mask_da) * area_in).sum(skipna=True)
                den = area_in.sum(skipna=True)
                if float(den.values) == 0:
                    return np.nan
                return float((num / den).values)
            bm_P = awm(P_mean)
            bm_ET = awm(ET_mean)
            bm_Q = awm(Q_mean)
            bm_RES = awm(RES_mean)
        basin_results.append(
            {"basin": bname, "grdc_no": gid, "P_mm_day": bm_P, "ET_mm_day": bm_ET, "Q_mm_day": bm_Q, "Residual_mm_day": bm_RES}
        )
    basin_df = pd.DataFrame(basin_results)
    basin_df = basin_df.set_index("basin")

    # 11) Save CSVs for global means and basin means
    try:
        global_csv = os.path.join(output_dir, f"{case_name}_ELM_global_means_1985-1989.csv")
        global_means_df.to_csv(global_csv, index=False)
        print(f"Saved global means to {global_csv}")
    except Exception as e:
        print(f"Failed to save global means CSV: {e}")

    try:
        basin_csv = os.path.join(output_dir, f"{case_name}_ELM_basin_means_1985-1989.csv")
        basin_df.to_csv(basin_csv)
        print(f"Saved basin means to {basin_csv}")
    except Exception as e:
        print(f"Failed to save basin means CSV: {e}")

    # 12) Plot composite figure
    try:
        fig = plt.figure(figsize=(14, 10))
        # Top panel: global residual map with basin outlines
        ax_map = plt.subplot2grid((3, 3), (0, 0), colspan=3, projection=ccrs.Robinson())
        ax_map.set_global()
        ax_map.coastlines(linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.2)
        # plot residual using scatter for generality
        res_vals = RES_mean.values.ravel()
        sc = ax_map.scatter(points[:, 0], points[:, 1], c=res_vals, s=1, cmap="RdBu_r", vmin=-2.0, vmax=2.0, transform=ccrs.PlateCarree())
        cb = plt.colorbar(sc, ax=ax_map, orientation="horizontal", pad=0.05, fraction=0.05)
        cb.set_label("Residual (P - ET - Q) [mm/day]")
        ax_map.set_title(f"ELM Water Balance Residual (1985-1989 mean) with Basin Outlines")

        # draw basin outlines
        for bname, gid in basins.items():
            outlines = basin_polylines.get(gid, [])
            for ring in outlines:
                ax_map.plot(ring[:, 0], ring[:, 1], color="black", linewidth=1.0, transform=ccrs.PlateCarree())

        # Bottom panels: per-basin bar charts
        components = ["P_mm_day", "ET_mm_day", "Q_mm_day", "Residual_mm_day"]
        colors = {"P_mm_day": "#1f77b4", "ET_mm_day": "#2ca02c", "Q_mm_day": "#ff7f0e", "Residual_mm_day": "#d62728"}

        # Create 2 rows x 3 columns for 6 basins
        basin_names_order = list(basins.keys())
        for i, bname in enumerate(basin_names_order):
            row = 1 + i // 3
            col = i % 3
            ax = plt.subplot2grid((3, 3), (row, col))
            vals = [basin_df.loc[bname, comp] for comp in components]
            ax.bar(components, vals, color=[colors[c] for c in components])
            ax.set_title(f"{bname} (GRDC {basins[bname]})")
            ax.set_ylabel("mm/day")
            ax.set_xticklabels(["P", "ET", "Q", "Res"], rotation=0)
            ax.axhline(0, color="k", linewidth=0.5)
            # Add grid
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        figpath = os.path.join(output_dir, f"{case_name}_ELM_water_balance_composite_1985-1989.png")
        plt.savefig(figpath, dpi=200)
        plt.close(fig)
        print(f"Saved composite figure to {figpath}")
    except Exception as e:
        print(f"Failed to create composite figure: {e}")

    # 13) Also save individual field maps for P, ET, Q if desired (optional)
    # Skipping to keep the script concise.

    print("Analysis complete.")

if __name__ == "__main__":
    main()