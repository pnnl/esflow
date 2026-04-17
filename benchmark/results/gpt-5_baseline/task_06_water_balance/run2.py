import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")

def lon_to_180(lon):
    return ((lon + 180) % 360) - 180

def build_file_list(base_dir, case_name, start_year=1985, end_year=1989):
    files = []
    for yr in range(start_year, end_year + 1):
        pattern = os.path.join(base_dir, f"{case_name}.elm.h0.{yr}-*.nc")
        files.extend(sorted(glob.glob(pattern)))
    return files

def open_elm_dataset(file_list, vars_needed):
    try:
        ds = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            parallel=False,
            decode_times=True,
            use_cftime=True
        )
        # Ensure required variables exist
        missing = [v for v in vars_needed if v not in ds.variables]
        if missing:
            raise KeyError(f"Missing variables in dataset: {missing}")
        return ds
    except Exception as e:
        print(f"Error opening ELM dataset: {e}")
        return None

def compute_climatology(ds):
    try:
        # Compute water balance components (in mm/s)
        # RAIN + SNOW are kg/m2/s, equal to mm/s for water density
        P = ds["RAIN"] + ds["SNOW"]
        ET = ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]
        Q = ds["QRUNOFF"]

        # Time slice for 1985-01-01 to 1989-12-31 if time coord exists
        if "time" in P.dims:
            P = P.sel(time=slice("1985-01-01", "1989-12-31"))
            ET = ET.sel(time=slice("1985-01-01", "1989-12-31"))
            Q = Q.sel(time=slice("1985-01-01", "1989-12-31"))

            P_mean = P.mean(dim="time", skipna=True)
            ET_mean = ET.mean(dim="time", skipna=True)
            Q_mean = Q.mean(dim="time", skipna=True)
        else:
            P_mean = P
            ET_mean = ET
            Q_mean = Q

        # Convert to mm/yr
        seconds_per_year = 365.0 * 24.0 * 3600.0
        P_mm_yr = P_mean * seconds_per_year
        ET_mm_yr = ET_mean * seconds_per_year
        Q_mm_yr = Q_mean * seconds_per_year

        residual_mm_yr = P_mm_yr - ET_mm_yr - Q_mm_yr

        # Build dataset
        ds_out = xr.Dataset(
            {
                "P_mm_yr": P_mm_yr.where(np.isfinite(P_mm_yr)),
                "ET_mm_yr": ET_mm_yr.where(np.isfinite(ET_mm_yr)),
                "Q_mm_yr": Q_mm_yr.where(np.isfinite(Q_mm_yr)),
                "Residual_mm_yr": residual_mm_yr.where(np.isfinite(residual_mm_yr)),
            }
        )
        ds_out["P_mm_yr"].attrs["long_name"] = "Total Precipitation"
        ds_out["P_mm_yr"].attrs["units"] = "mm/yr"
        ds_out["ET_mm_yr"].attrs["long_name"] = "Evapotranspiration"
        ds_out["ET_mm_yr"].attrs["units"] = "mm/yr"
        ds_out["Q_mm_yr"].attrs["long_name"] = "Total Runoff"
        ds_out["Q_mm_yr"].attrs["units"] = "mm/yr"
        ds_out["Residual_mm_yr"].attrs["long_name"] = "Water Balance Residual (P - ET - Q)"
        ds_out["Residual_mm_yr"].attrs["units"] = "mm/yr"
        return ds_out
    except Exception as e:
        print(f"Error computing climatology: {e}")
        return None

def area_weights(lat, lon):
    # Compute cosine(lat) weights
    lat_rad = np.deg2rad(lat)
    w = np.cos(lat_rad)
    if lat.ndim == 1 and lon.ndim == 1:
        W = np.outer(w, np.ones_like(lon))
    else:
        # Broadcast to shape
        W = np.cos(np.deg2rad(lat))
    return xr.DataArray(W, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

def weighted_mean(field, weights):
    # Align and mask
    try:
        aligned = field.broadcast_like(weights)
        w = weights.where(np.isfinite(aligned))
        num = (aligned * w).sum(dim=("lat", "lon"), skipna=True)
        den = w.sum(dim=("lat", "lon"), skipna=True)
        return (num / den).item()
    except Exception as e:
        print(f"Error computing weighted mean: {e}")
        return np.nan

def make_polygon_mask(lat2d, lon2d, polygon_geom):
    # polygon_geom: dict with 'type' and 'coordinates' following GeoJSON
    # Returns boolean mask with True inside polygon
    shape = lat2d.shape
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    def ring_contains_points(ring_coords):
        arr = np.array(ring_coords)
        # Ensure closed ring
        if not (arr[0, 0] == arr[-1, 0] and arr[0, 1] == arr[-1, 1]):
            arr = np.vstack([arr, arr[0]])
        path = mpath.Path(arr)
        return path.contains_points(points).reshape(shape)

    mask_total = np.zeros(shape, dtype=bool)

    if polygon_geom["type"] == "Polygon":
        rings = polygon_geom["coordinates"]
        exterior = rings[0]
        mask = ring_contains_points(exterior)
        # Subtract holes
        if len(rings) > 1:
            for hole in rings[1:]:
                mask_h = ring_contains_points(hole)
                mask = np.logical_and(mask, np.logical_not(mask_h))
        mask_total = np.logical_or(mask_total, mask)
    elif polygon_geom["type"] == "MultiPolygon":
        for poly in polygon_geom["coordinates"]:
            rings = poly
            exterior = rings[0]
            mask = ring_contains_points(exterior)
            if len(rings) > 1:
                for hole in rings[1:]:
                    mask_h = ring_contains_points(hole)
                    mask = np.logical_and(mask, np.logical_not(mask_h))
            mask_total = np.logical_or(mask_total, mask)
    else:
        pass
    return mask_total

def generate_basin_masks(lat, lon, geojson_path, basin_mapping):
    masks = {}
    try:
        with open(geojson_path, 'r') as f:
            gj = json.load(f)
    except Exception as e:
        print(f"Error reading basin polygons: {e}")
        return masks

    # Build lookup from grdc_no to geometry
    features = gj.get("features", [])
    geom_by_grdc = {}
    for feat in features:
        props = feat.get("properties", {})
        grdc_no = None
        for key in ["grdc_no", "GRDC_NO", "grdc", "GRDC"]:
            if key in props:
                grdc_no = props[key]
                break
        if grdc_no is None:
            continue
        geom = feat.get("geometry", None)
        if geom:
            geom_by_grdc[str(grdc_no)] = geom
            geom_by_grdc[int(grdc_no)] = geom
    # Prepare grid
    if lat.ndim == 1 and lon.ndim == 1:
        lon1d = lon.values
        lat1d = lat.values
        lon1d_180 = lon_to_180(lon1d)
        Lon2, Lat2 = np.meshgrid(lon1d_180, lat1d)
    else:
        Lon2 = lon_to_180(lon.values)
        Lat2 = lat.values

    for code, name in basin_mapping.items():
        geom = geom_by_grdc.get(code, None)
        if geom is None:
            print(f"Warning: Basin geometry for code {code} not found in GeoJSON.")
            continue
        # Wrap polygon longitudes to -180..180
        def wrap_coords(coords):
            return [[lon_to_180(np.array([pt[0]])).item(), pt[1]] for pt in coords]
        # Build normalized geometry
        geom_norm = {"type": geom["type"], "coordinates": None}
        if geom["type"] == "Polygon":
            coords = []
            for ring in geom["coordinates"]:
                coords.append(wrap_coords(ring))
            geom_norm["coordinates"] = coords
        elif geom["type"] == "MultiPolygon":
            coords = []
            for poly in geom["coordinates"]:
                rings = []
                for ring in poly:
                    rings.append(wrap_coords(ring))
                coords.append(rings)
            geom_norm["coordinates"] = coords
        else:
            print(f"Unsupported geometry type {geom['type']} for basin {name}")
            continue
        mask = make_polygon_mask(Lat2, Lon2, geom_norm)
        masks[name] = mask
    return masks

def area_weighted_mean_over_mask(field, weights, mask):
    try:
        da = field.copy()
        # Apply mask
        mask_da = xr.DataArray(mask, coords={"lat": da["lat"], "lon": da["lon"]}, dims=("lat", "lon"))
        da_masked = da.where(mask_da)
        w_masked = weights.where(mask_da)
        num = (da_masked * w_masked).sum(dim=("lat", "lon"), skipna=True)
        den = w_masked.sum(dim=("lat", "lon"), skipna=True)
        return (num / den).item()
    except Exception as e:
        print(f"Error computing area-weighted mean over mask: {e}")
        return np.nan

def reorder_longitude(ds_in):
    # Reorder dataset along lon so that longitudes are increasing from -180 to 180
    ds = ds_in.copy()
    try:
        lon = ds["lon"]
        if lon.ndim == 1:
            lon180 = lon_to_180(lon.values)
            sort_idx = np.argsort(lon180)
            lon_sorted = lon180[sort_idx]
            ds = ds.assign_coords(lon=lon_sorted)
            for var in ds.data_vars:
                if "lon" in ds[var].dims and ds[var].ndim >= 2:
                    ds[var] = ds[var].isel(lon=sort_idx)
        return ds
    except Exception as e:
        print(f"Warning: Could not reorder longitudes: {e}")
        return ds_in

def main():
    # Paths and settings
    data_base = "./data/sample/e3sm"
    elm_dir = os.path.join(data_base, "lnd")
    case_name = "sample.v3.LR.historical"
    basin_geojson = "./data/sample/obs/basin_polygons.geojson"
    # Output directory as specified by user
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_06_water_balance/run2_output"
    ensure_dir(output_dir)

    # Build file list and open dataset
    files = build_file_list(elm_dir, case_name, 1985, 1989)
    if len(files) == 0:
        print("No ELM files found for 1985-1989. Exiting.")
        return

    vars_needed = ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]
    ds = open_elm_dataset(files, vars_needed)
    if ds is None:
        return

    # Compute climatology fields (mm/yr)
    ds_clim = compute_climatology(ds)
    if ds_clim is None:
        return

    # Ensure coordinates are present
    if "lat" not in ds_clim.coords or "lon" not in ds_clim.coords:
        print("Dataset missing lat/lon coordinates. Exiting.")
        return

    # Reorder longitudes to [-180, 180] for plotting and masks
    ds_clim = reorder_longitude(ds_clim)

    # Save climatology dataset to NetCDF
    nc_out = os.path.join(output_dir, "ELM_WB_climatology_1985_1989_mm_per_yr.nc")
    try:
        ds_clim.to_netcdf(nc_out)
        print(f"Saved climatological fields to {nc_out}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

    # Compute global area-weighted means (over land grid cells)
    lat = ds_clim["lat"]
    lon = ds_clim["lon"]
    W = area_weights(lat, lon)

    P_global = weighted_mean(ds_clim["P_mm_yr"], W)
    ET_global = weighted_mean(ds_clim["ET_mm_yr"], W)
    Q_global = weighted_mean(ds_clim["Q_mm_yr"], W)
    Res_global = weighted_mean(ds_clim["Residual_mm_yr"], W)

    # Basin mapping: GRDC code to basin name
    basin_mapping = {
        3629000: "Amazon",
        4121801: "Missouri",
        4115200: "Columbia",
        6742900: "Danube",
        2969100: "Mekong",
        1159100: "Orange",
    }

    # Generate masks for basins
    basin_masks = generate_basin_masks(lat, lon, basin_geojson, basin_mapping)

    # Compute per-basin means
    basin_results = {}
    for code, name in basin_mapping.items():
        if name not in basin_masks:
            continue
        mask = basin_masks[name]
        basin_results[name] = {
            "P_mm_yr": area_weighted_mean_over_mask(ds_clim["P_mm_yr"], W, mask),
            "ET_mm_yr": area_weighted_mean_over_mask(ds_clim["ET_mm_yr"], W, mask),
            "Q_mm_yr": area_weighted_mean_over_mask(ds_clim["Q_mm_yr"], W, mask),
            "Residual_mm_yr": area_weighted_mean_over_mask(ds_clim["Residual_mm_yr"], W, mask),
        }

    # Save global and basin means to CSV
    try:
        rows = []
        # Global
        rows.append({
            "Region": "Global (land-weighted)",
            "P_mm_yr": P_global,
            "ET_mm_yr": ET_global,
            "Q_mm_yr": Q_global,
            "Residual_mm_yr": Res_global
        })
        # Basins
        for name in basin_mapping.values():
            vals = basin_results.get(name, None)
            if vals is None:
                continue
            row = {"Region": name}
            row.update(vals)
            rows.append(row)
        df_out = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "ELM_WB_means_1985_1989.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"Saved global and basin means to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV of means: {e}")

    # Plot composite figure
    try:
        # Prepare data for residual map
        lat_vals = lat.values
        lon_vals = lon_to_180(lon.values if lon.ndim == 1 else lon.values)
        residual = ds_clim["Residual_mm_yr"].values

        # Determine colormap limits
        finite_vals = residual[np.isfinite(residual)]
        if finite_vals.size > 0:
            vabs = np.nanpercentile(np.abs(finite_vals), 99)
            if not np.isfinite(vabs) or vabs == 0:
                vabs = np.nanmax(np.abs(finite_vals)) if finite_vals.size else 1.0
        else:
            vabs = 1.0
        vmin, vmax = -vabs, vabs

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.3)

        # Top panel: Residual map
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.coastlines(linewidth=0.8)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax_map.gridlines(draw_labels=False, linewidth=0.2, alpha=0.3)

        # Plot residual using pcolormesh with 1D lon/lat if possible
        if lat.ndim == 1 and lon.ndim == 1:
            Lon2, Lat2 = np.meshgrid(lon_vals, lat_vals)
            pcm = ax_map.pcolormesh(Lon2, Lat2, residual, transform=ccrs.PlateCarree(),
                                    cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        else:
            pcm = ax_map.pcolormesh(lon_vals, lat_vals, residual, transform=ccrs.PlateCarree(),
                                    cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")

        cbar = fig.colorbar(pcm, ax=ax_map, orientation="horizontal", pad=0.05, fraction=0.05)
        cbar.set_label("Residual (P - ET - Q) [mm/yr]")
        ax_map.set_title("ELM Water Balance Residual (P - ET - Q), 1985-1989")

        # Overlay basin outlines
        try:
            with open(basin_geojson, 'r') as f:
                gj = json.load(f)
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                geom = feat.get("geometry", None)
                if geom is None:
                    continue
                # Plot only requested basins
                include = False
                for key in ["grdc_no", "GRDC_NO", "grdc", "GRDC"]:
                    if key in props and props[key] in basin_mapping:
                        include = True
                    if key in props and str(props[key]) in [str(k) for k in basin_mapping.keys()]:
                        include = True
                if not include:
                    continue
                if geom["type"] == "Polygon":
                    for ring in geom["coordinates"]:
                        ring_arr = np.array(ring)
                        ax_map.plot(lon_to_180(ring_arr[:, 0]), ring_arr[:, 1], "-k", linewidth=1.0, transform=ccrs.PlateCarree())
                elif geom["type"] == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        for ring in poly:
                            ring_arr = np.array(ring)
                            ax_map.plot(lon_to_180(ring_arr[:, 0]), ring_arr[:, 1], "-k", linewidth=1.0, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"Error overlaying basin outlines: {e}")

        # Bottom panel: Per-basin bar charts
        ax_bar = fig.add_subplot(gs[1, 0])
        basin_names = [basin_mapping[k] for k in basin_mapping.keys() if basin_mapping[k] in basin_results]
        n = len(basin_names)
        if n == 0:
            print("No basin masks available for bar chart.")
        else:
            P_vals = [basin_results[name]["P_mm_yr"] for name in basin_names]
            ET_vals = [basin_results[name]["ET_mm_yr"] for name in basin_names]
            Q_vals = [basin_results[name]["Q_mm_yr"] for name in basin_names]
            Res_vals = [basin_results[name]["Residual_mm_yr"] for name in basin_names]

            x = np.arange(n)
            width = 0.2
            ax_bar.bar(x - 1.5*width, P_vals, width=width, label="P", color="#1f77b4")
            ax_bar.bar(x - 0.5*width, ET_vals, width=width, label="ET", color="#ff7f0e")
            ax_bar.bar(x + 0.5*width, Q_vals, width=width, label="Q", color="#2ca02c")
            ax_bar.bar(x + 1.5*width, Res_vals, width=width, label="Residual", color="#d62728")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(basin_names, rotation=0)
            ax_bar.set_ylabel("mm/yr")
            ax_bar.set_title("Basin-Mean P, ET, Q, and Residual (1985-1989)")
            ax_bar.legend(ncol=4, loc="upper right")

            # Annotate global means
            txt = f"Global land mean: P={P_global:.1f}, ET={ET_global:.1f}, Q={Q_global:.1f}, Res={Res_global:.1f} mm/yr"
            ax_bar.text(0.5, -0.25, txt, transform=ax_bar.transAxes, ha="center", va="top")

        fig_path = os.path.join(output_dir, "ELM_WB_residual_global_and_basin_bars_1985_1989.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved composite figure to {fig_path}")
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()