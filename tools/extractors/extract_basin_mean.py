"""
Extract area-weighted basin means from a gridded field using GeoJSON polygons.

For each basin polygon, identifies grid cells whose centres fall inside the
polygon, computes the area-weighted mean of the field, and returns the results
as a CSV.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param


def _point_in_polygon(px, py, polygon_coords):
    """Ray-casting test for a point inside a polygon (list of [lon, lat])."""
    n = len(polygon_coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i]
        xj, yj = polygon_coords[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _make_basin_mask(lat, lon, basin_feature):
    """Create a 2-D boolean mask for grid cells inside a basin polygon."""
    coords = basin_feature['geometry']['coordinates']
    geom_type = basin_feature['geometry']['type']

    # Collect all polygon rings
    rings = []
    if geom_type == 'Polygon':
        rings.append(coords[0])  # exterior ring
    elif geom_type == 'MultiPolygon':
        for poly in coords:
            rings.append(poly[0])  # exterior ring of each polygon

    mask = np.zeros((len(lat), len(lon)), dtype=bool)

    # Quick bounding-box pre-filter per ring
    for ring in rings:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        lon_min, lon_max = min(xs), max(xs)
        lat_min, lat_max = min(ys), max(ys)

        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]

        for li in lat_idx:
            for lj in lon_idx:
                if not mask[li, lj]:
                    if _point_in_polygon(float(lon[lj]), float(lat[li]), ring):
                        mask[li, lj] = True

    return mask


SPEC = ToolSpec(
    name='extract_basin_mean',
    description='Extract area-weighted basin means from a gridded NetCDF field '
                'using GeoJSON basin polygons. Returns per-basin mean values.',
    inputs={
        'field_file': Param('path', required=True,
                            description='Gridded NetCDF field (lat × lon)'),
        'basins_file': Param('path', required=True,
                             description='GeoJSON file with basin polygons (property: grdc_no)'),
        'gauge_ids': Param('str', required=True,
                           description='Comma-separated GRDC gauge IDs to extract'),
    },
    outputs={
        'basin_means_file': {'type': 'csv',
                             'description': 'CSV with gauge_id and basin_mean columns'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    field_file = config['field_file']
    basins_file = config['basins_file']
    gauge_ids_str = config['gauge_ids']
    output_dir = Path(config['output_dir'])

    gauge_ids = [g.strip() for g in gauge_ids_str.split(',')]

    # Load gridded field
    ds = xr.open_dataset(field_file)
    # Auto-select variable (skip coord vars)
    skip = {'lat', 'lon', 'latitude', 'longitude', 'time', 'time_bnds'}
    data_vars = [v for v in ds.data_vars if v.lower() not in skip]
    if len(data_vars) == 0:
        raise ValueError(f"No data variables found in {field_file}")
    var_name = data_vars[0]
    field = ds[var_name]

    # Handle time dimension
    if 'time' in field.dims:
        field = field.mean(dim='time')

    values = field.values.copy()
    lat = field.lat.values if 'lat' in field.coords else field.latitude.values
    lon = field.lon.values if 'lon' in field.coords else field.longitude.values

    # Mask out fill values (some obs datasets lack _FillValue metadata)
    fill_mask = np.abs(values) > 1e20
    if fill_mask.any():
        n_fill = fill_mask.sum()
        print(f"  Masked {n_fill} fill values (>{1e20})")
        values[fill_mask] = np.nan

    # Convert 0-360 longitudes to -180..180 for consistency with GeoJSON
    if lon.max() > 180:
        shift_idx = lon >= 180
        lon = lon.copy()
        lon[shift_idx] -= 360
        # Re-sort so longitudes are monotonically increasing
        sort_order = np.argsort(lon)
        lon = lon[sort_order]
        values = values[:, sort_order]
        print(f"  Shifted lon to -180..180 range")

    print(f"  Field: {var_name}, shape: {values.shape}")

    # Load basins
    with open(basins_file) as f:
        basins = json.load(f)

    # Area weights (cosine of latitude)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    weights = np.broadcast_to(cos_lat[:, np.newaxis], values.shape)

    results = []
    for gid in gauge_ids:
        grdc_no = int(gid)
        basin_feat = None
        for feat in basins['features']:
            if feat['properties']['grdc_no'] == grdc_no:
                basin_feat = feat
                break

        if basin_feat is None:
            print(f"  Warning: No basin polygon for {gid}")
            results.append({'gauge_id': gid, 'basin_mean': np.nan})
            continue

        mask = _make_basin_mask(lat, lon, basin_feat)
        n_cells = mask.sum()

        if n_cells == 0:
            print(f"  Warning: No grid cells in basin {gid}")
            results.append({'gauge_id': gid, 'basin_mean': np.nan})
            continue

        masked_vals = np.where(mask, values, np.nan)
        masked_weights = np.where(mask, weights, 0)

        # Area-weighted mean, ignoring NaN
        valid = ~np.isnan(masked_vals) & mask
        if valid.sum() == 0:
            basin_mean = np.nan
        else:
            basin_mean = float(np.nansum(masked_vals * masked_weights) / np.nansum(
                np.where(valid, masked_weights, 0)))

        results.append({'gauge_id': gid, 'basin_mean': basin_mean})
        print(f"  Basin {gid}: {n_cells} cells, mean = {basin_mean:.6f}")

    ds.close()

    df = pd.DataFrame(results)
    out_path = output_dir / 'basin_means.csv'
    df.to_csv(out_path, index=False)

    return {
        'basin_means_file': str(out_path),
    }
