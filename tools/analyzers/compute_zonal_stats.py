from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

import tempfile

def compute_zonal_stats(field_file: Path, variable: str = "", lat_bands: str = "") -> Path:
    """
    Compute area-weighted spatial statistics from a gridded field.

    Supports two modes: (1) global mean — returns a single value, or
    (2) latitude bands — splits the domain into bands and returns one value per band.
    Uses landfrac weighting when available. Outputs a CSV with one row per region.

    Args:
        field_file (Path): Input NetCDF file with a 2D field (from extract_gridded_field).
        variable (str, optional): Variable name in the NetCDF file. If omitted, uses the
            first data variable. Defaults to "".
        lat_bands (str, optional): Latitude band definitions as comma-separated 
            "name:south:north" (e.g., "boreal:55:90,temperate:30:55,tropical:0:30").
            Leave empty for global mean only. Defaults to "".

    Returns:
        Path: Zonal stats CSV file with columns: region, mean, area_weighted_mean.

    Raises:
        ValueError: If the variable is not found in the NetCDF file or if band
            definitions have invalid format.
    """
    ds = xr.open_dataset(field_file)

    # Determine variable name
    skip_vars = {'landfrac', 'area', 'landmask', 'pftmask'}
    if variable:
        vname = variable
    else:
        data_vars = [v for v in ds.data_vars if v not in skip_vars]
        if not data_vars:
            raise ValueError(f"No data variables found in {field_file}")
        vname = data_vars[0]
        print(f"  Auto-selected variable: {vname}")

    if vname not in ds:
        raise ValueError(f"Variable '{vname}' not found in {field_file}. "
                         f"Available: {list(ds.data_vars)}")

    # Get data as numpy arrays
    field = ds[vname].values.astype(float)
    if field.ndim == 3:  # (time, lat, lon) — take mean over time
        field = np.nanmean(field, axis=0)

    lat_1d = ds['lat'].values
    nlon = field.shape[1] if field.ndim == 2 else 1

    # Build 2D lat array and cos(lat) weights
    lat_2d = np.broadcast_to(lat_1d[:, np.newaxis], field.shape)
    cos_weights = np.cos(np.radians(lat_2d))

    # Apply landfrac if available
    if 'landfrac' in ds:
        lf = ds['landfrac'].values.astype(float)
        if lf.ndim == 3:
            lf = lf[0]  # take first time step
        cos_weights = cos_weights * lf

    # Parse latitude bands
    if lat_bands:
        bands = []
        for band_def in lat_bands.split(','):
            parts = band_def.strip().split(':')
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid band definition '{band_def}'. "
                    f"Expected 'name:south:north' (e.g., 'tropical:-30:30')"
                )
            bands.append({
                'name': parts[0].strip(),
                'south': float(parts[1]),
                'north': float(parts[2]),
            })
    else:
        bands = [{'name': 'global', 'south': -90, 'north': 90}]

    print(f"  Variable: {vname}, shape: {field.shape}")
    print(f"  Regions: {[b['name'] for b in bands]}")

    # Compute stats for each band
    results = []
    for band in bands:
        # Create lat band mask (2D)
        band_mask = (lat_2d >= band['south']) & (lat_2d <= band['north'])
        valid = np.isfinite(field) & band_mask & (cos_weights > 0)

        if not valid.any():
            results.append({
                'region': band['name'],
                'mean': np.nan,
                'area_weighted_mean': np.nan,
            })
            print(f"  {band['name']}: no valid data")
            continue

        simple_mean = float(np.nanmean(field[valid]))
        weighted_mean = float(np.average(field[valid], weights=cos_weights[valid]))

        results.append({
            'region': band['name'],
            'mean': simple_mean,
            'area_weighted_mean': weighted_mean,
        })

        print(f"  {band['name']}: mean={simple_mean:.6e}, weighted={weighted_mean:.6e}")

    df = pd.DataFrame(results)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        out_path = fp.name
        df.to_csv(out_path, index=False)

    ds.close()

    return out_path
