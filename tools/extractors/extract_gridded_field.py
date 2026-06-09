from pathlib import Path

import xarray as xr

import tempfile

from tools.core.e3sm import find_e3sm_files, open_e3sm_dataset

def extract_gridded_field(
    variable: str, 
    data_dir: Path = None, 
    case_name: str = '', 
    component: str = '', 
    obs_file: Path = None,
    years: list[int] = None) -> Path:
    """
    Extract a 2D gridded field from E3SM or observation NetCDF files.

    Computes the time-mean over the specified years and outputs a simple
    lat-lon NetCDF file suitable for spatial comparison and plotting.
    For composite variables, provide multiple names separated by "+" to sum them
    (e.g., "QVEGE+QVEGT+QSOIL" to sum three ET components).

    Args:
        variable (str): Variable name(s) to extract. Use "+" to sum multiple variables
            (e.g., "RAIN+SNOW" or "QVEGE+QVEGT+QSOIL").
        data_dir (Path, optional): Base directory for E3SM output (use for model data).
        case_name (str, optional): E3SM case name (use for model data).
        component (str, optional): Model component: elm, mosart, or eam (use for model data).
        obs_file (Path, optional): Path to observation NetCDF file (use for obs data).
        years (list[int], optional): Years to include (e.g., [2001]). If omitted, uses
            all available time steps.

    Returns:
        Path: Time-averaged 2D field as a lat-lon NetCDF file.

    Raises:
        ValueError: If neither E3SM data (data_dir + case_name + component) nor obs_file
            is provided, or if a requested variable is not found in the dataset.
    """

    var_names = [v.strip() for v in variable.split('+')]

    # Open dataset — either E3SM model or observation file
    if obs_file:
        print(f"  Opening observation file: {obs_file}")
        ds = xr.open_dataset(obs_file)
        source = 'obs'
    elif data_dir and case_name:
        print(f"  Opening E3SM {component} data: {case_name}")
        years_list = years if years else []
        files = find_e3sm_files(data_dir, case_name, component, years_list)
        print(f"  Files found: {len(files)}")
        ds = open_e3sm_dataset(files, variables=var_names)
        source = 'model'
    else:
        raise ValueError(
            "Provide either (data_dir + case_name + component) for model data "
            "or obs_file for observation data."
        )

    # Extract and sum variables
    print(f"  Variables: {var_names}")
    field = None
    for vname in var_names:
        if vname not in ds:
            available = [v for v in ds.data_vars if not v.startswith('_')]
            raise ValueError(
                f"Variable '{vname}' not found. Available: {sorted(available)[:20]}"
            )
        v = ds[vname].astype('float64')
        if hasattr(v, 'compute'):
            v = v.compute()
        field = v if field is None else field + v

    # Compute time mean and record time range
    time_range = ''
    if 'time' in field.dims:
        print(f"  Time steps: {len(field.time)}")
        try:
            tvals = field.time.values
            # Works for both datetime64 and cftime objects
            y0 = int(tvals[0].year) if hasattr(tvals[0], 'year') else None
            y1 = int(tvals[-1].year) if hasattr(tvals[-1], 'year') else None
            if y0 is not None and y1 is not None:
                time_range = f"{y0}-{y1}" if y0 != y1 else str(y0)
        except Exception:
            pass
        field_mean = field.mean(dim='time')
    else:
        field_mean = field

    # Build clean output dataset
    out_name = variable.replace('+', '_plus_')
    out_ds = xr.Dataset({out_name: field_mean})

    # Ensure lat/lon coordinates are preserved
    for coord in ['lat', 'lon', 'landfrac', 'area']:
        if coord in ds.coords or coord in ds.data_vars:
            if coord not in out_ds:
                out_ds[coord] = ds[coord]

    out_ds.attrs['source'] = source
    out_ds.attrs['variables_summed'] = variable
    if years:
        out_ds.attrs['years'] = str(years)
    if time_range:
        out_ds.attrs['time_range'] = time_range

    # Save
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        out_path = fp.name
        out_ds.to_netcdf(out_path)

    print(f"  Output shape: {dict(field_mean.sizes)}")
    print(f"  Saved: {out_path}")

    ds.close()

    return out_path
