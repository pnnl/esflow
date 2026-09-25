"""Per-gridcell linear trend of a NetCDF field.

Plain user science code -- no ESMFlow imports. Onboarded as a *diagnostics*
capability by the onboarding agent (returns an xarray Dataset -> NetCDF).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _pick_variable(dataset: xr.Dataset, variable: str | None) -> xr.DataArray:
    if variable:
        if variable not in dataset:
            raise ValueError(
                f"Variable '{variable}' not in dataset; available: {list(dataset.data_vars)}"
            )
        return dataset[variable]
    candidates = [
        name for name, da in dataset.data_vars.items() if "time" in da.dims and da.ndim >= 3
    ]
    if not candidates:
        raise ValueError("No time-varying gridded variable found in dataset")
    return dataset[candidates[0]]


def compute_gridded_trend(
    input_file: str,
    variable: str | None = None,
    per_decade: bool = True,
    min_valid_fraction: float = 0.6,
) -> xr.Dataset:
    """Fit a per-gridcell least-squares linear trend to a gridded field.

    Returns a dataset holding the trend slope, the intercept, the fraction of
    valid timesteps used, and the time-mean field. Cells with insufficient valid
    data are masked to NaN.

    Args:
        input_file: NetCDF file containing a time-varying gridded variable.
        variable: Variable to analyze. Defaults to the first 3-D time variable.
        per_decade: Report the slope per decade instead of per year.
        min_valid_fraction: Minimum fraction of finite timesteps required per cell.

    Returns:
        An xarray Dataset with ``trend``, ``intercept``, ``valid_fraction`` and
        ``time_mean`` variables.
    """

    with xr.open_dataset(input_file) as raw:
        dataset = raw.load()

    field = _pick_variable(dataset, variable)

    time = field["time"]
    time_values = time.values
    if np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
        years = (
            (np.asarray(time_values) - np.asarray(time_values)[0])
            / np.timedelta64(365, "D")
        ).astype(float)
    else:
        years = np.arange(field.sizes["time"], dtype=float)

    x = xr.DataArray(years, coords={"time": time}, dims="time")

    valid = field.notnull()
    valid_fraction = valid.mean("time")

    x_mean = x.where(valid).mean("time")
    y_mean = field.mean("time")
    covariance = ((x - x_mean) * (field - y_mean)).where(valid).mean("time")
    variance = ((x - x_mean) ** 2).where(valid).mean("time")

    slope = covariance / variance.where(variance > 0)
    intercept = y_mean - slope * x_mean

    if per_decade:
        slope = slope * 10.0
        slope_units = f"{field.attrs.get('units', 'unknown')} per decade"
    else:
        slope_units = f"{field.attrs.get('units', 'unknown')} per year"

    mask = valid_fraction >= float(min_valid_fraction)
    slope = slope.where(mask)
    intercept = intercept.where(mask)

    result = xr.Dataset(
        {
            "trend": slope,
            "intercept": intercept,
            "valid_fraction": valid_fraction,
            "time_mean": y_mean,
        }
    )
    result["trend"].attrs = {"units": slope_units, "long_name": f"{field.name} linear trend"}
    result["valid_fraction"].attrs = {"units": "1"}
    result.attrs = {
        "source_variable": str(field.name),
        "source_file": str(input_file),
        "n_timesteps": int(field.sizes["time"]),
    }
    return result
