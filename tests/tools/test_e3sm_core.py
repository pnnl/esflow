import numpy as np
import pandas as pd
import xarray as xr

from tools.core.e3sm import cftime_to_datetime, detect_mesh_type


class _CalendarValue:
    year = 2001
    month = 2

    def __str__(self):
        return "not-a-date"


class _UnknownTimeValue:
    def __str__(self):
        return "not-a-date"


def test_cftime_to_datetime_uses_standard_and_calendar_fallbacks():
    assert list(cftime_to_datetime(["2001-01-01", "2001-02-01"])) == list(
        pd.DatetimeIndex(["2001-01-01", "2001-02-01"])
    )
    assert list(cftime_to_datetime([_CalendarValue()])) == ["2001-02-01"]
    assert isinstance(cftime_to_datetime([_UnknownTimeValue()]), pd.RangeIndex)


def test_detect_mesh_type_for_known_and_fallback_dimension_layouts():
    assert detect_mesh_type(xr.Dataset({"value": (("lat", "lon"), np.zeros((1, 1)))})) == "latlon"
    assert detect_mesh_type(xr.Dataset({"value": ("ncol", [1])})) == "unstructured"
    assert detect_mesh_type(xr.Dataset({"value": (("row", "column"), np.zeros((1, 1)))})) == "latlon"
    assert detect_mesh_type(xr.Dataset({"value": ("station", [1])})) == "unstructured"
    assert detect_mesh_type(xr.Dataset({"value": ("time", [1])})) == "unknown"
