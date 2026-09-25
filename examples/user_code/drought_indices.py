"""Standardized drought/wetness index from a timeseries CSV.

Plain user science code -- no ESMFlow imports. Onboarded as a *diagnostics*
capability by the onboarding agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _read_timeseries(input_file: str, value_column: str | None) -> pd.Series:
    frame = pd.read_csv(input_file)

    time_col = next(
        (c for c in frame.columns if c.lower() in ("time", "date", "datetime")),
        frame.columns[0],
    )
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).set_index(time_col)

    if value_column:
        if value_column not in frame.columns:
            raise ValueError(
                f"Column '{value_column}' not in {input_file}; "
                f"available: {list(frame.columns)}"
            )
        series = frame[value_column]
    else:
        numeric = frame.select_dtypes("number")
        if numeric.empty:
            raise ValueError(f"No numeric columns found in {input_file}")
        series = numeric.iloc[:, 0]

    return pd.to_numeric(series, errors="coerce").dropna()


def compute_standardized_anomaly_index(
    input_file: str,
    value_column: str = None,
    accumulation_months: int = 3,
    drought_threshold: float = -1.0,
) -> dict:
    """Compute a standardized anomaly (SPI-like) drought index for a timeseries.

    The series is accumulated over a rolling window, then standardized against
    its own calendar-month climatology so that values are in units of standard
    deviations from normal. Negative excursions indicate drought.

    Args:
        input_file: CSV timeseries with a time column and at least one numeric column.
        value_column: Column to analyze. Defaults to the first numeric column.
        accumulation_months: Rolling accumulation window in months.
        drought_threshold: Index value below which a month counts as drought.

    Returns:
        Dict with the per-time index table, the drought month count, and the
        most negative index value observed.
    """

    series = _read_timeseries(input_file, value_column)
    if series.empty:
        raise ValueError(f"No usable data in {input_file}")

    accumulated = series.rolling(
        window=max(1, int(accumulation_months)), min_periods=1
    ).sum()

    month = accumulated.index.month
    climatology_mean = accumulated.groupby(month).transform("mean")
    climatology_std = accumulated.groupby(month).transform("std").replace(0.0, np.nan)

    index = (accumulated - climatology_mean) / climatology_std
    index = index.fillna(0.0)

    table = pd.DataFrame(
        {
            "time": accumulated.index,
            "value": series.to_numpy(),
            "accumulated": accumulated.to_numpy(),
            "anomaly_index": index.to_numpy(),
            "in_drought": (index < drought_threshold).to_numpy(),
        }
    )

    return {
        "index_file": table,
        "n_drought_months": int(table["in_drought"].sum()),
        "min_index": float(table["anomaly_index"].min()),
    }
