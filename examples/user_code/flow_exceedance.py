"""Flow duration curve exceedance thresholds for a streamflow timeseries.

Deliberately plain user code: pandas only, no ESMFlow imports. Used to exercise
the onboarding path with a function that is not yet registered.
"""
from __future__ import annotations

import pandas as pd


def compute_flow_exceedance_thresholds(
    input_file: str,
    flow_column: str | None = None,
    low_flow_percentile: float = 95.0,
    high_flow_percentile: float = 5.0,
) -> dict:
    """Compute high- and low-flow exceedance thresholds from a streamflow CSV.

    Reads a daily streamflow timeseries and reports the flow values exceeded
    ``high_flow_percentile`` and ``low_flow_percentile`` percent of the time
    (Q5 and Q95 by default), plus the ratio between them as a flashiness proxy.

    Args:
        input_file: CSV timeseries with a time column and a streamflow column.
        flow_column: Streamflow column name. Defaults to the first numeric column.
        low_flow_percentile: Exceedance percentile defining low flow (0-100).
        high_flow_percentile: Exceedance percentile defining high flow (0-100).

    Returns:
        Dict with the exceedance table, the two threshold values and their ratio.
    """
    frame = pd.read_csv(input_file)
    if flow_column is None:
        numeric = frame.select_dtypes("number")
        if numeric.empty:
            raise ValueError(f"No numeric column found in {input_file}")
        flow_column = str(numeric.columns[0])

    flow = frame[flow_column].dropna().astype(float)
    if flow.empty:
        raise ValueError(f"Column {flow_column!r} has no usable values")

    high_flow = float(flow.quantile(1.0 - high_flow_percentile / 100.0))
    low_flow = float(flow.quantile(1.0 - low_flow_percentile / 100.0))

    exceedance = pd.DataFrame(
        {
            "percentile": [high_flow_percentile, 50.0, low_flow_percentile],
            "flow": [
                high_flow,
                float(flow.quantile(0.5)),
                low_flow,
            ],
        }
    )

    return {
        "exceedance_file": exceedance,
        "high_flow_threshold": high_flow,
        "low_flow_threshold": low_flow,
        "flashiness_ratio": high_flow / low_flow if low_flow else float("nan"),
    }
