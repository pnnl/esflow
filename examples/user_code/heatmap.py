"""Month-by-year heatmap of a timeseries CSV.

Plain user science code -- no ESMFlow imports. Onboarded as a *visualization*
capability by the onboarding agent (returns a matplotlib Figure -> PNG).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_month_year_heatmap(
    input_file: str,
    value_column: str | None = None,
    title: str = "Month-by-year heatmap",
    colormap: str = "viridis",
    aggregation: str = "mean",
) -> Figure:
    """Render a year (rows) by month (columns) heatmap of a timeseries.

    Useful for spotting seasonal cycle shifts and anomalous years at a glance.

    Args:
        input_file: CSV timeseries with a time column and a numeric value column.
        value_column: Column to plot. Defaults to the first numeric column.
        title: Figure title.
        colormap: Matplotlib colormap name.
        aggregation: How to aggregate multiple values per month ('mean', 'sum', 'max', 'min').

    Returns:
        A matplotlib Figure containing the heatmap.
    """

    frame = pd.read_csv(input_file)
    time_col = next(
        (c for c in frame.columns if c.lower() in ("time", "date", "datetime")),
        frame.columns[0],
    )
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col])

    if value_column:
        if value_column not in frame.columns:
            raise ValueError(
                f"Column '{value_column}' not in {input_file}; "
                f"available: {list(frame.columns)}"
            )
        column = value_column
    else:
        numeric = frame.select_dtypes("number")
        if numeric.empty:
            raise ValueError(f"No numeric columns found in {input_file}")
        column = str(numeric.columns[0])

    if aggregation not in ("mean", "sum", "max", "min"):
        raise ValueError(f"Unsupported aggregation '{aggregation}'")

    frame["_year"] = frame[time_col].dt.year
    frame["_month"] = frame[time_col].dt.month
    grid = (
        frame.pivot_table(
            index="_year", columns="_month", values=column, aggfunc=aggregation
        )
        .reindex(columns=range(1, 13))
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.32 * len(grid) + 1.5)))
    mesh = ax.pcolormesh(grid.columns, grid.index, grid.to_numpy(), cmap=colormap, shading="nearest")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=f"{column} ({aggregation})")
    fig.tight_layout()
    return fig
