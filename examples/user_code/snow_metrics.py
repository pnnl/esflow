"""Snow season metrics from an SWE timeseries CSV.

Plain user science code -- no ESMFlow imports. Onboarded as a *diagnostics*
capability by the onboarding agent.
"""

from __future__ import annotations

import pandas as pd


def _load_swe(input_file: str, swe_column: str | None) -> pd.Series:
    frame = pd.read_csv(input_file)
    time_col = next(
        (c for c in frame.columns if c.lower() in ("time", "date", "datetime")),
        frame.columns[0],
    )
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).set_index(time_col).sort_index()

    if swe_column:
        if swe_column not in frame.columns:
            raise ValueError(
                f"Column '{swe_column}' not in {input_file}; "
                f"available: {list(frame.columns)}"
            )
        series = frame[swe_column]
    else:
        candidates = [c for c in frame.columns if "swe" in c.lower()]
        numeric = frame[candidates] if candidates else frame.select_dtypes("number")
        if numeric.empty:
            raise ValueError(f"No numeric SWE column found in {input_file}")
        series = numeric.iloc[:, 0]

    return pd.to_numeric(series, errors="coerce").dropna()


def compute_snow_season_metrics(
    input_file: str,
    swe_column: str | None = None,
    swe_threshold: float = 0.005,
    water_year_start_month: int = 10,
) -> dict:
    """Summarize snow season timing and magnitude per water year.

    For each water year the function reports peak snow water equivalent and its
    date, the first and last day exceeding ``swe_threshold`` (snow onset and
    melt-out), and the resulting snow-covered duration.

    Args:
        input_file: CSV timeseries containing a time column and an SWE column.
        swe_column: SWE column name. Defaults to a column containing 'swe'.
        swe_threshold: SWE value (same units as input) defining snow presence.
        water_year_start_month: Month that starts the water year (1-12).

    Returns:
        Dict with the per-water-year metrics table, the number of water years,
        and the mean snow-covered duration in days.
    """

    swe = _load_swe(input_file, swe_column)
    if swe.empty:
        raise ValueError(f"No usable SWE data in {input_file}")

    start_month = int(water_year_start_month)
    index = pd.DatetimeIndex(swe.index)
    water_year = index.year + (index.month >= start_month).astype(int)

    rows = []
    for year, group in swe.groupby(water_year):
        snow_days = group[group > float(swe_threshold)]
        peak_date = pd.Timestamp(group.idxmax())
        if snow_days.empty:
            onset = melt_out = None
            duration = 0
        else:
            onset = snow_days.index.min()
            melt_out = snow_days.index.max()
            duration = int((melt_out - onset).days) + 1
        rows.append(
            {
                "water_year": int(year),
                "peak_swe": float(group.max()),
                "peak_swe_date": peak_date.date().isoformat(),
                "snow_onset_date": onset.date().isoformat() if onset is not None else "",
                "melt_out_date": melt_out.date().isoformat() if melt_out is not None else "",
                "snow_duration_days": duration,
                "mean_swe": float(group.mean()),
            }
        )

    table = pd.DataFrame(rows)

    return {
        "metrics_file": table,
        "n_water_years": int(len(table)),
        "mean_snow_duration_days": float(table["snow_duration_days"].mean()),
    }
