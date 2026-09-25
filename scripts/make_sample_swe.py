#!/usr/bin/env python3
"""Generate a mock daily Snow Water Equivalent timeseries for manual testing.

Written for exercising ``examples/user_code/snow_metrics.py`` (and its onboarded
capability ``compute_snow_season_metrics``) by hand, without needing real data.

The output CSV matches what that tool expects by default:

* a ``time`` column of daily dates,
* an SWE column whose name contains ``swe`` (``swe_m``), in **metres**, so the
  default ``swe_threshold=0.005`` reads as a physically sensible 5 mm of snow,
* several complete October-start water years, so the default
  ``water_year_start_month=10`` produces more than one row of metrics.

A deliberately distracting extra column (``tair_c``) is included so the
column-sniffing path in the tool is genuinely exercised rather than trivially
picking the only numeric column.

Rather than drawing an arbitrary curve, SWE is produced by a tiny
temperature-index snow model: precipitation falls as snow below a threshold
temperature and melts above it at a fixed degree-day rate. That yields a
realistic asymmetric season -- slow winter accumulation, a spring peak, a fast
melt-out -- and interannual variability, so the metrics differ year to year
instead of being suspiciously identical.

Deterministic: a fixed seed means repeated runs produce byte-identical output,
so results you see in the chat can be compared against a later run.

Usage:
    python scripts/make_sample_swe.py
    python scripts/make_sample_swe.py --start 2015-10-01 --water-years 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sample" / "user_code" / "sample_swe_timeseries.csv"

# Temperature-index snow model constants.
SNOW_THRESHOLD_C = 0.5  # precipitation falls as snow at or below this
MELT_RATE_MM_PER_C_PER_DAY = 3.2
MEAN_TEMPERATURE_C = 3.5
TEMPERATURE_AMPLITUDE_C = 13.0
WET_DAY_PROBABILITY = 0.38
MEAN_WET_DAY_PRECIP_MM = 7.5


def _seasonal_temperature(day_of_year: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Sinusoidal annual temperature cycle, coldest in mid-January."""
    # Phase chosen so the minimum lands near day 15 of the calendar year.
    phase = 2.0 * math.pi * (day_of_year - 15) / 365.25
    return MEAN_TEMPERATURE_C - TEMPERATURE_AMPLITUDE_C * np.cos(phase) + noise


def _simulate(dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Run the temperature-index model over ``dates``."""
    rng = np.random.default_rng(seed)

    day_of_year = dates.dayofyear.to_numpy()
    # Day-to-day weather noise. Kept modest (2 K) on purpose: with a larger
    # spread, one freak cold-and-wet day in late summer deposits >5 mm of snow
    # and, because the tool reports the *last* day above the threshold, that
    # single blip pushes melt-out into September and inflates the season
    # duration to ~350 days. Physically defensible, confusing as sample data.
    temperature = _seasonal_temperature(
        day_of_year,
        noise=rng.normal(0.0, 2.0, size=len(dates)),
    )

    # Wet days are more frequent in the cool season; precipitation is gamma-ish.
    wet_season_boost = 1.0 + 0.45 * np.cos(2.0 * math.pi * (day_of_year - 15) / 365.25)
    wet = rng.random(len(dates)) < (WET_DAY_PROBABILITY * wet_season_boost)
    precip_mm = np.where(
        wet,
        rng.gamma(shape=1.6, scale=MEAN_WET_DAY_PRECIP_MM / 1.6, size=len(dates)),
        0.0,
    )

    swe_mm = np.zeros(len(dates))
    storage = 0.0
    for i, (temp, precip) in enumerate(zip(temperature, precip_mm)):
        if temp <= SNOW_THRESHOLD_C:
            storage += precip
        else:
            melt = MELT_RATE_MM_PER_C_PER_DAY * (temp - SNOW_THRESHOLD_C)
            storage = max(0.0, storage - melt)
        swe_mm[i] = storage

    return pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "swe_m": np.round(swe_mm / 1000.0, 6),
            "tair_c": np.round(temperature, 2),
        }
    )


def build(start: str = "2016-10-01", water_years: int = 5, seed: int = 20240501) -> pd.DataFrame:
    """Build a daily SWE table spanning whole October-start water years."""
    first = pd.Timestamp(start)
    last = pd.Timestamp(year=first.year + water_years, month=first.month, day=first.day)
    dates = pd.date_range(first, last, freq="D", inclusive="left")
    return _simulate(dates, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-10-01", help="first date (a water-year start)")
    parser.add_argument("--water-years", type=int, default=5, help="number of whole water years")
    parser.add_argument("--seed", type=int, default=20240501, help="RNG seed for reproducibility")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination CSV")
    args = parser.parse_args()

    table = build(start=args.start, water_years=args.water_years, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    swe = table["swe_m"]
    snow_days = int((swe >= 0.005).sum())
    print(f"wrote {len(table)} daily rows to {args.output}")
    print(f"  columns:        {', '.join(table.columns)}")
    print(f"  period:         {table['time'].iloc[0]} .. {table['time'].iloc[-1]}")
    print(f"  peak SWE:       {swe.max():.3f} m")
    print(f"  days >= 5 mm:   {snow_days} of {len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
