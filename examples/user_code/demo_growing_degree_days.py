"""Reference "user code" for the onboarding walkthrough.

This file is the worked example the onboarding agent shows a scientist who asks
"what does my code have to look like to be onboardable?". It is deliberately
plain science code: **no ESMFlow imports, no decorators, no tool contract**. The
onboarding machinery reads the signature and docstring below and generates the
adapter for you.

Everything the introspector relies on is annotated with a ``FORMAT:`` comment.
The five rules, in short:

1. **Module-level, public, synchronous function.** Nested, underscore-prefixed
   and ``async`` functions are skipped. Helpers can (and should) be private --
   ``_read_daily_temperature`` below is skipped on purpose.
2. **Type hints on every argument.** They become the tool's parameter types, and
   the runner coerces incoming workflow YAML values to them. A ``str`` argument
   whose name looks like a file gets the ``path`` type; annotate deliberately and
   confirm what the agent inferred.
3. **A docstring with a one-line summary and an ``Args:`` section.** The summary
   becomes the planner-visible tool description -- the *only* text the planner
   uses when deciding whether to select your tool -- and each ``Args:`` entry
   becomes that parameter's description.
4. **Return a dict of named results** when there is more than one output. Keys
   ending in ``_file`` (or holding a DataFrame/Dataset/Figure) become artifacts
   written to the workflow's output directory; plain numbers and strings are
   returned inline in the step result.
5. **Make each returned scalar's type explicit at the return boundary.** The
   introspector reads your ``return`` statement without executing it, so it can
   only see what is written there. ``int(...)``, ``float(...)``, ``str(...)`` and
   plain literals are read as certain; a bare variable name is not, and gets
   guessed (``float``) and flagged for you to confirm. Coercing in the returned
   dict is the cheapest way to get an exact tool contract first time.
6. **Take an input path, not hard-coded paths, and never write files yourself.**
   ESMFlow passes ``output_dir`` and materializes your returned objects, so your
   function stays a pure computation that is easy to test.

Whatever the introspector could not determine with certainty it reports as a
note rather than guessing silently -- always read those notes back before
registering.

Where this file lives matters too: user code belongs under
``examples/user_code/`` (or any importable package in the repo), *not* under
``tools/``. ``tools/<category>/`` holds the generated adapters, which are
machine-written -- see ``docs/onboarding_design.md``.
"""

from __future__ import annotations

import pandas as pd


# FORMAT: private helper. The leading underscore keeps it out of the onboarding
# scan, so shared plumbing does not turn into accidental tools.
def _read_daily_temperature(input_file: str, temperature_column: str | None) -> pd.Series:
    """Load a daily temperature series, indexed by date."""
    frame = pd.read_csv(input_file)

    time_columns = [c for c in frame.columns if c.lower() in ("time", "date", "day")]
    if not time_columns:
        raise ValueError(
            f"{input_file} needs a time column named one of: time, date, day"
        )
    index = pd.to_datetime(frame[time_columns[0]])

    if temperature_column is None:
        numeric = frame.select_dtypes("number").columns
        if len(numeric) == 0:
            raise ValueError(f"{input_file} has no numeric column to use")
        temperature_column = str(numeric[0])
    elif temperature_column not in frame.columns:
        raise ValueError(
            f"column {temperature_column!r} is not in {input_file}; "
            f"available: {', '.join(map(str, frame.columns))}"
        )

    series = pd.Series(
        pd.to_numeric(frame[temperature_column], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(index, name="time"),
        name=str(temperature_column),
    )
    return series.dropna().sort_index()


# FORMAT: public, module-level, synchronous -> this is what gets onboarded.
def compute_growing_degree_days(
    # FORMAT: a path-like first argument. Annotated `str`, named `*_file`, so the
    # introspector proposes the `path` type and the planner knows to wire an
    # upstream step's output into it.
    input_file: str,
    # FORMAT: optional arguments with literal defaults become optional tool
    # params carrying the same default. `str | None` is understood.
    temperature_column: str | None = None,
    base_temperature: float = 10.0,
    upper_cutoff: float = 30.0,
) -> dict:
    """Accumulate growing degree days from a daily mean temperature timeseries.

    Growing degree days (GDD) measure the heat available for plant development.
    Each day contributes ``max(0, min(T, upper_cutoff) - base_temperature)``
    degree-days, and the season total is the cumulative sum. Useful for
    comparing the thermal growing season between model runs or against
    observations.

    Args:
        input_file: CSV with a time column (time/date/day) and a daily mean
            temperature column in degrees Celsius.
        temperature_column: Name of the temperature column. Defaults to the
            first numeric column in the file.
        base_temperature: Lower threshold in degrees Celsius; days at or below
            it contribute nothing.
        upper_cutoff: Temperature in degrees Celsius above which extra heat no
            longer accelerates development.

    Returns:
        A dict with the daily accumulation table plus season-total scalars.
    """
    # FORMAT: the docstring above is not decoration. Its summary becomes the
    # tool description, and each `Args:` entry becomes a param description.
    temperature = _read_daily_temperature(input_file, temperature_column)

    effective = temperature.clip(upper=upper_cutoff) - base_temperature
    daily = effective.clip(lower=0.0)
    cumulative = daily.cumsum()

    table = pd.DataFrame(
        {
            "temperature": temperature,
            "degree_days": daily,
            "cumulative_degree_days": cumulative,
        }
    )

    growing_days = int((daily > 0).sum())
    first_growing_day = (
        str(daily[daily > 0].index[0].date()) if growing_days else "none"
    )

    # FORMAT: a dict of named results. `gdd_file` holds a DataFrame and is
    # declared a `csv` output, so ESMFlow writes it into the run's output_dir;
    # the scalars come back inline in the step result. Returning the DataFrame
    # rather than writing it yourself is what keeps this function testable and
    # lets the workflow control where artifacts land.
    #
    # FORMAT: note the explicit int()/float()/str() calls. The introspector reads
    # this dict statically, so those coercions are what let it type each output
    # exactly instead of guessing. `growing_days` is already an int and
    # `first_growing_day` already a str -- the casts are here to declare intent
    # to the reader and to the introspector alike.
    return {
        "gdd_file": table,
        "total_degree_days": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
        "growing_season_days": int(growing_days),
        "first_growing_day": str(first_growing_day),
    }
