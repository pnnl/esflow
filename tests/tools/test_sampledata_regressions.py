from pathlib import Path

import pandas as pd
import pytest

from evals.structural_grading_helpers import csv_matches
from tools.analyzers.compute_summary_stats import run


@pytest.mark.sampledata
def test_compute_summary_stats_matches_task_01_reference(
    tmp_path, sample_data_dir: Path, reference_results_dir: Path
):
    result = run(
        {
            "timeseries_file": reference_results_dir / "task_01_obs_summary" / "obs_timeseries.csv",
            "gauge_metadata": sample_data_dir / "obs" / "gauge_metadata.csv",
            "output_dir": tmp_path,
        }
    )

    actual = Path(result["stats_file"])
    expected = reference_results_dir / "task_01_obs_summary" / "summary_stats.csv"
    matches, reason = csv_matches(actual, expected)
    assert matches, reason


@pytest.mark.sampledata
def test_sample_metadata_has_expected_columns(sample_data_dir: Path):
    metadata = pd.read_csv(sample_data_dir / "obs" / "gauge_metadata.csv")
    assert {"gauge_id", "river_name", "area_km2"}.issubset(metadata.columns)
