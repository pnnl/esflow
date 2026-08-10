import pandas as pd

from tools.analyzers.compute_summary_stats import run


def test_compute_summary_stats_generates_ranked_statistics(tmp_path):
    timeseries = tmp_path / "timeseries.csv"
    pd.DataFrame(
        {"100": [1.0, 3.0, None], "200": [10.0, 20.0, 30.0]},
        index=pd.date_range("2000-01-01", periods=3),
    ).to_csv(timeseries)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"gauge_id": ["100", "200"], "river_name": ["A", "B"], "area_km2": [1.0, 2.0]}
    ).to_csv(metadata, index=False)

    result = run(
        {
            "timeseries_file": timeseries,
            "gauge_metadata": metadata,
            "top_n": "1",
            "rank_by": "mean",
            "output_dir": tmp_path / "output",
        }
    )

    stats = pd.read_csv(result["stats_file"])
    assert result["n_columns"] == 1
    assert stats["column_name"].astype(str).tolist() == ["200"]
    assert stats.loc[0, "mean"] == 20.0
    assert stats.loc[0, "river_name"] == "B"
