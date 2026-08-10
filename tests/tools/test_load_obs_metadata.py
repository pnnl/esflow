import pandas as pd
import pytest

from tools.loaders.load_obs_metadata import run


def test_load_obs_metadata_validates_and_writes_a_copy(tmp_path):
    source = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "gauge_id": [123],
            "lat": [45.0],
            "lon": [-120.0],
            "area_km2": [12.5],
            "river_name": ["Example"],
        }
    ).to_csv(source, index=False)

    result = run({"metadata_file": source, "output_dir": tmp_path / "output"})
    output = pd.read_csv(result["metadata_file"])
    assert result["n_gauges"] == 1
    assert output["gauge_id"].astype(str).tolist() == ["123"]


def test_load_obs_metadata_rejects_missing_required_columns(tmp_path):
    source = tmp_path / "metadata.csv"
    pd.DataFrame({"gauge_id": [123], "lat": [45.0]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        run({"metadata_file": source, "output_dir": tmp_path / "output"})
