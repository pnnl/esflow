import pandas as pd
import pytest

from tools.plotters.plot_timeseries import run


def test_plot_timeseries_creates_nonempty_png(tmp_path):
    index = pd.date_range("2000-01-01", periods=3)
    sim = tmp_path / "sim.csv"
    obs = tmp_path / "obs.csv"
    pd.DataFrame({"gauge": [1.0, 2.0, 3.0]}, index=index).to_csv(sim)
    pd.DataFrame({"gauge": [1.5, 2.5, 3.5]}, index=index).to_csv(obs)

    result = run({"sim_file": sim, "obs_file": obs, "output_dir": tmp_path / "output"})
    plot = result["plot_file"]

    with open(plot, "rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"


def test_plot_timeseries_rejects_inputs_without_matching_columns(tmp_path):
    index = pd.date_range("2000-01-01", periods=2)
    sim = tmp_path / "sim.csv"
    obs = tmp_path / "obs.csv"
    pd.DataFrame({"simulated": [1.0, 2.0]}, index=index).to_csv(sim)
    pd.DataFrame({"observed": [1.0, 2.0]}, index=index).to_csv(obs)

    with pytest.raises(ValueError, match="No matching columns between sim and obs files"):
        run({"sim_file": sim, "obs_file": obs, "output_dir": tmp_path / "output"})
