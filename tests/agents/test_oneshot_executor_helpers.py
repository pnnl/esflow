from pathlib import Path

from agents.planner.oneshot_planner_executor import (
    _collect_plot_urls,
    _plot_url,
    _step_statuses_from_execution_context,
)
from common.workflow import DataDiscoveryStep, DiagnosticVisualizationStep


def test_plot_url_only_exposes_files_under_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_plot = tmp_path / "output" / "run" / "plot.png"
    output_plot.parent.mkdir(parents=True)
    output_plot.touch()

    assert _plot_url(output_plot) == "/output/run/plot.png"
    assert _plot_url(tmp_path / "other" / "plot.png") is None


def test_collect_plot_urls_filters_non_png_and_deduplicates(workflow_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    workflow_state.workflow.steps.extend(
        [
            DataDiscoveryStep(id="metadata", tool="load_obs_metadata"),
            DiagnosticVisualizationStep(id="plot", tool="plot_timeseries"),
        ]
    )
    context = {
        "metadata": {"outputs": {"metadata_file": str(tmp_path / "output" / "metadata.csv")}},
        "plot": {
            "outputs": {
                "plot_file": str(tmp_path / "output" / "plots" / "result.png"),
                "duplicate": str(tmp_path / "output" / "plots" / "result.png"),
            }
        },
    }

    assert _collect_plot_urls(workflow_state.workflow, context) == ["/output/plots/result.png"]


def test_step_statuses_prioritize_errors_and_preserve_runner_states(workflow_state):
    workflow_state.workflow.steps.extend(
        [
            DataDiscoveryStep(id="completed", tool="load_obs_metadata"),
            DataDiscoveryStep(id="failed", tool="load_obs_metadata"),
            DataDiscoveryStep(id="reused", tool="load_obs_metadata"),
            DataDiscoveryStep(id="skipped", tool="load_obs_metadata"),
        ]
    )
    context = {
        "failed": {"result": {"error": "failed to load"}},
        "reused": {"result": {"reused": True}},
        "skipped": {"result": {"skipped": True}},
    }

    statuses = _step_statuses_from_execution_context(workflow_state.workflow, context)
    assert [(status.step_id, status.status, status.error) for status in statuses] == [
        ("completed", "completed", None),
        ("failed", "failed", "failed to load"),
        ("reused", "reused", None),
        ("skipped", "skipped", None),
    ]
