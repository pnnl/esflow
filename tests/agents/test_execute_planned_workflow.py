from pathlib import Path

import pandas as pd

from agents.planner.oneshot_planner_executor import execute_planned_workflow
from common.workflow import DiagnosticsAndSkillMetricsStep


async def test_execute_planned_workflow_blocks_empty_workflow(run_context):
    result = await execute_planned_workflow(run_context)

    assert result.status == "execution_blocked"
    assert result.step_statuses == []
    assert "No workflow steps exist" in result.message


async def test_execute_planned_workflow_blocks_completeness_gaps(run_context):
    run_context.deps.workflow.steps.append(
        DiagnosticsAndSkillMetricsStep(
            id="stats",
            tool="compute_summary_stats",
            params={"timeseries_file": ""},
        )
    )

    result = await execute_planned_workflow(run_context)

    assert result.status == "execution_blocked"
    assert "stats.timeseries_file is empty" in result.message


async def test_execute_planned_workflow_blocks_catalog_validation_errors(run_context):
    run_context.deps.workflow.steps.append(
        DiagnosticsAndSkillMetricsStep(id="stats", tool="compute_summary_stats")
    )

    result = await execute_planned_workflow(run_context)

    assert result.status == "execution_blocked"
    assert "missing required input: timeseries_file" in result.message


async def test_execute_planned_workflow_runs_valid_workflow_and_persists_plan(run_context, tmp_path):
    input_csv = tmp_path / "timeseries.csv"
    pd.DataFrame(
        {"gauge": [1.0, 2.0]}, index=pd.date_range("2000-01-01", periods=2)
    ).to_csv(input_csv)
    output_dir = tmp_path / "output"
    run_context.deps.workflow.settings.output_dir = str(output_dir)
    run_context.deps.workflow.steps.append(
        DiagnosticsAndSkillMetricsStep(
            id="stats",
            tool="compute_summary_stats",
            params={"timeseries_file": str(input_csv)},
            outputs={"stats_file": "renamed_stats.csv"},
        )
    )

    result = await execute_planned_workflow(run_context)

    assert result.status == "executed"
    assert [(status.step_id, status.status) for status in result.step_statuses] == [("stats", "completed")]
    assert Path(result.workflow_file).is_file()
    assert result.output_dir == str(output_dir)
    assert (output_dir / "renamed_stats.csv").is_file()
