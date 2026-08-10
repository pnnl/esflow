from agents.planner.planner_tools import check_completeness, render_dag, run_workflow_validation
from common.workflow import DataDiscoveryStep, DiagnosticsAndSkillMetricsStep


async def test_check_completeness_reports_invalid_values_and_unknown_outputs(run_context):
    run_context.deps.workflow.steps.append(
        DiagnosticsAndSkillMetricsStep(
            id="stats",
            tool="compute_summary_stats",
            params={
                "timeseries_file": "",
                "gauge_metadata": "",
                "bad": None,
                "placeholder": "TBD",
                "missing": "${other.outputs.file}",
                "data_dir": "${settings.data_dir}",
            },
        )
    )

    assert await check_completeness(run_context) == [
        "stats.timeseries_file is empty",
        "stats.bad is null",
        "stats.placeholder is a placeholder",
        "stats.missing references unknown output '${other.outputs.file}'",
    ]


async def test_check_completeness_accepts_known_references(run_context):
    run_context.deps.workflow.steps.extend(
        [
            DataDiscoveryStep(
                id="metadata",
                tool="load_obs_metadata",
                outputs={"metadata_file": "metadata.csv"},
            ),
            DiagnosticsAndSkillMetricsStep(
                id="stats",
                tool="compute_summary_stats",
                params={
                    "timeseries_file": "${metadata.outputs.metadata_file}",
                    "gauge_metadata": "",
                },
            ),
        ]
    )

    assert await check_completeness(run_context) == []


async def test_render_dag_handles_empty_workflow_and_nested_references(run_context):
    assert await render_dag(run_context) == "_(no steps in the workflow yet)_"

    run_context.deps.workflow.steps.extend(
        [
            DataDiscoveryStep(id="metadata", tool="load_obs_metadata"),
            DiagnosticsAndSkillMetricsStep(
                id="stats",
                tool="compute_summary_stats",
                params={"inputs": ["${metadata.outputs.metadata_file}", {"same": "${stats.outputs.file}"}]},
            ),
        ]
    )

    assert await render_dag(run_context) == "\n".join(
        [
            "```mermaid",
            "flowchart TD",
            '    metadata["metadata: load_obs_metadata"]',
            '    stats["stats: compute_summary_stats"]',
            "    metadata --> stats",
            "```",
        ]
    )


async def test_run_workflow_validation_returns_errors_for_incomplete_workflow(run_context):
    run_context.deps.workflow.steps.append(
        DiagnosticsAndSkillMetricsStep(id="stats", tool="compute_summary_stats")
    )

    errors = await run_workflow_validation(run_context)
    assert any("missing required input: timeseries_file" in error for error in errors)
