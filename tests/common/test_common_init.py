from common import _with_context
from common.workflow import DataDiscoveryStep


def test_workflow_state_context_summary_for_empty_workflow(workflow_state):
    assert workflow_state.context_summary() == "No steps yet. You are creating the beginning of the workflow."


def test_workflow_state_context_summary_and_task_include_existing_outputs(workflow_state):
    workflow_state.workflow.steps.append(
        DataDiscoveryStep(
            id="metadata",
            tool="load_obs_metadata",
            outputs={"metadata_file": "gauge_metadata.csv"},
        )
    )

    assert workflow_state.context_summary() == (
        "Existing steps. Use their outputs with ${step_id.outputs.key}:\n"
        "- metadata (tool=load_obs_metadata) -> metadata_file: gauge_metadata.csv"
    )
    assert _with_context(workflow_state, "Summarize observations") == (
        "Existing steps. Use their outputs with ${step_id.outputs.key}:\n"
        "- metadata (tool=load_obs_metadata) -> metadata_file: gauge_metadata.csv\n\n"
        "Task:\nSummarize observations"
    )
