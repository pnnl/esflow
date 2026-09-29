from common import _with_context
from common.workflow import DataDiscoveryStep


def _settings_header(workflow_state) -> str:
    s = workflow_state.workflow.settings
    return (
        "Workflow settings (resolved from environment — use ${settings.<key>} in params):\n"
        f"  data_dir:   {s.data_dir}\n"
        f"  output_dir: {s.output_dir}\n"
        "  case_name:  (auto-detected from E3SM filenames)\n"
    )


def test_workflow_state_context_summary_for_empty_workflow(workflow_state):
    assert workflow_state.context_summary() == (
        _settings_header(workflow_state)
        + "\nNo steps yet. You are creating the beginning of the workflow."
    )


def test_workflow_state_context_summary_and_task_include_existing_outputs(workflow_state):
    workflow_state.workflow.steps.append(
        DataDiscoveryStep(
            id="metadata",
            tool="load_obs_metadata",
            outputs={"metadata_file": "gauge_metadata.csv"},
        )
    )

    assert workflow_state.context_summary() == (
        _settings_header(workflow_state)
        + "\nExisting steps. Use their outputs with ${step_id.outputs.key}:\n"
        "- metadata (tool=load_obs_metadata) -> metadata_file: gauge_metadata.csv"
    )
    assert _with_context(workflow_state, "Summarize observations") == (
        _settings_header(workflow_state)
        + "\nExisting steps. Use their outputs with ${step_id.outputs.key}:\n"
        "- metadata (tool=load_obs_metadata) -> metadata_file: gauge_metadata.csv\n\n"
        "Task:\nSummarize observations"
    )
