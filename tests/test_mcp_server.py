from pathlib import Path

import pandas as pd
from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

import mcp_server
from common.workflow import DiagnosticsAndSkillMetricsStep, Settings, Workflow


def _stats_workflow(tmp_path: Path) -> Workflow:
    input_csv = tmp_path / "timeseries.csv"
    pd.DataFrame(
        {"gauge": [1.0, 2.0]}, index=pd.date_range("2000-01-01", periods=2)
    ).to_csv(input_csv)

    return Workflow(
        name="MCP test workflow",
        description="Compute summary statistics through MCP.",
        settings=Settings(
            data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output")
        ),
        steps=[
            DiagnosticsAndSkillMetricsStep(
                id="stats",
                tool="compute_summary_stats",
                params={"timeseries_file": str(input_csv)},
                outputs={"stats_file": "stats.csv"},
            )
        ],
    )


async def test_validate_workflow_over_mcp(tmp_path):
    workflow = _stats_workflow(tmp_path)

    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "validate_workflow", {"workflow": workflow.model_dump()}
        )

    assert result.data == []


async def test_execute_workflow_over_mcp_returns_structured_result(tmp_path):
    workflow = _stats_workflow(tmp_path)

    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "execute_workflow", {"workflow": workflow.model_dump()}
        )

    execution = result.structured_content
    output_dir = Path(workflow.settings.output_dir)
    assert execution["output_dir"] == str(output_dir)
    assert execution["workflow_file"] == str(output_dir / "workflow.yaml")
    assert Path(execution["workflow_file"]).is_file()
    assert execution["step_statuses"] == [
        {"step_id": "stats", "status": "completed", "error": None}
    ]
    assert execution["outputs"] == {
        "stats": {"stats_file": str(output_dir / "stats.csv")}
    }
    assert (output_dir / "stats.csv").is_file()


async def test_execute_workflow_reports_validation_errors_before_writing_files(tmp_path):
    output_dir = tmp_path / "output"
    workflow = Workflow(
        name="Invalid MCP workflow",
        description="Missing a required tool parameter.",
        settings=Settings(data_dir=str(tmp_path / "data"), output_dir=str(output_dir)),
        steps=[
            DiagnosticsAndSkillMetricsStep(
                id="stats",
                tool="compute_summary_stats",
            )
        ],
    )

    async with Client(mcp_server.mcp) as client:
        with pytest.raises(ToolError, match="missing required input: timeseries_file"):
            await client.call_tool("execute_workflow", {"workflow": workflow.model_dump()})

    assert not output_dir.exists()


async def test_execute_workflow_blocks_placeholder_values_before_writing_files(tmp_path):
    output_dir = tmp_path / "output"
    workflow = Workflow(
        name="Incomplete MCP workflow",
        description="Contains an unfilled required tool parameter.",
        settings=Settings(data_dir=str(tmp_path / "data"), output_dir=str(output_dir)),
        steps=[
            DiagnosticsAndSkillMetricsStep(
                id="stats",
                tool="compute_summary_stats",
                params={"timeseries_file": "UNKNOWN"},
            )
        ],
    )

    async with Client(mcp_server.mcp) as client:
        with pytest.raises(ToolError, match="stats.timeseries_file is a placeholder"):
            await client.call_tool("execute_workflow", {"workflow": workflow.model_dump()})

    assert not output_dir.exists()


async def test_execute_workflow_rejects_unknown_start_from_before_writing_files(tmp_path):
    output_dir = tmp_path / "output"
    workflow = _stats_workflow(tmp_path)
    workflow.settings.output_dir = str(output_dir)

    async with Client(mcp_server.mcp) as client:
        with pytest.raises(ToolError, match="start_from step 'missing' not found"):
            await client.call_tool(
                "execute_workflow",
                {"workflow": workflow.model_dump(), "start_from": "missing"},
            )

    assert not output_dir.exists()


async def test_plan_workflow_over_mcp_uses_structured_workflow(tmp_path, monkeypatch):
    expected_settings = Settings(
        data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output")
    )

    async def fake_plan(user_goal, settings):
        assert user_goal == "Create a workflow"
        assert settings == expected_settings
        return Workflow(
            name="Planned workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )

    monkeypatch.setattr(mcp_server, "plan_workflow_one_shot", fake_plan)

    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "plan_workflow",
            {
                "user_goal": "Create a workflow",
                "data_dir": expected_settings.data_dir,
                "output_dir": expected_settings.output_dir,
            },
        )

    workflow = result.structured_content
    assert workflow == {
        "name": "Planned workflow",
        "description": "Create a workflow",
        "settings": expected_settings.model_dump(),
        "steps": [],
    }
