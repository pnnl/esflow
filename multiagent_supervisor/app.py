from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

from workflow import (
    BasinScaleWaterCycleSynthesisStep,
    DataDiscoveryStep,
    DiagnosticsAndSkillMetricsStep,
    DiagnosticVisualizationStep,
    Settings,
    SpatialTemporalExtractionStep,
    Workflow,
)


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


class RuntimeConfig(BaseSettings):
    """Runtime configuration loaded from environment and .env file."""

    AI_INCUBATOR_KEY: str

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


runtime_config = RuntimeConfig()


def _load_prompt(extra_instructions: str = "") -> str:
    base_prompt = (_HERE / "system_prompt.md").read_text(encoding="utf-8")
    tool_catalog = (_REPO_ROOT / "tool_catalog_generated.yaml").read_text(encoding="utf-8")
    return (
        f"{base_prompt}\n\n"
        f"## Tool Catalog\n\n"
        f"```yaml\n{tool_catalog}\n```\n\n"
        f"{extra_instructions}".strip()
    )


@dataclass
class WorkflowState:
    """Shared mutable workflow state used by the supervisor and subagents."""

    workflow: Workflow

    def context_summary(self) -> str:
        """Summarize existing steps with output key->filename mappings."""
        if not self.workflow.steps:
            return "No steps yet. You are creating the beginning of the workflow."

        lines = []
        for step in self.workflow.steps:
            outputs_text = ", ".join(f"{k}: {v}" for k, v in step.outputs.items())
            lines.append(f"- {step.id} (tool={step.tool}) -> {outputs_text}")

        return (
            "Existing steps. Use their outputs with ${step_id.outputs.key}:\n"
            + "\n".join(lines)
        )


model = AnthropicModel(
    "claude-haiku-4-5-20251001-v1-project",
    provider=AnthropicProvider(
        api_key=runtime_config.AI_INCUBATOR_KEY,
        base_url="https://ai-incubator-api.pnnl.gov",
    ),
)


def _make_subagent(category_name: str, result_type):
    return Agent(
        model,
        output_type=result_type,
        instructions=_load_prompt(
            f"You are the {category_name} subagent. "
            "Return only new steps for your category. "
            "Rely on the typed output schema to enforce allowed tool names."
        ),
    )


data_discovery_agent: Agent[None, List[DataDiscoveryStep]] = _make_subagent(
    "ESM Data Discovery and Intake", List[DataDiscoveryStep]
)

extraction_agent: Agent[None, List[SpatialTemporalExtractionStep]] = _make_subagent(
    "ESM Spatial-Temporal Extraction", List[SpatialTemporalExtractionStep]
)

diagnostics_agent: Agent[None, List[DiagnosticsAndSkillMetricsStep]] = _make_subagent(
    "ESM Diagnostics and Skill Metrics", List[DiagnosticsAndSkillMetricsStep]
)

water_cycle_agent: Agent[None, List[BasinScaleWaterCycleSynthesisStep]] = _make_subagent(
    "Basin-Scale Water Cycle Synthesis", List[BasinScaleWaterCycleSynthesisStep]
)

visualization_agent: Agent[None, List[DiagnosticVisualizationStep]] = _make_subagent(
    "ESM Diagnostic Visualization", List[DiagnosticVisualizationStep]
)


supervisor: Agent[WorkflowState, Workflow] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=Workflow,
    instructions=_load_prompt(
        "You are the Workflow Planning and Routing supervisor. "
        "Break requests into subproblems and call subagents in dependency order: "
        "data -> extraction -> diagnostics/water cycle -> visualization. "
        "After composing steps, return the complete structured Workflow object."
    ),
)


def _with_context(state: WorkflowState, task: str) -> str:
    return f"{state.context_summary()}\n\nTask:\n{task}"


@supervisor.tool
async def call_data_discovery(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create data intake steps (fetch/load metadata)."""
    result = await data_discovery_agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} data discovery step(s): {[s.id for s in result.output]}"


@supervisor.tool
async def call_extraction(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create extraction and matching steps."""
    result = await extraction_agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} extraction step(s): {[s.id for s in result.output]}"


@supervisor.tool
async def call_diagnostics(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create diagnostics and metric computation steps."""
    result = await diagnostics_agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} diagnostics step(s): {[s.id for s in result.output]}"


@supervisor.tool
async def call_water_cycle_synthesis(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create basin-scale water cycle synthesis steps."""
    result = await water_cycle_agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} water cycle step(s): {[s.id for s in result.output]}"


@supervisor.tool
async def call_visualization(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create visualization steps for diagnostics outputs."""
    result = await visualization_agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} visualization step(s): {[s.id for s in result.output]}"


@supervisor.tool
def get_workflow(ctx: RunContext[WorkflowState]) -> str:
    """Return the currently assembled workflow as JSON."""
    return ctx.deps.workflow.model_dump_json(indent=2)


async def build_workflow(user_goal: str, settings: Settings) -> Workflow:
    """Generate an ESMFlow workflow from a user goal using the supervisor chain."""
    state = WorkflowState(
        workflow=Workflow(
            name="Generated Workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )
    )

    result = await supervisor.run(user_goal, deps=state)
    return result.output


app = supervisor.to_web()