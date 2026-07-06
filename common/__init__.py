"""Shared state and utilities for supervisor and subagents."""

from dataclasses import dataclass

from common.workflow import Workflow


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


def _with_context(state: WorkflowState, task: str) -> str:
    """Format task with current workflow context."""
    return f"{state.context_summary()}\n\nTask:\n{task}"
