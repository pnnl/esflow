"""ESM Diagnostics and Skill Metrics subagent."""

from agents.domain._factory import make_domain_subagent
from common.workflow import DiagnosticsAndSkillMetricsStep


agent, call_diagnostics = make_domain_subagent(
    DiagnosticsAndSkillMetricsStep,
    "ESM Diagnostics and Skill Metrics",
    "diagnostics",
    "call_diagnostics",
    "Create diagnostics and metric computation steps.",
)
