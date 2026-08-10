"""ESM Diagnostic Visualization subagent."""

from agents.domain._factory import make_domain_subagent
from common.workflow import DiagnosticVisualizationStep


agent, call_visualization = make_domain_subagent(
    DiagnosticVisualizationStep,
    "ESM Diagnostic Visualization",
    "visualization",
    "call_visualization",
    "Create visualization steps for diagnostics outputs.",
)
