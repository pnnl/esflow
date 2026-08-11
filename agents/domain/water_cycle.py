"""Basin-Scale Water Cycle Synthesis subagent."""

from agents.domain._factory import make_domain_subagent
from common.workflow import BasinScaleWaterCycleSynthesisStep


agent, call_water_cycle_synthesis = make_domain_subagent(
    BasinScaleWaterCycleSynthesisStep,
    "Basin-Scale Water Cycle Synthesis",
    "water cycle",
    "call_water_cycle_synthesis",
    "Create basin-scale water cycle synthesis steps.",
)
