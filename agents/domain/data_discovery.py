"""ESM Data Discovery and Intake subagent."""

from agents.domain._factory import make_domain_subagent
from common.workflow import DataDiscoveryStep


agent, call_data_discovery = make_domain_subagent(
    DataDiscoveryStep,
    "ESM Data Discovery and Intake",
    "data discovery",
    "call_data_discovery",
    "Create data intake steps (fetch/load metadata).",
)
