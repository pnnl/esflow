"""ESM Spatial-Temporal Extraction subagent."""

from agents.domain._factory import make_domain_subagent
from common.workflow import SpatialTemporalExtractionStep


agent, call_extraction = make_domain_subagent(
    SpatialTemporalExtractionStep,
    "ESM Spatial-Temporal Extraction",
    "extraction",
    "call_extraction",
    "Create extraction and matching steps.",
)
