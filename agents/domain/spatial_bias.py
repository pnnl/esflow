"""Spatial Bias Deep-Dive subagent.

Specialises in field-level model-vs-observation bias analysis:
  extract_gridded_field → compute_spatial_bias → compute_zonal_stats
  → plot_bias_comparison / plot_gridded_map

Use this agent when the user asks about spatial patterns of bias, map-level
model-obs differences, zonal mean errors, or regional mean statistics for
gridded variables (precipitation, ET, runoff, temperature, etc.).
"""

from agents.domain._factory import make_domain_subagent
from common.workflow import SpatialBiasStep

agent, call_spatial_bias = make_domain_subagent(
    SpatialBiasStep,
    "Spatial Bias Deep-Dive",
    "spatial bias",
    "call_spatial_bias",
    "Create spatial bias analysis steps (field extraction, bias computation, map plots).",
    instructions=(
        "You are the Spatial Bias Deep-Dive subagent. "
        "Your role is to build workflow steps that quantify and visualise the spatial "
        "difference between E3SM model output and gridded observations. "
        "Typical step sequences: "
        "(1) extract_gridded_field for the model variable, "
        "(2) extract_gridded_field for the matching observation dataset, "
        "(3) compute_spatial_bias (field_a=model, field_b=obs to get model-minus-obs), "
        "(4) optionally compute_zonal_stats to summarise by latitude band, "
        "(5) plot_bias_comparison for the 3-panel obs/sim/bias figure, "
        "    and/or plot_gridded_map for individual field maps. "
        "Return only new steps for your category. "
        "Use ${step_id.outputs.key} to reference prior step outputs. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)
