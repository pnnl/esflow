"""Extreme Events / Flow Statistics subagent.

Specialises in streamflow distribution analysis, drought and flood diagnostics:
  match_to_grid → extract_e3sm_timeseries / extract_obs_timeseries
  → compute_fdc_metrics / compute_climatology / compute_summary_stats
  → plot_fdc / plot_timeseries / plot_basin_timeseries

Use this agent when the user asks about:
- Flow duration curves (FDCs) or distributional streamflow metrics
- High/low flow extremes, drought or flood frequency
- Seasonal climatology of discharge
- Summary statistics across gauges (mean, peak, low-flow percentiles)
"""

from agents.domain._factory import make_domain_subagent
from common.workflow import ExtremesFlowStep

agent, call_extremes = make_domain_subagent(
    ExtremesFlowStep,
    "Extreme Events and Flow Statistics",
    "extremes/flow",
    "call_extremes",
    "Create extreme events and flow statistics steps (FDC metrics, climatology, flow plots).",
    instructions=(
        "You are the Extreme Events and Flow Statistics subagent. "
        "Your role is to build workflow steps that analyse the distribution and "
        "extremes of streamflow (and related variables) for model evaluation and "
        "drought/flood diagnostics. "
        "Typical step sequences: "
        "(1) match_to_grid to align gauges to E3SM grid (if not already done), "
        "(2) extract_e3sm_timeseries and extract_obs_timeseries to get discharge series, "
        "(3) compute_fdc_metrics for distributional skill (volume bias, Wasserstein distance, "
        "    Q10/Q50/Q90 ratios), "
        "(4) compute_climatology for monthly mean discharge cycles, "
        "(5) compute_summary_stats for peak/low-flow rankings, "
        "(6) plot_fdc for FDC map + panel figure, "
        "    plot_timeseries for sim-vs-obs time series, "
        "    plot_basin_timeseries for per-basin spatial+temporal view. "
        "Return only new steps for your category. "
        "Use ${step_id.outputs.key} to reference prior step outputs. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)
