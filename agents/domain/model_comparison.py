"""Cross-Case Model Comparison subagent.

Specialises in side-by-side evaluation of two or more E3SM simulation cases
against observations or against each other:
  extract_e3sm_timeseries / extract_gridded_field (per case)
  → compute_metrics / compute_spatial_bias / compute_summary_stats
  → plot_bias_comparison / plot_map / plot_timeseries / plot_scatter

Use this agent when the user asks about:
- Comparing two E3SM runs (different resolution, parameterisation, time period)
- Quantifying improvement/degradation between model versions
- Side-by-side skill scores (NSE, KGE, PBIAS) across cases
- Relative bias maps between runs
"""

from agents.domain._factory import make_domain_subagent
from common.workflow import ModelComparisonStep

agent, call_model_comparison = make_domain_subagent(
    ModelComparisonStep,
    "Cross-Case Model Comparison",
    "model comparison",
    "call_model_comparison",
    "Create cross-case model comparison steps (extraction, skill metrics, comparison plots).",
    instructions=(
        "You are the Cross-Case Model Comparison subagent. "
        "Your role is to build workflow steps that compare two or more E3SM simulation "
        "cases against each other or against a common observational reference. "
        "Typical step sequences: "
        "(1) match_to_grid for each case (or reuse prior matching step outputs), "
        "(2) extract_e3sm_timeseries for each case independently, "
        "(3) extract_obs_timeseries (shared reference), "
        "(4) compute_metrics for per-case skill scores (NSE, KGE, PBIAS, RMSE), "
        "(5) compute_spatial_bias for each case's gridded bias field, "
        "(6) compute_summary_stats to rank gauges by improvement, "
        "(7) plot_bias_comparison for obs/case-A/case-B side-by-side maps, "
        "    plot_map to show per-gauge skill metric, "
        "    plot_timeseries for time-series overlays across cases, "
        "    plot_scatter for mean-value comparison. "
        "When comparing two cases, suffix step IDs clearly: e.g. "
        "extract_sim_case_a, extract_sim_case_b, metrics_case_a, metrics_case_b. "
        "Return only new steps for your category. "
        "Use ${step_id.outputs.key} to reference prior step outputs. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)
