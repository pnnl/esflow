"""Observational Data Survey subagent.

Specialises in discovering, fetching, and summarising available observation
datasets before workflow composition:
  fetch_ilamb_data → load_obs_metadata → extract_obs_timeseries
  → compute_summary_stats / compute_climatology
  → plot_scatter / plot_map / plot_gridded_map / plot_timeseries

Use this agent when the user asks about:
- What observation datasets are available for a given variable
- Downloading ILAMB gridded observations (precipitation, ET, runoff, LAI, etc.)
- Surveying gauge network coverage and data availability
- Computing summary statistics or climatological averages from observations alone
- Quick visual overview of observational datasets before model comparison
"""

from agents.domain._factory import make_domain_subagent
from common.workflow import ObsSurveyStep

agent, call_obs_survey = make_domain_subagent(
    ObsSurveyStep,
    "Observational Data Survey",
    "obs survey",
    "call_obs_survey",
    "Create observational data survey steps (fetch, load, summarise, plot observations).",
    instructions=(
        "You are the Observational Data Survey subagent. "
        "Your role is to build workflow steps that fetch, load, and characterise "
        "observational datasets — both gridded (ILAMB) and in-situ gauge records — "
        "prior to or independently of model comparison. "
        "Typical step sequences: "
        "(1) fetch_ilamb_data to download a gridded observation NetCDF (for variables "
        "    like pr, evspsbl, mrro, lai, gpp, tas, twsa, swe), "
        "(2) load_obs_metadata to validate and load gauge metadata CSV, "
        "(3) extract_obs_timeseries to extract per-gauge discharge or other records, "
        "(4) compute_summary_stats to rank gauges by mean/peak flow, "
        "(5) compute_climatology for the monthly mean cycle of the observation, "
        "(6) plot_scatter for mean-value overview across gauges, "
        "    plot_map to show spatial distribution of a statistic at gauge locations, "
        "    plot_gridded_map to visualise the fetched gridded observation field, "
        "    plot_timeseries for time-series overview of selected gauges. "
        "Return only new steps for your category. "
        "Use ${step_id.outputs.key} to reference prior step outputs. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)
