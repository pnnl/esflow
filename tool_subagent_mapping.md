# Tool to Subagent Mapping (ESM Domain, Non-Exclusive)

This mapping is intentionally compact and specific to Earth System Model (ESM) search, extraction, evaluation, and diagnostic reporting workflows.

## 1) ESM Data Discovery and Intake Subagent
Purpose: Find and ingest model/observation sources needed for a requested scientific diagnostic.

Mapped tools:
- fetch_ilamb_data
- load_obs_metadata

## 2) ESM Spatial-Temporal Extraction Subagent
Purpose: Align observations to model space and extract analysis-ready time series and gridded fields.

Mapped tools:
- match_to_grid
- extract_e3sm_timeseries
- extract_obs_timeseries
- extract_gridded_field
- extract_basin_mean

## 3) ESM Diagnostics and Skill Metrics Subagent
Purpose: Compute objective model skill and climatological/spatial diagnostics for search targets (streamflow, ET, runoff, bias patterns).

Mapped tools:
- compute_climatology
- compute_summary_stats
- compute_metrics
- compute_fdc_metrics
- compute_spatial_bias
- compute_zonal_stats

## 4) Basin-Scale Water Cycle Synthesis Subagent
Purpose: Aggregate component diagnostics into basin-level water-cycle consistency assessments.

Mapped tools:
- compute_basin_budget
- extract_basin_mean
- compute_fdc_metrics
- compute_spatial_bias

## 5) ESM Diagnostic Visualization Subagent
Purpose: Render figures for scientific interpretation, comparison, and reporting across basin, map, and time-series views.

Mapped tools:
- plot_map
- plot_gridded_map
- plot_timeseries
- plot_scatter
- plot_fdc
- plot_basin_timeseries
- plot_water_balance_basins
- plot_basin_budget_comparison
- plot_basin_radar
- plot_bias_comparison

## 6) Spatial Bias Deep-Dive Subagent  *(bespoke)*
Purpose: Focused field-level model-vs-observation spatial bias quantification and visualisation. Use for map-based bias analysis, zonal mean errors, and obs/sim/bias comparison figures.

Mapped tools:
- extract_gridded_field
- compute_spatial_bias
- compute_zonal_stats
- plot_gridded_map
- plot_bias_comparison

Typical chain:
  extract_gridded_field (model) → extract_gridded_field (obs)
  → compute_spatial_bias → compute_zonal_stats
  → plot_bias_comparison / plot_gridded_map

## 7) Extreme Events / Flow Statistics Subagent  *(bespoke)*
Purpose: Streamflow distribution analysis, flow duration curves, drought/flood diagnostics. Use for FDC skill metrics, extreme-flow characterisation, seasonal climatology of discharge.

Mapped tools:
- match_to_grid
- extract_e3sm_timeseries
- extract_obs_timeseries
- compute_fdc_metrics
- compute_climatology
- compute_summary_stats
- plot_fdc
- plot_timeseries
- plot_basin_timeseries

Typical chain:
  match_to_grid → extract_e3sm_timeseries + extract_obs_timeseries
  → compute_fdc_metrics → compute_climatology
  → plot_fdc / plot_timeseries / plot_basin_timeseries

## 8) Cross-Case Model Comparison Subagent  *(bespoke)*
Purpose: Side-by-side evaluation of two or more E3SM simulation cases. Use when comparing model versions, parameterisations, or resolutions against a shared observational reference.

Mapped tools:
- match_to_grid
- extract_e3sm_timeseries
- extract_gridded_field
- extract_obs_timeseries
- compute_metrics
- compute_spatial_bias
- compute_summary_stats
- plot_bias_comparison
- plot_map
- plot_timeseries
- plot_scatter

Typical chain:
  match_to_grid → extract_e3sm_timeseries (case_a) + extract_e3sm_timeseries (case_b)
  + extract_obs_timeseries
  → compute_metrics (case_a) + compute_metrics (case_b)
  → compute_spatial_bias (case_a) + compute_spatial_bias (case_b)
  → plot_bias_comparison / plot_map / plot_timeseries / plot_scatter

## 9) Observational Data Survey Subagent  *(bespoke)*
Purpose: Fetch, load, and characterise observational datasets independently of model comparison. Use for ILAMB data downloads, gauge network surveys, obs-only climatology, and dataset overviews.

Mapped tools:
- fetch_ilamb_data
- load_obs_metadata
- extract_obs_timeseries
- compute_summary_stats
- compute_climatology
- plot_scatter
- plot_map
- plot_gridded_map
- plot_timeseries

Typical chain:
  fetch_ilamb_data → plot_gridded_map (obs overview)
  load_obs_metadata → extract_obs_timeseries
  → compute_summary_stats / compute_climatology
  → plot_scatter / plot_map / plot_timeseries

## 10) Workflow Planning and Routing Subagent
Purpose: Plan end-to-end ESM diagnostic workflows by chaining the above subagents based on user intent and requested artifacts.

Mapped tools:
- All tools in tools/tool_catalog.yaml (through delegated subagents)

## Planner Routing Hints (Domain-Specific)

- If intent mentions spatial bias, map difference, zonal mean, or regional mean error:
  Route to **Spatial Bias Deep-Dive** subagent.
- If intent mentions FDC, flow duration curve, drought, flood, high/low flow, or extreme events:
  Route to **Extreme Events / Flow Statistics** subagent.
- If intent mentions two cases, model versions, parameterisation comparison, or improvement assessment:
  Route to **Cross-Case Model Comparison** subagent.
- If intent mentions observational survey, ILAMB fetch, dataset availability, or gauge inventory:
  Route to **Observational Data Survey** subagent.
- If intent mentions streamflow skill, NSE/KGE/PBIAS, or FDCs for a single case:
  Route to ESM Spatial-Temporal Extraction then ESM Diagnostics and Skill Metrics then ESM Diagnostic Visualization.
- If intent mentions ET/runoff/precipitation closure or water balance:
  Route to ESM Spatial-Temporal Extraction plus Basin-Scale Water Cycle Synthesis then ESM Diagnostic Visualization.
- Keep routing non-exclusive. Multi-subagent composition is expected for most scientific queries.
