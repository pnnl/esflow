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

## 6) Workflow Planning and Routing Subagent
Purpose: Plan end-to-end ESM diagnostic workflows by chaining the above subagents based on user intent and requested artifacts.

Mapped tools:
- All tools in tools/tool_catalog.yaml (through delegated subagents)

## Planner Routing Hints (Domain-Specific)
- If intent mentions streamflow skill, NSE/KGE/PBIAS, or FDCs:
  Route first to ESM Spatial-Temporal Extraction, then ESM Diagnostics and Skill Metrics, then ESM Diagnostic Visualization.
- If intent mentions ET/runoff/precipitation closure or water balance:
  Route to ESM Spatial-Temporal Extraction plus Basin-Scale Water Cycle Synthesis, then ESM Diagnostic Visualization.
- If intent mentions map bias, zonal pattern, or global/regional mean:
  Route to ESM Spatial-Temporal Extraction, then ESM Diagnostics and Skill Metrics, then ESM Diagnostic Visualization.
- Keep routing non-exclusive. Multi-subagent composition is expected for most scientific queries.
