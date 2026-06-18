# Workflow Comparison

- A: `multiagent_supervisor/workflow_test_run.yaml`
- B: `multiagent_supervisor/workflow_benchmark_task04.yaml`

## Summary

| Field | Workflow A | Workflow B |
|---|---|---|
| Name | Precipitation Spatial Bias Workflow | E3SM MOSART Streamflow FDC Analysis (1985-1989) |
| Description focus | Precipitation spatial bias | Streamflow FDC benchmarking |
| Step count | 5 | 6 |
| Case name | test_run | sample.v3.LR.historical |
| Data dir | ../data | ./data/sample/e3sm/ |
| Output dir | ../outputs | ./output/task04_streamflow_fdc |

## Tool Coverage

- Only in A: compute_spatial_bias, extract_gridded_field, fetch_ilamb_data, plot_bias_comparison
- Only in B: compute_fdc_metrics, extract_e3sm_timeseries, extract_obs_timeseries, load_obs_metadata, match_to_grid, plot_fdc
- Shared: -

## Step Role Alignment

| Role | Workflow A step | Tool | Workflow B step | Tool |
|---|---|---|---|---|
| Ingest | fetch_ilamb_precip | fetch_ilamb_data | load_metadata | load_obs_metadata |
| Align / match | - | - | match_gauges_to_grid | match_to_grid |
| Extract model | extract_e3sm_precip | extract_gridded_field | extract_e3sm_discharge | extract_e3sm_timeseries |
| Extract obs | extract_obs_precip_field | extract_gridded_field | extract_obs_discharge | extract_obs_timeseries |
| Diagnose | compute_precip_spatial_bias | compute_spatial_bias | compute_fdc_metrics_1985_1989 | compute_fdc_metrics |
| Visualize | plot_precip_bias_comparison | plot_bias_comparison | plot_fdc_comparison_1985_1989 | plot_fdc |

## Dependency Chains

### Workflow A

- fetch_ilamb_precip: (no step refs)
- extract_e3sm_precip: settings.data_dir, settings.case_name
- extract_obs_precip_field: fetch_ilamb_precip.outputs.data_file
- compute_precip_spatial_bias: extract_e3sm_precip.outputs.field_file, extract_obs_precip_field.outputs.field_file
- plot_precip_bias_comparison: extract_obs_precip_field.outputs.field_file, extract_e3sm_precip.outputs.field_file, compute_precip_spatial_bias.outputs.bias_file, compute_precip_spatial_bias.outputs.stats_file

### Workflow B

- load_metadata: (no step refs)
- match_gauges_to_grid: load_metadata.outputs.metadata_file, settings.data_dir, settings.case_name
- extract_e3sm_discharge: settings.data_dir, settings.case_name, match_gauges_to_grid.outputs.matched_file
- extract_obs_discharge: load_metadata.outputs.metadata_file
- compute_fdc_metrics_1985_1989: extract_e3sm_discharge.outputs.timeseries_file, extract_obs_discharge.outputs.timeseries_file
- plot_fdc_comparison_1985_1989: compute_fdc_metrics_1985_1989.outputs.metrics_file, compute_fdc_metrics_1985_1989.outputs.fdc_file, load_metadata.outputs.metadata_file

## Notes

- Workflow A is a compact map-bias pipeline with one observational source and gridded field operations.
- Workflow B adds metadata loading and grid matching, then evaluates distributional behavior via FDC metrics.
- Workflow B is closer to benchmark-style hydrologic evaluation, while A is a minimal smoke-style diagnostic flow.