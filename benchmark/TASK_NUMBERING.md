# Task numbering

The internal task IDs in this repo (`task_01` … `task_07`) differ from the
task numbers used in the paper (Protocol-First Agentic AI Framework for
Earth System Model Analysis, GMD) for T4 and T5. T4 and T5 were swapped
late in the writing process; the benchmark artifacts were not renamed to
avoid breaking the frozen grading pipeline and run history.

| Internal (this repo)                | Paper  | Short name                  |
|-------------------------------------|--------|-----------------------------|
| `task_01_obs_summary`          | T1     | Observation Summary          |
| `task_02_seasonal_runoff`      | T2     | Global Runoff Map            |
| `task_03_et_benchmark`         | T3     | ET Bias Map                  |
| `task_04_streamflow_fdc`       | **T5** | Streamflow FDC               |
| `task_05_basin_streamflow`     | **T4** | Basin Streamflow             |
| `task_06_water_balance`        | T6     | Water Balance                |
| `task_07_integrated_diagnostic`| T7     | Basin Water Cycle Evaluation |

The reference YAMLs under `reference_workflows/` follow the internal
numbering (e.g. `task04_reference.yaml` is the FDC workflow; this is
paper T5).
