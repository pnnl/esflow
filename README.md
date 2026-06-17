# ESFlow

A module-grounded framework for Earth System Model analysis. Scientists register analysis tools with typed metadata; any LLM reads the auto-generated tool catalog and composes YAML workflows. The human reviews, the engine executes.

## Associated Paper

This repository accompanies the manuscript:

> **Can We Trust LLMs for Complex Earth System Model Analysis? Silent Failure and Evidence from Module-Grounded Benchmarking**
> Tian Zhou, Yun Qian, L. Ruby Leung
> Pacific Northwest National Laboratory
> Submitted to *Geoscientific Model Development* (GMD), 2026

The paper introduces ESFlow and benchmarks it against unconstrained LLM code generation across six contemporary LLMs and seven E3SM land-surface-hydrology analysis tasks, with a focus on silent failures — plausible, well-formatted output that numerically disagrees with hand-crafted references. The Zenodo v2 data release, including the sample data used by the reference workflows, is available at <https://zenodo.org/records/20584449>.

Correspondence: Tian Zhou (<tian.zhou@pnnl.gov>).

### Cite this work

```bibtex
@article{zhou_2026_esflow,
  author  = {Zhou, Tian and Qian, Yun and Leung, L. Ruby},
  title   = {Can We Trust {LLMs} for Complex {Earth} System Model Analysis?
             Silent Failure and Evidence from Module-Grounded Benchmarking},
  journal = {Geoscientific Model Development},
  year    = {2026},
  note    = {Submitted}
}
```

The citation will be updated with volume, pages, and DOI once the paper is accepted.

## How It Works

```
Scientist writes tool  →  @esmflow_tool decorator  →  tool_catalog.yaml (auto)
                                                            ↓
User describes task    →  Any LLM reads catalog    →  workflow.yaml
                                                            ↓
                          run_workflow.py           →  figures, metrics, CSVs
```

1. **Tools** are Python functions decorated with `@esmflow_tool(ToolSpec(...))`. The decorator registers typed inputs and outputs, validates parameters, and rejects unknown parameters.
2. **`generate_catalog.py`** auto-discovers all tools and writes `tool_catalog.yaml` — the sole interface between LLMs and tools.
3. **Any LLM** (ChatGPT, Claude, Gemini, Llama, etc.) reads the catalog and generates a YAML workflow from a natural-language request.
4. **`run_workflow.py`** executes the YAML step-by-step, passing outputs between tools via `${step_id.outputs.key}` references.

## Design Principles

**LLMs connect building blocks. Tools handle internals. Minimize decisions the LLM must make.**

- **24 tools, 92 total input parameters**
- **Strict validation**: unknown parameters are rejected immediately
- **No modal behavior**: each tool does exactly one thing, no mode/format switches
- **Match by column name**: no `match_by`, `x_column`, `y_column` parameters — tools match gauge_id columns automatically
- **Standardized file schemas**: common CSV/NetCDF contracts are documented in the catalog and schema reference
- **Simple observation format**: one CSV per gauge (`date`, `discharge_m3s`)

## Quick Start

```bash
# Clone and install
git clone https://github.com/pnnl/esflow.git
cd esflow
pip install -r requirements.txt

# Validate a workflow (no data needed)
python run_workflow.py reference_workflows/task01_reference.yaml --dry-run

# Download the Zenodo v2 sample data and unpack it into data/ before execution
# https://zenodo.org/records/20584449
python run_workflow.py reference_workflows/task01_reference.yaml

# Reuse existing intermediate files
python run_workflow.py reference_workflows/task01_reference.yaml --reuse
```

## Generating Workflows with an LLM

1. Combine `docs/system_prompt.txt` + `tools/tool_catalog.yaml` as the system message
2. Describe your analysis task in natural language as the user message
3. Save the generated YAML and validate with `--dry-run`
4. Run it

See **[docs/guide.md](docs/guide.md)** for the full protocol, task description examples, and instructions for adding your own tools.

## Tools (24)

| # | Category | Tool | Parameters | Description |
|---|----------|------|--------|-------------|
| 1 | fetchers | `fetch_ilamb_data` | 2 | Download observation NetCDF files from the ILAMB server |
| 2 | loaders | `load_obs_metadata` | 1 | Load and validate gauge metadata CSV |
| 3 | matchers | `match_to_grid` | 4 | Match gauges to E3SM model grid |
| 4 | extractors | `extract_basin_mean` | 3 | Extract area-weighted basin means from gridded fields |
| 5 | extractors | `extract_e3sm_timeseries` | 7 | Extract model time series at gauge locations |
| 6 | extractors | `extract_obs_timeseries` | 4 | Extract observations from per-gauge CSVs |
| 7 | extractors | `extract_gridded_field` | 6 | Extract 2D fields from E3SM or observation NetCDF files |
| 8 | analyzers | `compute_climatology` | 1 | Compute monthly climatology from time series |
| 9 | analyzers | `compute_spatial_bias` | 2 | Compute gridded spatial bias and summary statistics |
| 10 | analyzers | `compute_zonal_stats` | 3 | Compute area-weighted global or latitude-band means |
| 11 | analyzers | `compute_summary_stats` | 4 | Compute mean/std/min/max per time-series column |
| 12 | analyzers | `compute_basin_budget` | 8 | Build per-basin water budget summary tables |
| 13 | analyzers | `compute_metrics` | 2 | Compute NSE, KGE, PBIAS, RMSE, and correlation |
| 14 | analyzers | `compute_fdc_metrics` | 2 | Compute flow duration curve distributional metrics |
| 15 | plotters | `plot_water_balance_basins` | 8 | Plot global residual maps with basin water-balance bars |
| 16 | plotters | `plot_scatter` | 2 | Plot mean-value scatter comparisons |
| 17 | plotters | `plot_gridded_map` | 5 | Plot 2D gridded fields on geographic maps |
| 18 | plotters | `plot_basin_radar` | 1 | Plot basin water-cycle diagnostic radar charts |
| 19 | plotters | `plot_timeseries` | 2 | Plot simulated vs observed time series panels |
| 20 | plotters | `plot_map` | 3 | Plot validation metrics at gauge locations |
| 21 | plotters | `plot_fdc` | 6 | Plot flow duration curve comparisons |
| 22 | plotters | `plot_basin_timeseries` | 6 | Plot basin maps with discharge time series |
| 23 | plotters | `plot_bias_comparison` | 9 | Plot observation, simulation, and bias map comparisons |
| 24 | plotters | `plot_basin_budget_comparison` | 1 | Plot basin water-budget component comparisons |

## Standard Data Schemas

Tools communicate through typed files. The table below lists common CSV contracts; tool-specific NetCDF, CSV, and figure outputs are documented in `tools/tool_catalog.yaml`.

| Schema | Index | Columns | Producers | Consumers |
|--------|-------|---------|-----------|-----------|
| **gauge_metadata** | row | `gauge_id`, `lat`, `lon`, `area_km2`, `river_name` | loaders | matchers, extractors, plotters |
| **matched_gauges** | row | `gauge_id`, `lat`, `lon`, `model_lat`, `model_lon`, `lat_idx`, `lon_idx` | matchers | extractors |
| **timeseries** | `time` | one column per `gauge_id` | extractors | analyzers, plotters |
| **climatology** | `month` (1-12) | one column per `gauge_id` | compute_climatology | plotters |
| **metrics** | row | `gauge_id`, `nse`, `kge`, `pbias`, `rmse`, `correlation`, `n_valid` | compute_metrics | plotters |
| **summary_stats** | row | `column_name`, `mean`, `std`, `min`, `max` (+ optional `river_name`, `area_km2`) | compute_summary_stats | — |
| **zonal_stats** | row | `region`, `mean`, `area_weighted_mean` | compute_zonal_stats | — |
| **bias_stats** | row | `mean_bias`, `rmse`, `spatial_correlation` | compute_spatial_bias | — |

## Sample Data

Sample data is distributed separately to keep this repository lightweight. Download the Zenodo v2 data release from <https://zenodo.org/records/20584449> and unpack it into `data/`.

```
data/sample/
├── e3sm/                           # Sample E3SM output used by reference workflows
└── obs/
    ├── gauge_metadata.csv          # gauge_id, lat, lon, area_km2, river_name
    └── streamflow/                 # per-gauge CSVs: date, discharge_m3s
```

## Example Workflows

| Workflow | Steps | What it does |
|----------|-------|-------------|
| `reference_workflows/task01_reference.yaml` | 3 | Observation streamflow summary statistics |
| `reference_workflows/task02_reference.yaml` | 3 | Seasonal runoff field extraction, statistics, and map |
| `reference_workflows/task03_reference.yaml` | 5 | Evapotranspiration benchmark against ILAMB observations |
| `reference_workflows/task04_reference.yaml` | 6 | Streamflow distribution and FDC comparison |
| `reference_workflows/task05_reference.yaml` | 6 | Basin-scale streamflow analysis |
| `reference_workflows/task06_reference.yaml` | 13 | Water-balance diagnostic workflow |
| `reference_workflows/task07_reference.yaml` | 23 | Integrated basin water-cycle evaluation |

## Benchmarking LLMs

ESFlow includes a benchmark system that evaluates LLMs in two modes:

- **Protocol mode**: LLM generates a YAML workflow using the tool catalog
- **Baseline mode**: LLM generates free-form Python code (no tools)

### Reproducing the Paper Benchmark

The benchmark in the paper is reproduced in five stages. Sample data (E3SM output and GRDC streamflow) is not included in this repository to keep it lightweight. Download the Zenodo v2 data release from <https://zenodo.org/records/20584449>.

```bash
# 1. Download sample data from Zenodo v2 and unpack into data/
#    (produces data/sample/e3sm/... and data/sample/obs/...)

# 2. Run the reference workflows to generate ground-truth outputs
for t in 01 02 03 04 05 06 07; do
    python run_workflow.py reference_workflows/task${t}_reference.yaml
done

# 3. Run the benchmark for both conditions (6 models x 7 tasks x 4 runs each)
export LLM_API_KEY="your-key-here"
python benchmark/run_benchmark.py --all --runs 4 --mode protocol
python benchmark/run_benchmark.py --all --runs 4 --mode baseline

# 4. Run the self-debug experiment on crashed runs (up to 3 repair rounds)
python benchmark/self_debug_crashes.py --max-rounds 3

# 5. Grade and merge results
python benchmark/structural_grading.py
python benchmark/grade_selfdebug.py
python benchmark/merge_grades.py
python benchmark/merge_grades_selfdebug.py
```

The paper's reference run (`claude-opus-4-6` protocol `run2`) is pre-included in `benchmark/results/claude-opus-4-6_protocol/` — so Step 2 only needs to be re-run if you change tools or want fresh reference outputs.

### 3-Step Structural Grading

| Step | What it checks | Applies to | Auto? |
|------|---------------|------------|-------|
| **Step 1: Crash** | Final deliverable missing (CSV for T1, PNG for T2–T7) | Both modes | Yes |
| **Step 2: Success** | Key data file matches reference within 1% tolerance | Protocol only | Yes |
| **Step 3: Manual review** | Human assigns silent failure or obvious failure | Undetermined | No |

**Final grades**: crash, success, silent failure, obvious failure.

Reference run: `claude-opus-4-6` protocol run2. Grading script: `benchmark/structural_grading.py`. Manual labels: `benchmark/results/manual_overrides.json`.

```bash
# Protocol mode (default)
export LLM_API_KEY="your-key-here"
python benchmark/run_benchmark.py --task benchmark/protocol/task_01_obs_summary.txt

# Baseline mode (free-form Python)
python benchmark/run_benchmark.py --task benchmark/baselines/task_01_obs_summary.txt --baseline

# Full benchmark (all models, 4 runs each, both modes)
python benchmark/run_benchmark.py --all --runs 4

# Run structural grading
python benchmark/structural_grading.py

# Local models via LM Studio
python benchmark/run_benchmark.py --local
```

## CLI Options

```
python run_workflow.py <workflow.yaml> [options]

Options:
  --dry-run, -n       Validate workflow without executing
  --verbose, -v       Print detailed progress
  --start-from, -s    Jump to a specific step (earlier steps assumed complete)
  --reuse, -r         Reuse existing intermediate files
```

## Adding a New Tool

See **[docs/guide.md](docs/guide.md)** for a complete walkthrough with examples. The short version:

1. Create `tools/<category>/my_tool.py` with a `ToolSpec` and `@esmflow_tool` decorator
2. Run `python tools/generate_catalog.py --overwrite` to update the catalog
3. The LLM sees your tool in the catalog automatically

## Project Structure

```
esflow/
├── run_workflow.py              # Workflow engine
├── requirements.txt
├── docs/
│   ├── guide.md                # Full protocol guide (adding tools, prompting LLMs)
│   └── system_prompt.txt       # System prompt template for LLM workflow generation
├── tools/
│   ├── generate_catalog.py      # Auto-generates tool_catalog.yaml
│   ├── tool_catalog.yaml        # LLM-readable tool metadata (24 tools, 92 input parameters)
│   ├── core/                    # Framework internals
│   │   ├── base.py              # @esmflow_tool, Param, ToolSpec, TOOL_REGISTRY
│   │   ├── e3sm.py              # ESM utilities (cftime, file discovery, dataset opening)
│   │   ├── schemas.py           # CSV schema documentation
│   │   ├── data_io.py           # MOSART/ELM data loading
│   │   ├── spatial.py           # Grid matching, river tracing
│   │   └── styling.py           # Plot style presets
│   ├── fetchers/                # Remote data fetchers (1)
│   ├── loaders/                 # Data loading tools (1)
│   ├── matchers/                # Gauge matching tools (1)
│   ├── extractors/              # Time series, basin, and field extraction tools (4)
│   ├── analyzers/               # Metrics, stats, bias, budget, and zonal tools (7)
│   └── plotters/                # Visualization tools (10)
├── reference_workflows/         # Human-reviewed benchmark reference workflows
├── data/sample/                 # Populated after downloading Zenodo sample data
├── benchmark/
│   ├── protocol/                # System prompt + task descriptions (protocol mode)
│   ├── baselines/               # Task descriptions for free-form Python (baseline mode)
│   ├── run_benchmark.py         # Multi-model, mode-aware benchmark runner
│   ├── structural_grading.py    # 3-step reproducible grading script
│   └── results/                 # Outputs, scores, manual review labels
├── visualize_workflow.py        # Workflow visualization helper
└── workflow_to_dag.py           # Workflow DAG conversion helper
```

## Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

```
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
```
