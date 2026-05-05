# ESFlow

A module-grounded framework for Earth System Model analysis. Scientists register analysis tools with typed metadata; any LLM reads the auto-generated tool catalog and composes YAML workflows. The human reviews, the engine executes.

## Associated Paper

This repository accompanies the manuscript:

> **Can We Trust LLMs for Complex Earth System Model Analysis? Silent Failure and Evidence from Module-Grounded Benchmarking**
> Tian Zhou, Yun Qian, L. Ruby Leung
> Pacific Northwest National Laboratory
> Submitted to *Geoscientific Model Development* (GMD), 2026

The paper introduces ESFlow and benchmarks it against unconstrained LLM code generation across six contemporary LLMs and seven E3SM land-surface-hydrology analysis tasks, with a focus on silent failures — plausible, well-formatted output that numerically disagrees with hand-crafted references. A frozen snapshot of this repository — together with the sample data (E3SM output, GRDC streamflow) and full per-model benchmark outputs — is archived on Zenodo (DOI: 10.5281/zenodo.19350842).

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

1. **Tools** are Python functions decorated with `@esmflow_tool(ToolSpec(...))`. The decorator registers typed inputs and outputs, validates parameters, and rejects unknown params.
2. **`generate_catalog.py`** auto-discovers all tools and writes `tool_catalog.yaml` — the sole interface between LLMs and tools.
3. **Any LLM** (ChatGPT, Claude, Gemini, Llama, etc.) reads the catalog and generates a YAML workflow from a natural-language request.
4. **`run_workflow.py`** executes the YAML step-by-step, passing outputs between tools via `${step_id.outputs.key}` references.

## Design Principles (v3)

**LLMs connect building blocks. Tools handle internals. Minimize decisions the LLM must make.**

- **15 tools, 45 total params** (v2 had 16 tools with ~90 params)
- **Strict validation**: unknown parameters are rejected immediately
- **No modal behavior**: each tool does exactly one thing, no mode/format switches
- **Match by column name**: no `match_by`, `x_column`, `y_column` params — tools match gauge_id columns automatically
- **Standardized CSV schemas**: 5 schemas, all lowercase column names
- **Simple observation format**: one CSV per gauge (`date`, `discharge_m3s`)

## Quick Start

```bash
# Clone and install
git clone https://github.com/pnnl-int/esflow.git
cd esflow
pip install -r requirements.txt

# Validate a workflow (no data needed)
python run_workflow.py workflows/examples/obs_only_validation.yaml --dry-run

# Run the self-test workflow (uses included sample data)
python run_workflow.py workflows/examples/obs_only_validation.yaml

# Reuse existing intermediate files
python run_workflow.py workflows/examples/obs_only_validation.yaml --reuse
```

## Generating Workflows with an LLM

1. Combine `docs/system_prompt.txt` + `tools/tool_catalog.yaml` as the system message
2. Describe your analysis task in natural language as the user message
3. Save the generated YAML and validate with `--dry-run`
4. Run it

See **[docs/guide.md](docs/guide.md)** for the full protocol, task description examples, and instructions for adding your own tools.

## Tools (15)

| # | Category | Tool | Params | Description |
|---|----------|------|--------|-------------|
| 1 | fetchers | `fetch_ilamb_data` | 2 | Download obs NetCDF from ILAMB server (18 variable/dataset pairs) |
| 2 | loaders | `load_obs_metadata` | 1 | Load and validate gauge metadata CSV |
| 3 | matchers | `match_to_grid` | 4 | Match gauges to E3SM model grid |
| 4 | extractors | `extract_e3sm_timeseries` | 7 | Extract model time series at gauge locations (monthly/daily) |
| 5 | extractors | `extract_obs_timeseries` | 3 | Extract obs from per-gauge CSVs |
| 6 | extractors | `extract_gridded_field` | 6 | Extract 2D field from E3SM or obs NetCDF; supports composite variables |
| 7 | analyzers | `compute_climatology` | 1 | Monthly means (1-12) |
| 8 | analyzers | `compute_spatial_bias` | 2 | Spatial bias (A minus B) with regridding |
| 9 | analyzers | `compute_zonal_stats` | 3 | Area-weighted global/latitude-band means |
| 10 | analyzers | `compute_summary_stats` | 4 | Mean/std/min/max per column with optional ranking |
| 11 | analyzers | `compute_metrics` | 2 | NSE, KGE, PBIAS, RMSE, correlation |
| 12 | plotters | `plot_scatter` | 2 | Mean discharge scatter plot |
| 13 | plotters | `plot_gridded_map` | 3 | 2D field on geographic map |
| 14 | plotters | `plot_timeseries` | 2 | Sim vs obs time series panels |
| 15 | plotters | `plot_map` | 3 | Metric values at gauge locations on a map |

## Standard Data Schemas

Tools communicate through typed CSV files. The catalog declares exact column schemas so an LLM can wire tools together without guessing.

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

ESFlow includes 94 representative GRDC gauges for testing:

```
data/sample/obs/
├── gauge_metadata.csv              # 94 gauges: gauge_id, lat, lon, area_km2, river_name
└── streamflow/
    ├── 1147013.csv                 # date, discharge_m3s (Congo at Kinshasa)
    ├── 2181900.csv                 # date, discharge_m3s (Yangtze at Datong)
    └── ...                         # 94 files total
```

## Example Workflows

| Workflow | Steps | What it does |
|----------|-------|-------------|
| `obs_only_validation.yaml` | 7 | Self-test: obs vs obs (all metrics should be perfect) |
| `single_gauge_comparison.yaml` | 6 | Compare E3SM MOSART discharge vs observations |
| `multi_gauge_validation.yaml` | 9 | Full validation with metrics, climatology, scatter, and map |

## Benchmarking LLMs

ESFlow includes a benchmark system that evaluates LLMs in two modes:

- **Protocol mode**: LLM generates a YAML workflow using the tool catalog
- **Baseline mode**: LLM generates free-form Python code (no tools)

### Reproducing the Paper Benchmark

The benchmark in the paper is reproduced in four stages. Sample data (E3SM output and GRDC streamflow) is not included in this repository to keep it lightweight — a frozen snapshot of this code together with the sample data and full per-model benchmark outputs is archived on Zenodo (DOI: 10.5281/zenodo.19350842).

```bash
# 1. Download sample data from Zenodo and unpack into data/
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
│   ├── tool_catalog.yaml        # LLM-readable tool metadata (15 tools, 45 params)
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
│   ├── extractors/              # Time series & field extraction tools (3)
│   ├── analyzers/               # Metrics, stats, bias, zonal tools (5)
│   └── plotters/                # Visualization tools (4)
├── data/sample/                 # Sample observation data (94 GRDC gauges)
├── workflows/examples/          # Example & reference YAML workflows
├── benchmark/
│   ├── protocol/                # System prompt + task descriptions (protocol mode)
│   ├── baselines/               # Task descriptions for free-form Python (baseline mode)
│   ├── run_benchmark.py         # Multi-model, mode-aware benchmark runner
│   ├── structural_grading.py    # 3-step reproducible grading script
│   └── results/                 # Outputs, scores, manual review labels
├── scripts/                     # Data conversion utilities
└── archive/v2/                  # Archived v2 tools (16 tools, ~90 params)
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
