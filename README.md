# ESFlow

A protocol-first framework for Earth System Model analysis. Scientists register analysis tools with typed metadata; any LLM reads the auto-generated tool catalog and composes YAML workflows. The human reviews, the engine executes.

## How It Works

```
Scientist writes tool  →  @esflow_tool decorator  →  tool_catalog.yaml (auto)
                                                            ↓
User describes task    →  Any LLM reads catalog    →  workflow.yaml
                                                            ↓
                          run_workflow.py           →  figures, metrics, CSVs
```

1. **Tools** are Python functions decorated with `@esflow_tool(ToolSpec(...))`. The decorator registers typed inputs and outputs, validates parameters, and rejects unknown params.
2. **`generate_catalog.py`** auto-discovers all tools and writes `tool_catalog.yaml` — the sole interface between LLMs and tools.
3. **Any LLM** (ChatGPT, Claude, Gemini, Llama, etc.) reads the catalog and generates a YAML workflow from a natural-language request.
4. **`run_workflow.py`** executes the YAML step-by-step, passing outputs between tools via `${step_id.outputs.key}` references.

## Design Principles (v3)

**LLMs connect building blocks. Tools handle internals. Minimize decisions the LLM must make.**

- **9 tools, 24 total params** (v2 had 16 tools with ~90 params)
- **Strict validation**: unknown parameters are rejected immediately
- **No modal behavior**: each tool does exactly one thing, no mode/format switches
- **Match by column name**: no `match_by`, `x_column`, `y_column` params — tools match gauge_id columns automatically
- **Standardized CSV schemas**: 5 schemas, all lowercase column names
- **Simple observation format**: one CSV per gauge (`date`, `discharge_m3s`)

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/esflow.git
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

## Tools (9)

| Category | Tool | Params | Description |
|----------|------|--------|-------------|
| loaders | `load_obs_metadata` | 1 | Load and validate gauge metadata CSV |
| matchers | `match_to_grid` | 4 | Match observation gauges to E3SM model grid cells |
| extractors | `extract_e3sm_timeseries` | 6 | Extract model time series at matched locations |
| extractors | `extract_obs_timeseries` | 3 | Extract observation time series from per-gauge CSVs |
| analyzers | `compute_metrics` | 2 | Compute NSE, KGE, PBIAS, RMSE, correlation |
| analyzers | `compute_climatology` | 1 | Compute monthly climatology (mean by month 1-12) |
| plotters | `plot_timeseries` | 2 | Multi-panel sim vs obs time series comparison |
| plotters | `plot_scatter` | 2 | Scatter plot of mean discharge (sim vs obs) |
| plotters | `plot_map` | 3 | Validation metric values on a geographic map |

## Standard Data Schemas

Tools communicate through typed CSV files. The catalog declares exact column schemas so an LLM can wire tools together without guessing.

| Schema | Index | Columns | Producers | Consumers |
|--------|-------|---------|-----------|-----------|
| **gauge_metadata** | row | `gauge_id`, `lat`, `lon`, `area_km2`, `river_name` | loaders | matchers, extractors, plotters |
| **matched_gauges** | row | `gauge_id`, `lat`, `lon`, `model_lat`, `model_lon`, `lat_idx`, `lon_idx` | matchers | extractors |
| **timeseries** | `time` | one column per `gauge_id` | extractors | analyzers, plotters |
| **climatology** | `month` (1-12) | one column per `gauge_id` | compute_climatology | plotters |
| **metrics** | row | `gauge_id`, `nse`, `kge`, `pbias`, `rmse`, `correlation`, `n_valid` | compute_metrics | plotters |

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

ESFlow includes a benchmark runner that evaluates how well different LLMs compose workflows from the tool catalog.

### Scoring Levels

| Level | What it tests | Automated? |
|-------|--------------|------------|
| S0 | Valid YAML with `steps` key | Yes |
| S1 | Passes `--dry-run` (correct tool names, required params, valid wiring) | Yes |
| S2 | Executes without runtime error (requires data) | Yes |
| S3 | Scientifically correct output (right variables, methods, interpretation) | Human review |

```bash
# Quick test
export LLM_API_KEY="your-key-here"
python benchmark/run_benchmark.py --task benchmark/protocol/task_01_obs_summary.txt

# Full benchmark (all models, 3 runs each)
python benchmark/run_benchmark.py --all --runs 3

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

1. Create `tools/<category>/my_tool.py` with a `ToolSpec` and `@esflow_tool` decorator
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
│   ├── tool_catalog.yaml        # LLM-readable tool metadata (9 tools, 207 lines)
│   ├── core/                    # Framework internals
│   │   ├── base.py              # @esflow_tool, Param, ToolSpec, TOOL_REGISTRY
│   │   ├── e3sm.py              # ESM utilities (cftime, file discovery, dataset opening)
│   │   ├── schemas.py           # CSV schema documentation
│   │   ├── data_io.py           # MOSART/ELM data loading
│   │   ├── spatial.py           # Grid matching, river tracing
│   │   └── styling.py           # Plot style presets
│   ├── loaders/                 # Data loading tools (1)
│   ├── matchers/                # Gauge matching tools (1)
│   ├── extractors/              # Time series extraction tools (2)
│   ├── analyzers/               # Metrics and statistics tools (2)
│   └── plotters/                # Visualization tools (3)
├── data/sample/                 # Sample observation data (94 GRDC gauges)
├── workflows/examples/          # Example YAML workflows
├── benchmark/
│   ├── protocol/                # System prompt + task descriptions
│   ├── baselines/               # Code-gen baseline prompts
│   ├── run_benchmark.py         # Multi-model benchmark runner
│   └── results/                 # Auto-generated outputs + scores
├── scripts/                     # Data conversion utilities
└── archive/v2/                  # Archived v2 tools (16 tools, ~90 params)
```
