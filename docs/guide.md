# ESMFlow User Guide

This guide explains how to use ESMFlow's module-grounded workflow — from adding your own tools to generating and running YAML workflows with any LLM.

## The Protocol

ESMFlow separates *what tools can do* (defined by scientists) from *how to compose them* (delegated to an LLM). The protocol works in four stages:

```
1. Register tool    →  @esmflow_tool(ToolSpec(...))
2. Generate catalog →  python tools/generate_catalog.py --overwrite
3. Prompt any LLM   →  system_prompt.txt + tool_catalog.yaml + your task description
4. Validate & run   →  python run_workflow.py workflow.yaml --dry-run && python run_workflow.py workflow.yaml
```

The key artifact is `tools/tool_catalog.yaml` — a machine-readable description of every tool's name, inputs, outputs, and types. Any LLM reads this catalog and composes a YAML workflow. The human reviews the workflow before execution. No code is generated.

## 1. Adding a Tool

Each tool is a Python function in `tools/<category>/` decorated with `@esmflow_tool`. The decorator handles parameter validation, type coercion, unknown-param rejection, and output directory creation.

### Step 1: Create the tool file

```python
# tools/analyzers/compute_my_metric.py
from core.base import esmflow_tool, ToolSpec, Param

SPEC = ToolSpec(
    name='compute_my_metric',
    description='Compute custom metric between simulated and observed time series.',
    inputs={
        'sim_file': Param('path', required=True,
                         description='Simulated time series CSV (time index, gauge_id columns)'),
        'obs_file': Param('path', required=True,
                         description='Observed time series CSV (time index, gauge_id columns)'),
        'threshold': Param('float', required=False, default=0.5,
                          description='Threshold for flagging (0-1)'),
    },
    outputs={
        'metrics_file': {'type': 'csv', 'description': 'CSV with gauge_id and my_metric columns'},
        'n_gauges': {'type': 'int', 'description': 'Number of gauge pairs analyzed'},
    },
)

@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    import pandas as pd

    sim = pd.read_csv(config['sim_file'], index_col=0, parse_dates=True)
    obs = pd.read_csv(config['obs_file'], index_col=0, parse_dates=True)
    threshold = config['threshold']
    output_dir = config['output_dir']

    # Your analysis logic here
    results = []
    for col in sim.columns:
        if col in obs.columns:
            # compute your metric...
            results.append({'gauge_id': col, 'my_metric': 0.85})

    df = pd.DataFrame(results)
    out_path = f"{output_dir}/my_metrics.csv"
    df.to_csv(out_path, index=False)

    return {'metrics_file': out_path, 'n_gauges': len(results)}
```

### Step 2: Regenerate the catalog

```bash
python tools/generate_catalog.py --overwrite
```

This auto-discovers your new tool and adds it to `tools/tool_catalog.yaml`. The catalog entry is generated directly from your `ToolSpec` — no manual YAML editing needed.

### Step 3: Verify

```bash
# Check that your tool appears in the catalog
grep 'compute_my_metric' tools/tool_catalog.yaml
```

### Design guidelines

- **One tool, one job.** A tool should do exactly one thing. Don't add mode switches.
- **Typed inputs.** Use `Param` types: `str`, `int`, `float`, `bool`, `path`, `list[str]`, `list[int]`.
- **Describe outputs precisely.** The LLM wires tools together based on output descriptions and types. Include column names for CSV outputs (e.g., "CSV with gauge_id, nse, kge columns").
- **Use standardized schemas.** Match existing conventions — `time` index for time series, `gauge_id` for station data, `month` (1-12) for climatology.
- **No side effects.** Tools receive a `config` dict, write files to `output_dir`, and return a result dict. That's it.
- **Strict validation.** The `@esmflow_tool` decorator rejects unknown parameters automatically. Don't disable this.

### Tool categories

| Category | Purpose | Directory |
|----------|---------|-----------|
| fetchers | Download external data | `tools/fetchers/` |
| loaders | Load and validate local data | `tools/loaders/` |
| matchers | Match observations to model grid | `tools/matchers/` |
| extractors | Extract time series or fields | `tools/extractors/` |
| analyzers | Compute metrics and statistics | `tools/analyzers/` |
| plotters | Generate visualizations | `tools/plotters/` |

## 2. Generating Workflows with an LLM

### What you need

1. **The system prompt** — `docs/system_prompt.txt`
2. **The tool catalog** — `tools/tool_catalog.yaml`
3. **Your task description** — a natural-language request

### How to prompt

Construct the LLM prompt by concatenating three pieces:

```
┌─────────────────────────────────────────┐
│  System message                          │
│  ┌───────────────────────────────────┐  │
│  │ docs/system_prompt.txt            │  │
│  │ (rules + YAML structure)          │  │
│  │                                   │  │
│  │ + tools/tool_catalog.yaml         │  │
│  │ (appended after "TOOL CATALOG:")  │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│  User message                            │
│  ┌───────────────────────────────────┐  │
│  │ Your task description             │  │
│  │ (natural language)                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Step 1:** Copy the system prompt from `docs/system_prompt.txt`.

**Step 2:** Append the full contents of `tools/tool_catalog.yaml` after the `TOOL CATALOG:` line.

**Step 3:** Send this as the system message. Then write your task as the user message.

### Example task descriptions

**Simple — observation summary:**
```
Load GRDC gauge metadata from ./data/sample/obs/gauge_metadata.csv.
Extract observed streamflow for gauges 3629000, 4121801, 4115200
from ./data/sample/obs/streamflow/ for years 1985-1989.
Compute summary statistics (mean, std, min, max) for each gauge,
including river names from the metadata.
Save outputs to ./output/obs_summary.
```

**Medium — model vs observation validation:**
```
Compare E3SM MOSART monthly discharge against GRDC observations.
- Case: v3.LR.historical_0091
- Data directory: ./data/sample/e3sm
- Observation metadata: ./data/sample/obs/gauge_metadata.csv
- Observation streamflow: ./data/sample/obs/streamflow/
- Years: 1985-1989
- Variable: RIVER_DISCHARGE_OVER_LAND_LIQ

Steps: load metadata, match gauges to MOSART grid, extract simulated
and observed time series, compute metrics (NSE, KGE, PBIAS),
plot time series comparison and metric map.
Save to ./output/streamflow_validation.
```

**Complex — gridded field comparison with ILAMB observations:**
```
Compare E3SM ELM evapotranspiration against MODIS observations.
- Fetch MODIS ET from ILAMB (variable: evspsbl, dataset: MODIS)
- Extract E3SM ET as sum of QVEGE+QVEGT+QSOIL from ELM output
  - Case: v3.LR.historical_0091, data_dir: ./data/sample/e3sm, years: [2001]
- Extract MODIS ET field (variable: et) from the fetched file
- Compute spatial bias (model minus obs)
- Compute zonal statistics by latitude band
- Plot bias map and model/obs maps
Save to ./output/et_benchmark.
```

### Tips for good task descriptions

- **Be explicit about file paths.** The LLM needs to know where your data lives.
- **Specify years.** E.g., `years: [2000, 2001, 2002]` or `years 2000-2005`.
- **Name the output directory.** Tell the LLM where to save results.
- **Distinguish fetch vs extract variables.** For ILAMB data, the CMIP variable name used for fetching (e.g., `evspsbl`) may differ from the NetCDF variable inside the file (e.g., `et`). State both explicitly.
- **Mention the model component.** E3SM has multiple components (mosart, elm, eam) — tell the LLM which one.
- **List the analysis steps.** A numbered list of what you want helps the LLM plan the workflow.

### Using any LLM

This works with any LLM — ChatGPT, Claude, Gemini, Llama, Mistral, etc. The system prompt + catalog is typically ~6,000 tokens, well within any model's context window.

**With a chat interface (ChatGPT, Claude.ai):**
1. Paste the system prompt + catalog as the first message (or system/custom instructions)
2. Type your task description as the next message
3. Copy the YAML response and save it as `my_workflow.yaml`

**With an API:**
```python
messages = [
    {"role": "system", "content": system_prompt + tool_catalog},
    {"role": "user", "content": task_description},
]
response = client.chat.completions.create(model="...", messages=messages)
workflow_yaml = response.choices[0].message.content
```

## 3. Validating and Running Workflows

### Dry run (no data needed)

```bash
python run_workflow.py my_workflow.yaml --dry-run
```

This checks:
- Valid YAML syntax
- All tool names exist in the registry
- Required parameters are present
- Output references (`${step.outputs.key}`) are valid
- No unknown parameters

### Run

```bash
python run_workflow.py my_workflow.yaml
```

### Options

```
--dry-run, -n       Validate without executing
--verbose, -v       Print detailed progress
--start-from, -s    Jump to a specific step (earlier steps assumed complete)
--reuse, -r         Reuse existing intermediate files (skip steps whose outputs exist)
```

### Iterating

If the LLM-generated workflow fails validation or execution:
1. Read the error message — it usually points to a specific step and parameter
2. Fix the YAML manually or re-prompt the LLM with the error
3. Use `--reuse` to skip already-completed steps when re-running

## 4. Reference Workflows

The `reference_workflows/` directory contains validated, human-reviewed workflows that demonstrate correct tool wiring for common analysis patterns:

| Workflow | Tools used | Pattern |
|----------|-----------|---------|
| `task01_reference.yaml` | 3 | Load obs → summary stats |
| `task02_reference.yaml` | 6-7 | Model-obs matching → metrics → map |
| `task03_reference.yaml` | 5-6 | ILAMB fetch → gridded extraction → spatial bias |
| `task04_reference.yaml` | 5-6 | Daily discharge → FDC metrics → scatter |
| `task05_reference.yaml` | 5-6 | Composite variables → spatial comparison |
| `task06_reference.yaml` | 7-8 | Multi-variable → ILAMB + model → bias + zonal |
| `task07_reference.yaml` | 12+ | Multi-component integrated diagnostic |

These serve as ground truth for the expected output of a well-prompted LLM. Study them to understand the wiring patterns between tools.

## 5. Data Schemas

Tools communicate through typed files with standardized column names. The catalog declares exact schemas so an LLM can wire tools together without guessing.

### CSV schemas

| Schema | Index | Columns | Produced by | Consumed by |
|--------|-------|---------|-------------|-------------|
| gauge_metadata | row | `gauge_id`, `lat`, `lon`, `area_km2`, `river_name` | loaders | matchers, extractors, plotters |
| matched_gauges | row | `gauge_id`, `lat`, `lon`, `model_lat`, `model_lon`, `lat_idx`, `lon_idx` | matchers | extractors |
| timeseries | `time` | one column per `gauge_id` | extractors | analyzers, plotters |
| climatology | `month` (1-12) | one column per `gauge_id` | compute_climatology | plotters |
| metrics | row | `gauge_id`, `nse`, `kge`, `pbias`, `rmse`, `correlation`, `n_valid` | compute_metrics | plotters |
| summary_stats | row | `column_name`, `mean`, `std`, `min`, `max` | compute_summary_stats | — |

### NetCDF schemas

| Schema | Structure | Produced by | Consumed by |
|--------|-----------|-------------|-------------|
| field_file | 2D lat-lon | extract_gridded_field | compute_spatial_bias, plot_gridded_map |
| bias_file | 2D lat-lon | compute_spatial_bias | plot_gridded_map |
| data_file | raw obs | fetch_ilamb_data | extract_gridded_field |

### Data flow

```
fetchers → loaders → matchers → extractors → analyzers → plotters
```

Each arrow represents a typed file handoff. The catalog describes both ends of every connection, so an LLM can trace the data flow from source to plot.
