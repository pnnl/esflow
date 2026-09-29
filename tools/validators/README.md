# ESFlow Data Validation

Data validation system for ESFlow. Automatically validates user data against requirements derived from the tool catalog using atomic validation primitives and LLM-powered plan generation (with rule-based fallback).

## Quick Start

```python
from agents.data_preflight import preflight_check

# Pre-workflow check
report = preflight_check(
    data_dir='data',
    catalog_file='tool_catalog.yaml',
    case_name='sample.v3.LR.historical',
    years=[1985, 1986, 1987, 1988, 1989]
)

print(report.summary)        # User-friendly text
print(report.feasible)       # Set of tool names that can run
print(report.gauge_ids)      # Recommended gauge IDs
```

## Components

### primitives.py
12 atomic validation checks: file/dir existence, CSV structure, GeoJSON properties, NetCDF variables, E3SM output.

### planning_agent.py
LLM-powered (gpt-4o-mini/claude-3.5-haiku) or rule-based plan generation. Plans cached by catalog hash.

### validate_workflow_data.py
ESFlow tool registered in `tool_catalog.yaml`. Runs as workflow step 1. Outputs: validation_report.csv, validation_summary.txt, feasible_tools.csv

## Integration

**Preflight check (before workflow composition):**
```python
from agents.data_preflight import preflight_check
report = preflight_check(data_dir, catalog_file, case_name, years)
```

**Validation tool (workflow step 1):**
steps:
  - id: validate_data
    tool: validate_workflow_data
    params:
      data_dir: data
      catalog_file: tools/tool_catalog.yaml
      case_name: sample.v3.LR.historical
      years: [1985, 1986, 1987, 1988, 1989]
    outputs:
      report_file: validation_report.csv
      summary_file: validation_summary.txt
      feasible_tools_file: feasible_tools.csv

**Direct API:**
```python
from tools.validators import primitives, planning_agent
plan = planning_agent.get_plan(catalog, context)
for check in plan:
    result = primitives.execute(check)
```

## Structure

```
tools/validators/
├── __init__.py
├── primitives.py              # 12 atomic checks
├── planning_agent.py          # LLM + fallback plan generation
└── validate_workflow_data.py   # ESFlow tool wrapper
```

## Features

- ✅ Automatic validation plan generation from catalog
- ✅ Plan caching by catalog hash (no redundant LLM calls)
- ✅ LLM-powered or rule-based fallback (no API key required)
- ✅ Cross-file consistency checking (gauge_id matching)
- ✅ E3SM model output detection
- ✅ Tool feasibility analysis (ready/partial/blocked)
- ✅ Actionable error/warning hints

## Dependencies

Built-in: pandas, yaml, pathlib, json, re
Optional: openai, anthropic (for LLM-powered plans)
