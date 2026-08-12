# AGENTS.md

ESMFlow: an LLM planner/executor that composes and runs Earth System Model (ESM)
analysis workflows from a validated tool catalog, served via a `pydantic-ai` chat
web app.

## Setup & run

- Python 3.10 venv already at `.venv` (repo uses `/usr/bin/python` 3.10, not the
  system default `python3` which may be newer — use `.venv/bin/python`).
- Secrets/config live in `.env` (gitignored): `AI_INCUBATOR_KEY` (PNNL AI Incubator
  "Depot" gateway key) and `WEB_AGENT_MODE` (`planner` or `planner_executor`).
- `data/` and `/output` are gitignored and not present in a fresh clone. Sample
  E3SM/obs data (`./data/sample/e3sm`, `./data/sample/obs`) must exist locally
  before running workflows or evals — there is no fetch step for it.
- Run the web app: `uvicorn app:app --env-file .env --host 127.0.0.1 --port 7932`.
- `WEB_AGENT_MODE=planner` serves the interactive multistep planner (chat only,
  never executes); `planner_executor` also lets the agent run the workflow and
  render plots inline.

## Architecture (non-obvious wiring)

- **One gateway, native protocols per vendor.** `common/config.py` points the PNNL
  Depot gateway's single base URL at each vendor's native `Model`/`Provider` pair —
  `OpenAIChatModel`/`OpenAIProvider` for GPT, `AnthropicModel`/`AnthropicProvider`
  for Claude, `GoogleModel`/`GoogleProvider` for Gemini/Gemma — since the gateway
  speaks all three wire protocols (chat completions, Anthropic messages, Google
  generateContent) at that same host. Model IDs must exactly match the gateway's
  `/v1/models` listing, not the vendor's public model names.
- **Two independent planning stacks** share the same subagents and tools:
  - One-shot (`agents/planner/oneshot_planner*.py`): must return a structured
    `Workflow` in a single pass, never asks clarifying questions. Used
    programmatically by evals (`plan_workflow_one_shot`).
  - Multistep (`agents/planner/multistep_planner*.py`): conversational, may ask
    clarifying questions and return plain text instead of a `Workflow`. Used by
    the web app (`app.py`).
  Both delegate to the same category subagents in `agents/domain/*.py` (data
  discovery, extraction, diagnostics, water-cycle synthesis, visualization),
  each constrained to a `Literal[...]` tool subset defined on a `Step` subclass
  in `common/workflow.py` — Pydantic enforces the tool allow-list per category,
  not the prompt.
- Each domain module (`agents/domain/{data_discovery,extraction,diagnostics,
  water_cycle,visualization}.py`) is a thin wrapper around
  `agents/domain/_factory.py::make_domain_subagent(step_type, subagent_name,
  result_label, tool_name, description)`, which builds the `Agent` (with
  `output_type=list[step_type]`) and its `call_*` tool function. Add a new
  category by calling the factory, not by copy-pasting a module.
- **The tool catalog (`tools/tool_catalog.yaml`) is generated, not hand-edited.**
  Source of truth is the `@esmflow_tool` decorator + `ToolSpec`/`Param` in
  `tools/core/base.py`, attached to each tool module under
  `tools/{fetchers,loaders,matchers,extractors,analyzers,plotters}/`. Regenerate
  after adding/changing a tool: `python tools/generate_catalog.py --overwrite`
  (paths are resolved relative to the script itself, so it works from any CWD).
  `common/config.load_prompt()` inlines this YAML into every agent's system
  prompt, so a stale catalog silently desyncs prompts from actual tool behavior.
  `common.workflow_validation.load_raw_catalog()` loads the generated catalog
  from disk once per resolved path, then reuses that dict for the rest of the
  process (a lazy singleton, not an LRU/TTL cache; nothing is evicted or
  considered stale). Restart long-lived processes after regenerating
  `tool_catalog.yaml`.
- **Workflow execution is dynamic-import based and Python-API only**
  (`common/workflow_runner.py`): there is no CLI entrypoint (no `run_workflow.py`,
  no argparse) — call `run_workflow_definition()`/`run_workflow_file()` directly
  from Python (as `agents/planner/oneshot_planner_executor.py` and the evals do).
  Tools are loaded by filename lookup in the catalog/category dirs, params are
  resolved via `${step_id.outputs.key}` / `${settings.key}` string interpolation
  against a runtime `context` dict, and declared output filenames are used to
  rename whatever file the tool actually returned. A step with no declared
  `outputs` is always treated as not-yet-done (relevant to the `start_from`/
  `reuse` kwargs on those functions, which skip/resume previously-run steps).
- Planner/executor tools (`check_completeness`, `run_workflow_validation` in
  `agents/planner/planner_tools.py`) must both pass before `execute_planned_workflow`
  will run — the agent is instructed to call these itself; don't assume the
  runner alone will reject an invalid workflow (it does, via `validate_workflow`,
  but the friendlier path is via the planner tools).
- Generated plot files are only shown inline in the web UI if they resolve under
  `output/` relative to CWD (see `_plot_url` in `oneshot_planner_executor.py`) —
  writing outputs outside `Settings.output_dir`/`output/` breaks inline rendering.

## Conventions / gotchas

- Never emit placeholder values (`UNKNOWN`, `<UNKNOWN>`, `TBD`, `N/A`, `""`) for
  required tool params — this is enforced by convention/prompt, not by Pydantic;
  `check_completeness` in `planner_tools.py` is what actually catches it.
- Don't set a plot `units` param to a unit the data isn't already in — no tool
  converts units; `units` only relabels the axis (see `system_prompt.md`).
- `tools/core/base.ToolSpec.parse_config` is strict: unknown params raise
  `ValueError` unless prefixed `output_` or named `output_dir`.
- `list[int]` params accept `"2000-2005"` range syntax or comma lists; a bare
  numeric string like `"2000"` is a common LLM mistake the validator flags.
- Tool modules use `logger = logging.getLogger(__name__)`; console logging only
  enables the `common` and `esmflow` namespaces. `workflow_runner.load_tool()`
  loads tools as `esmflow.tools.<name>`, so their INFO logs are visible there,
  but directly imported `tools.*` modules remain quiet. A standalone tool CLI
  that needs INFO logs must configure the `tools` namespace or load tools via
  `load_tool()`.
- No lint/format/typecheck/pre-commit/CI config exists in this repo (no
  ruff/black/mypy/pyright config, no `.github/workflows`). Don't assume or
  invent a check command; `pytest` is the only automated verification.
- `system_prompt.md` is load-bearing — `common/config.load_prompt()` reads it
  and concatenates it with the generated tool catalog into every subagent's
  system prompt. `tool_subagent_mapping.md` is a human-facing design doc only;
  it isn't loaded by any code path and can drift from `common/workflow.py`'s
  actual `Literal[...]` tool lists without breaking anything.

## Testing / evals

- Run deterministic unit and tool tests with `.venv/bin/python -m pytest`.
  The suite lives under `tests/`, uses no live LLM calls, and requires
  `requirements-test.txt` in addition to application dependencies. Tests marked
  `sampledata` use the optional local `data/sample` dataset and reference outputs;
  they skip automatically when either dataset is unavailable.
- `pydantic_evals` datasets under `evals/` remain the separate live-LLM evaluation
  harnesses for planner behavior and end-to-end workflow quality.
- Real eval entrypoints (run from the repository root):
  - `python -m evals.validate_workflow_eval` — checks the one-shot planner
    produces catalog-valid workflows.
  - `python evals/workflow_execution_numerical_tolerance_eval.py` — plans,
    executes, and numerically grades (rtol=1e-12) full workflows per
    (model, task) against references in `evals/reference_workflows/results/`.
    Requires `./data/sample` to exist locally.
- `requirements-eval.txt` (just `pydantic-evals`) is separate from
  `requirements.txt`; install both to run evals.
