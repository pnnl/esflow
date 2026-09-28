# AGENTS.md

ESMFlow: an LLM planner/executor that composes and runs Earth System Model (ESM)
analysis workflows from a validated tool catalog, served via a `pydantic-ai` chat
web app.

## Setup & run

- Python 3.10 venv already at `.venv` (repo uses `/usr/bin/python` 3.10, not the
  system default `python3` which may be newer — use `.venv/bin/python`).
- Secrets/config live in `.env` (gitignored): `AI_INCUBATOR_KEY` (PNNL AI Incubator
  "Depot" gateway key) and `WEB_AGENT_MODE` (`planner`, `planner_executor` or
  `onboarding`).
- `data/` and `/output` are gitignored and not present in a fresh clone. Sample
  E3SM/obs data (`./data/sample/e3sm`, `./data/sample/obs`) must exist locally
  before running workflows or evals — there is no fetch step for it.
- Run the web app: `uvicorn app:app --env-file .env --host 127.0.0.1 --port 7932`.
- `WEB_AGENT_MODE=planner` serves the interactive multistep planner (chat only,
  never executes); `planner_executor` also lets the agent run the workflow and
  render plots inline; `onboarding` serves the onboarding agent
  (`agents/onboarding/onboarding_agent.py`) instead of a planner — it does not
  plan or execute workflows, it registers the user's own Python as new tools
  (see "Onboarding" below).

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
    programmatically by `benchmark/` and `mcp_server.py` (`plan_workflow_one_shot`).
  - Multistep (`agents/planner/multistep_planner*.py`): conversational, may ask
    clarifying questions and return plain text instead of a `Workflow`. Used by
    the web app (`app.py`).
  Both delegate to the same category subagents in `agents/domain/*.py` (data
  discovery, extraction, diagnostics, water-cycle synthesis, visualization),
  each constrained to a `Literal[...]` tool subset defined on a `Step` subclass
  in `common/workflow.py` — Pydantic enforces the tool allow-list per category,
  not the prompt.
- **The model's final structured-output turn can silently discard the caller's
  `Settings`.** All three planner entrypoints (`plan_workflow_one_shot` in
  `oneshot_planner.py`; `plan_with_planner` in `oneshot_planner_executor.py`;
  `plan_with_multistep_planner` in `multistep_planner_executor.py`) build a
  fresh `Workflow` from `result.output`/`ctx.deps.workflow = result.output`,
  and the model is free to rewrite `settings` from the task text (e.g. a
  literal "save to ./output/foo" instruction) instead of echoing the
  `Settings` object it was actually given — confirmed live: a v2 planner run
  produced numerically correct output but wrote it to the model's own
  guessed path, not the caller's `output_dir`, and got mis-graded as a crash
  by the benchmark harness as a result. All three entrypoints now force the
  caller's/session's `settings` back onto the model's output before
  returning/updating state (`workflow.settings = settings` /
  `result.output.settings = ctx.deps.workflow.settings`); a fourth
  entrypoint needs the same fix if one is ever added.
- **`mcp_server.py` is a third, stateless entrypoint** for MCP clients. It
  exposes plan, validation, execution, and combined plan-and-execute tools over
  FastMCP; Docker configures HTTP transport, while direct execution defaults to
  stdio. Unlike `app.py`'s process-global `WorkflowState`, each MCP call
  receives or returns the complete `Workflow`; it reuses the same plain-argument
  planning and runner APIs used by `benchmark/`. The "check
  completeness -> validate -> write plan -> run" gate sequence is independently
  implemented in `execute_planned_workflow`
  (`agents/planner/oneshot_planner_executor.py`) and `execute_workflow`
  (`mcp_server.py`); both use the shared validation helpers but own their
  ordering. Update both call sites if this sequence changes; extract a shared
  helper if a third entrypoint needs the same sequence.
- **`tool_mcp/server.py` is a fourth, tool-only MCP entrypoint** with no
  planning/execution/workflow concept at all -- one MCP tool per esmflow tool,
  called directly. It auto-discovers every `ToolSpec` registered via
  `@esmflow_tool` (the same `TOOL_REGISTRY` `tools/generate_catalog.py` reads)
  and synthesizes a real, introspectable function per tool with
  `tool_mcp/_signature.py`, so adding a tool under `tools/<category>/` needs no
  edits in `tool_mcp/` -- unlike the separate `esflow-tool-mcp` sibling repo,
  which hand-ports each tool and requires two edits per tool in its own
  `server.py`. Has its own `requirements.txt`, `Dockerfile` (built from the
  `esflow-v2/` repo root: `docker build -f tool_mcp/Dockerfile .`, since it
  needs the sibling `tools/` directory), and standalone test suite
  (`python -m pytest tool_mcp/tests`, not part of the root `pytest.ini`
  testpaths).
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
  from Python (as `agents/planner/oneshot_planner_executor.py` and `benchmark/` do).
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

## Onboarding (adding user code as capabilities/agents)

- **Adding a tool used to require three coupled edits**; the third is now
  data-driven. (a) a module under `tools/<category>/` with `@esmflow_tool`,
  (b) a regenerated `tools/tool_catalog.yaml`, and (c) the per-category
  `Literal[...]` tool allow-list. (c) now lives in
  `common/tool_categories.py::BUILTIN_CATEGORIES` (frozen `CategorySpec`s), and
  `common/workflow.py::_build_step_class()` builds each `Step` subclass with
  `pydantic.create_model` at import time. Don't hand-write new `Step` classes.
- **`extensions/registry.yaml` is the user-extension half of that source of
  truth.** `category_specs()` merges `BUILTIN_CATEGORIES` first, then registry
  subagents — a registry entry can never shadow a builtin category. New
  user-defined subagents become real planner categories (with a generated
  `SnowHydrologyStep` / `call_snow_hydrology` pair via
  `agents/domain/extensions.py`) purely from that YAML.
- **The onboarding pipeline never imports the user's module to inspect it.**
  `onboarding/introspect.py` is pure `ast`: it reads signatures, annotations,
  defaults and docstrings to propose a `CapabilityDraft`
  (`onboarding/models.py`). Only `onboarding/verify.py` and the generated
  adapter actually import the code. This is deliberate — scanning must be safe
  on code with import-time side effects.
- **Generated adapters are the bridge, not rewrites of the user's code.**
  `onboarding/scaffold.py` renders a thin `tools/<category>/<tool_name>.py`
  carrying `GENERATED_MARKER`; it calls `tools/core/adapters.py`'s
  `call_user_function()` + `materialize_result()`, which forward only the
  declared inputs the user's signature actually accepts, then turn the return
  value into files (DataFrame→csv, xarray→netcdf, Figure→png) plus scalars.
  `remove_adapter()` refuses to delete a file lacking that marker, so
  hand-written tools can't be clobbered by `onboarding remove`.
- **Generated adapters must use the repo's `sys.path` import convention**
  (`sys.path.insert(0, ...parent.parent)` then `from core.base import ...`), not
  `from tools.core.base import ...`. `tools/generate_catalog.py` reads
  `core.base.TOOL_REGISTRY`; importing via the `tools` package creates a
  *second* module instance with its own registry, so the tool registers into a
  dict the catalog generator never sees and silently vanishes from the catalog.
  `tests/onboarding/test_scaffold.py` asserts `"from tools.core" not in source`.
- **`ToolSpec.outputs` must use the dict form**
  (`{'stats_file': {'type': 'csv', 'description': ...}}`). `esmflow_tool`'s
  wrapper calls `out_spec.get('type', '')`, so a plain-string value crashes at
  runtime; `adapters.output_type_of()` normalizes both shapes defensively.
- **`register_capability()` is all-or-nothing.** It writes the adapter, records
  the registry entry, then shells out to `generate_catalog.py`; if that
  subprocess fails it restores the previous adapter source *and* registry text
  before raising. Don't add a write between those steps without extending the
  rollback.
- CLI (all subcommands under `python -m onboarding.cli`, which needs
  `AI_INCUBATOR_KEY` set like everything else since `common/config.py` validates
  it at import time): `scan <source> [--function] [--show-adapter]`,
  `register <source> [--function|--all] [--tool-name] [--subagent |
  --new-subagent] [--overwrite] [--no-catalog] [--no-verify]`, `list`,
  `verify [tool_name]`, `remove <tool_name> [--keep-adapter]`.
- **A newly registered tool cannot be *planned with* until the process
  restarts.** The catalog itself is cheap to reload (`load_raw_catalog()` is a
  lazy singleton and `_catalog_singletons.clear()` resets it — `tests/conftest.py`
  does exactly that per test). The real blocker is *import-time class
  construction*: `common/workflow.py` runs `CATEGORY_SPECS = category_specs()`
  and `STEP_CLASSES = {name: _build_step_class(spec) ...}` at module import,
  each step class pins `tool: Literal[tuple(spec.tools)]`, and the planner's
  `output_type=[Workflow, str]` schema is derived from those classes when the
  `Agent` is constructed. Rebuilding a step class produces a *new* object that
  already-imported consumers never see. A brand-new *category* is a harder stop
  still: `agents/domain/extensions.py` builds `EXTENSION_AGENTS` /
  `EXTENSION_PLANNER_TOOLS` at import, and tools cannot be added to an
  already-constructed `Agent`. So: registration is permanent and immediately
  visible on disk, but restart the web app / MCP server before putting the new
  tool in a workflow.
- `examples/user_code/` holds exemplar "user code" modules (drought indices,
  snow metrics, gridded trends, a month×year heatmap, flow exceedance) that are
  plain pandas/xarray/matplotlib functions with no ESMFlow imports. They are
  already onboarded, so `tools/analyzers/compute_{standardized_anomaly_index,
  snow_season_metrics,gridded_trend,flow_exceedance_thresholds}.py` and
  `tools/plotters/plot_month_year_heatmap.py` are *generated* files — edit the
  `examples/user_code/` source and re-register, don't patch the adapter.
- **`examples/user_code/demo_growing_degree_days.py` is the one exemplar that
  must stay *un*-onboarded.** It is the teaching material for the self-cleaning
  demonstration (below), so its `# FORMAT:` comments and its explicit
  `int()`/`str()`/`float()` coercions at the `return` boundary are load-bearing:
  strip the casts and `onboarding/introspect.py` falls back to guessing, and the
  demo starts teaching the wrong types.
  `tests/onboarding/test_demo.py::test_the_exemplar_output_types_are_inferred_exactly`
  pins them.
- **The onboarding demonstration is register → run → offboard, and it must
  leave nothing behind.** `onboarding/demo.py` is the engine:
  `demo_status()` reads on-disk evidence only (registry entry, generated
  adapter, catalog entry) and refuses to touch a same-named capability it did
  not generate (`foreign_capability`); `start_demo()` registers the exemplar
  with `overwrite=True` so any partial state self-repairs; `run_demo()`
  synthesizes a deterministic 180-day temperature CSV and executes a real
  one-step workflow **in a fresh subprocess** (the restart problem above means
  this process can never see the tool it just registered — the subprocess *is*
  the restart, same `@@JSON@@`-on-stdout trick as `onboarding/verify.py`);
  `end_demo()` un-registers, deletes the adapter, regenerates the catalog and
  then *re-checks*, reporting `leftovers` rather than a false success. The
  exemplar source under `examples/user_code/` is never deleted.
  - The agent-facing wrapper `demo_onboarding_walkthrough` calls `end_demo()` in
    a `finally`, so a failed run still offboards; if cleanup itself fails it
    returns a `WARNING` plus the `python -m onboarding.cli demo end` command.
    Don't refactor that `try/finally` away.
  - **The committed repo must always be demo-clean.** Four guard tests in
    `tests/onboarding/test_demo.py` fail if `compute_growing_degree_days`
    appears in `extensions/registry.yaml`, `tools/tool_catalog.yaml` or as a
    generated adapter. If you ever interrupt a demo, run
    `python -m onboarding.cli demo end` before committing.
  - Headless: `python -m onboarding.cli demo {status,start,run,end,all}`.
    `demo all` regenerates the real catalog, which is why the pytest suite stubs
    the stages instead of running it.
- **The planner has read-only onboarding advisor tools, by design.**
  `agents/planner/onboarding_advisor.py` gives the interactive planner
  `explain_onboarding`, `show_onboarding_example`, `preview_user_code_as_tool`
  and `list_onboarded_capabilities` so a user who hits a capability gap mid-analysis
  is told onboarding exists instead of getting an improvised answer. They are
  typed `RunContext[WorkflowState]` (planner deps), so unlike the onboarding
  agent's `scan_user_code` they *cannot* persist drafts — they are stateless
  wrappers over `scan_source()` / `registry_summary()` / `demo_draft()`.
  `show_onboarding_example` *shows* the exemplar and offers the live
  demonstration via `delegate_to_onboarding`; it never runs it, because the demo
  writes. All repo mutation stays
  inside the onboarding *agent* (see the delegation bullet below);
  `tests/agents/test_onboarding_advisor.py` patches `register_capability`,
  `register_subagent`, `remove_capability`, `save_registry`,
  `regenerate_catalog`, `write_adapter` and `remove_adapter` with fail-fast
  stubs to keep it that way.
- **The planner onboards *in session* by delegating, not by switching modes.**
  `agents/planner/onboarding_bridge.py` exposes one tool,
  `delegate_to_onboarding(ctx, request)`, which hands the turn to
  `onboarding_agent.run(...)` with `model=ctx.model` and `usage=ctx.usage` so the
  UI's model picker applies and delegated tokens count against the parent run's
  limits. `WEB_AGENT_MODE=onboarding` was never a technical requirement — it is
  just an `if/elif` in `app.py` — and now only exists for a dedicated,
  planner-free onboarding session. Details that matter:
  - The delegated agent needs `OnboardingState`, not `WorkflowState`, so the
    bridge parks an `OnboardingSession` (state + `message_history`) on
    `WorkflowState.onboarding`. That field is deliberately typed `Any`:
    importing `agents.onboarding` from `common/__init__.py` would be circular.
  - Continuity is manual. The bridge persists `result.all_messages()` back onto
    the session, so drafts survive across planner turns; a naive one-shot `run`
    would silently lose them. Route the user's onboarding follow-ups back
    through the same tool.
  - The write boundary is machine-checked, not commented:
    `tests/agents/test_onboarding_bridge.py` AST-walks the bridge module and
    fails if it ever imports one of the seven write functions.
  - **Never run the onboarding agent under a plain `TestModel` in a test.**
    `TestModel` calls *every* tool it is offered, which here includes
    `register_draft` — it will write adapters into `tools/` from a unit test.
    Use `TestModel(call_tools=[])`, or `FunctionModel` to script specific calls.
- **Advisor and delegation tools attach to `PLANNER_TOOLS` and
  `MULTISTEP_PLANNER_EXECUTOR_TOOLS` only, never `ONESHOT_PLANNER_TOOLS`** —
  the latter is what `oneshot_planner.py` gives the benchmark and the MCP
  server, and its tool set must stay fixed for scores to remain comparable —
  and, for `delegate_to_onboarding`, must stay non-mutating. Tests assert the
  advisor sets are disjoint and that `delegate_to_onboarding` is absent from
  `ONESHOT_PLANNER_TOOLS`.
- Only file-valued outputs (`csv`/`png`/`netcdf`) belong in a step's `outputs:`
  mapping, and each must be a filename *with* an extension (the runner renames
  the tool's artifact to it inside `Settings.output_dir`). Scalar outputs
  (`int`/`float`/`str`/`dict`) are **not** listed there — they stay in
  `context[step_id]['result']`. A later step consumes a file with
  `${<step_id>.outputs.<output_key>}`; a bare `${<step_id>}` fails to resolve.
  `render_workflow_snippet()` emits a correct example per capability.

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
  `pytest-asyncio` (`requirements-test.txt`) is **required**, not optional —
  `pytest.ini` sets `asyncio_mode = auto`, and without the plugin every `async def`
  test fails with "async def functions are not natively supported" (32 of them)
  alongside a misleading `PytestConfigWarning: Unknown config option: asyncio_mode`.
  That is a missing-dependency symptom, not a regression.
  The suite lives under `tests/`, uses no live LLM calls, and requires
  `requirements-test.txt` in addition to application dependencies. Tests marked
  `sampledata` use the optional local `data/sample` dataset and reference outputs;
  they skip automatically when either dataset is unavailable.
- `pydantic-evals` is a hard dependency (`requirements.txt`), not a separate
  optional install — there is no standalone `evals/` package anymore. Live-LLM
  evaluation of planner behavior and end-to-end workflow quality is done
  through `benchmark/` (see below), which builds `pydantic_evals.Dataset`/
  `Case` objects graded by the shared `StructuralGrade` evaluator in
  `benchmark/grading.py`.
- Onboarding tests live under `tests/onboarding/` (`test_introspect.py`,
  `test_scaffold.py`, `test_adapters.py`, `test_registry.py`, `test_demo.py`,
  `test_committed_registry.py`) plus `tests/common/test_tool_categories.py`. They
  are hermetic: every registry test passes an explicit `path=`/`tools_root=` under
  `tmp_path` and `regenerate=False`, so they never touch the real
  `extensions/registry.yaml` or `tools/` tree. Keep it that way — a test that
  regenerates the catalog would rewrite a tracked file.
- `tests/onboarding/test_committed_registry.py` pins the *committed* registry:
  exactly five capabilities with their documented subagent routing, `subagents: []`,
  and each adapter/source file present on disk and in the generated catalog. The
  three guards in `test_demo.py` only check the demo tool name, which is why a
  stray subagent once shipped through a fully green suite. Update
  `EXPECTED_CAPABILITIES` deliberately when the shipped set changes.
- **`TestModel` calls every tool an agent exposes, including delegation tools.**
  The planner exposes `delegate_to_onboarding`, which runs the onboarding agent
  with `model=ctx.model` (see `agents/planner/onboarding_bridge.py`). So
  `planner.override(model=TestModel())` propagates `TestModel` into the onboarding
  agent, which then calls onboarding's **write** tools with synthesized dummy
  arguments — `create_subagent(name='a')` against the real
  `extensions/registry.yaml`. This actually happened: it committed an `a` /
  `AStep` / `call_a` subagent, i.e. a user-visible planner category, while the
  suite stayed green. Any test that drives a planner with `TestModel` must use the
  `no_onboarding_writes` fixture in `tests/agents/test_planner_agents.py`, which
  redirects `REGISTRY_PATH`/`CATALOG_PATH` into `tmp_path` (patching **both**
  `common.tool_categories` and the already-resolved copy in `onboarding.registry`,
  because `from ... import` binds a separate reference) and stubs
  `regenerate_catalog`. Redirecting beats raising: the tools still succeed, so the
  planner loop is exercised as in production.
- The backstop is the autouse `_protect_committed_repo_state` fixture in
  `tests/conftest.py`: it snapshots `extensions/registry.yaml` and
  `tools/tool_catalog.yaml`, then after each test both reverts and **fails** on any
  change. If a test legitimately rewrites those files as the behaviour under test,
  mark it `@pytest.mark.mutates_repo_state` and restore them itself.

## Benchmark

- `benchmark/` replicates the paper's comparison of v2's supervisor/planner
  architecture against a single-agent baseline, using the seven prompts
  under `benchmark/protocol/`. Both arms read the exact same task prompt --
  the task doesn't change, only which planning architecture receives it:
  - `--mode protocol` uses the one-shot planner
    (`agents/planner/oneshot_planner.py`), which delegates to category
    subagents and self-checks via `check_completeness`/
    `run_workflow_validation` before returning.
  - `--mode single_agent` uses `agents/planner/single_agent_planner.py`, a
    single undelegated LLM call given the full tool catalog inlined via
    `common.config.load_prompt()`, with no subagents and no self-check
    tools -- reproducing v1's original single-agent architecture
    (`call_llm` in v1's `benchmark/run_benchmark.py`) against v2's own tool
    catalog and data, instead of comparing against LLM-generated free-form
    Python.
  Both arms produce structured `Workflow` YAML and execute identically via
  `run_workflow_definition()`, unsandboxed on the host with real network
  access -- there is no Docker sandbox or free-code-generation path in this
  benchmark.
- Run a small pilot before the paid full grid, for example:
  `python benchmark/run_benchmark.py --mode protocol --models "GPT 5.4" --tasks task_01_obs_summary --runs 1`
  and `python benchmark/run_benchmark.py --mode single_agent --models "GPT 5.4" --tasks task_01_obs_summary --runs 1`.
  Each writes a `pydantic_evals.EvaluationReport` to
  `benchmark/results/{mode}_report.json`; grading happens inline during the
  run via the shared `StructuralGrade` evaluator (`benchmark/grading.py`), not
  as a separate pass over JSON files. `StructuralGrade`'s Step 2 (numerical
  reference comparison) applies identically to both arms, since both produce
  structured, predictable-filename output. Then run
  `python benchmark/make_manual_queue.py --report benchmark/results/protocol_report.json --mode protocol`
  to export cases graded `undetermined` (score 0.5) to
  `benchmark/results/manual_review_protocol.csv` for human review, and
  `python benchmark/merge_grades.py --report benchmark/results/protocol_report.json --mode protocol`
  to write the resolved `final_grade` back onto each case as a report label
  (in place, no separate `scores_final_*.json`).
- `benchmark/self_debug_crashes.py --report benchmark/results/protocol_report.json --mode protocol --max-rounds 3`
  retries only structurally crashed cases (`StructuralGrade` reason starting
  with `"crash:"`) using the original model and traceback feedback. It also
  makes live model calls. Each retry round builds a small `Dataset`/`Case`
  pair and re-grades the repaired artifact with the *same* `StructuralGrade`
  evaluator used for the main run — there is no separate resolve/grade/queue/
  merge script family duplicating that logic. Both arms share one repair code
  path (`execute_workflow()`) since both produce YAML. Writes
  `benchmark/results/self_debug_{mode}.json` (plain JSON rows, since these are
  retry-attempt metadata rather than a `pydantic_evals` report); results under
  `benchmark/results/` are intentionally ignored.
- `benchmark/run_workflow_definition` (via `common/workflow_runner.py`) swallows
  per-step tool exceptions and keeps going rather than raising, so a crash
  from a missing deliverable often has no traceback in the normal sense.
  `benchmark/datasets.py`'s `_collect_step_errors()` walks the execution context
  for any step whose result contains an `error` key and writes it to
  `run{N}.error.txt`, which `self_debug_crashes.py` then uses as real repair
  signal instead of falling back to `has_deliverable()`'s generic reason string.
- Deterministic tests for this package live under `tests/benchmark/` (no
  live LLM calls) and run as part of the normal `pytest` suite. Do **not**
  add an `__init__.py` to `tests/benchmark/` — pytest's default import mode
  would then register it as the top-level module `benchmark`, shadowing the
  real `benchmark/` package at the repo root (this bit us once).
- `benchmark/reference_workflows/results/` holds the numeric reference outputs
  `StructuralGrade`'s Step 2 (numerical comparison) checks against;
  `benchmark/reference_workflows/prompts/` holds the human-facing
  reference YAML for each task. Both moved here from the now-deleted `evals/`
  package.
- `benchmark/common.py`'s `SAMPLE_CASE_NAME = "sample.v3.LR.historical"` is
  hardcoded into every run's `Settings.case_name` in `run_planned_case()`
  (`benchmark/datasets.py`). E3SM tools (`extract_gridded_field`,
  `match_to_grid`, etc.) require `case_name` as a separate required param
  from `data_dir`, and `task_02`–`task_07`'s prompts all name it literally —
  a model may reference it via `${settings.case_name}` instead of repeating
  the literal string per step. Before the settings-preservation fix above,
  this was masked because models silently substituted their own guessed
  `case_name` into the `settings` block they returned; adding an 8th task
  against different sample data means updating `SAMPLE_CASE_NAME` too, or
  `${settings.case_name}` references will fail to resolve at execution time.
