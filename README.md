# Local development instructions

Start the webapp by activating the virtual environment and then running `uvicorn app:app --env-file .env --host 127.0.0.1 --port 7932`

Set `WEB_AGENT_MODE` in `.env` to choose which chat agent the web app serves:

- `WEB_AGENT_MODE=planner` for the interactive multistep planner
- `WEB_AGENT_MODE=planner_executor` for the interactive multistep planner-executor
- `WEB_AGENT_MODE=onboarding` for the onboarding agent, which adds your own
  Python code to ESMFlow as new tools (see [Onboarding your own code](#onboarding-your-own-code))

## Onboarding your own code

ESMFlow can wrap your existing analysis functions as first-class workflow tools,
so the planner can compose them with the built-in ones. Your code stays yours:
plain `pandas`/`xarray`/`matplotlib` functions with no ESMFlow imports. The
onboarding system reads the function's signature and docstring, generates a thin
*adapter* under `tools/<category>/`, records it in `extensions/registry.yaml`,
regenerates the tool catalog, and verifies the result.

You don't have to know about any of this in advance. If you are planning an
analysis and no built-in tool covers a step, just ask the planner in the normal
chat. It can explain what onboarding requires and, given a path to your `.py`
file, show you the exact tool, parameters and workflow step your function would
become — all without touching the file. When you say to go ahead, it hands that
turn to an onboarding specialist in the *same* session, which walks you through
what it inferred and writes the tool once you approve. No mode switch and no
restart are needed to register.

### See the demonstration first

If you'd rather be shown than told, ask the chat:

- *"What format does my code need to be in, and where do I put it?"* — you get the
  format rules, a minimal template, and a map of which files you write versus
  which ones onboarding generates for you.
- *"Show me the demonstration running end to end."* — the chat registers the
  bundled exemplar `examples/user_code/demo_growing_degree_days.py` as a real
  tool, runs it on synthesized sample data through the real workflow runner,
  reports the numbers it produced, and then **un-registers it again**.

The demonstration is deliberately self-cleaning, so it can be shown to the next
user unchanged and it never leaves a `compute_growing_degree_days` tool in your
catalog. The exemplar source file stays in place as teaching material — it is the
one example in `examples/user_code/` that is intentionally *not* onboarded.

Headless equivalent, for CI or a quick look without spending tokens:

```bash
python -m onboarding.cli demo status   # is anything left registered?
python -m onboarding.cli demo all      # register -> run -> offboard, then report
python -m onboarding.cli demo end      # cleanup, if a run was interrupted
```

`demo end` is safe to run at any time, including twice. It refuses to touch a
capability of the same name that onboarding did not generate itself.

Other ways in:

- **CLI**: `python -m onboarding.cli <subcommand>` (see below) — for scripted or
  bulk registration.
- **Dedicated chat session**: set `WEB_AGENT_MODE=onboarding` to talk to the
  onboarding agent on its own, with no planner in the way. Optional; the planner
  chat above can do the same registration.

```bash
# See what would be inferred, without writing anything.
python -m onboarding.cli scan examples/user_code/snow_metrics.py

# Register one function as a tool in an existing planner category.
python -m onboarding.cli register examples/user_code/snow_metrics.py \
    --function compute_snow_season_metrics --subagent diagnostics

# Register into a brand-new planner category (a new subagent).
python -m onboarding.cli register examples/user_code/snow_metrics.py \
    --all --new-subagent snow_hydrology \
    --subagent-description "Snowpack timing and melt diagnostics"

python -m onboarding.cli list              # show the registry
python -m onboarding.cli verify            # re-check every registered capability
python -m onboarding.cli remove compute_snow_season_metrics
```

Notes:

- **Restart the web app / MCP server before *planning with* a newly registered
  tool.** Registering is permanent and takes effect on disk immediately, but a
  running process fixed its plannable tool list when it started: the workflow
  step classes (and the planner's output schema) are built at import time, so
  the new tool can't appear in a plan until the process restarts.
- **Write good docstrings and type hints.** They are the only thing the scanner
  has to work from: annotations become parameter types, defaults become optional
  params, and the `Args:` section becomes the parameter descriptions the planner
  reads when deciding whether to use your tool.
- **Return a dict for multiple outputs**, e.g.
  `return {"metrics_file": df, "plot_file": fig, "n_events": 12}`. DataFrames are
  written as CSV, xarray objects as NetCDF, and matplotlib figures as PNG; scalars
  are passed back to the planner as values. A single return value works too.
- Generated adapters are marked as generated — edit your original function and
  re-register with `--overwrite` rather than editing the adapter.
- `examples/user_code/` contains working examples to copy from, including the
  fully annotated `demo_growing_degree_days.py` used by the demonstration above.
- Registered tools are permanent — they live in the catalog and every later chat
  session can plan with them, with no further setup.

## Running the MCP server

Build the MCP server image from the repository root:

```bash
docker build -t esmflow-mcp .
```

Create the local directories that the container will access:

```bash
mkdir -p data output
```

The container runs as a non-root user with UID 1000. If your host user's UID
is not 1000 (check with `id -u`), run `chmod o+rwx data output` so the container
can write to these bind-mounted directories; the directories contain only local
datasets and generated outputs.

Run the MCP server as a detached HTTP service, passing the same `.env`
configuration used by the web app and mounting local data and output directories:

```bash
docker run -d --name esmflow-mcp \
  --env-file .env \
  -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  esmflow-mcp
```

The `.env` file must include `AI_INCUBATOR_KEY`. The server listens at
`http://127.0.0.1:8000/mcp` and exposes `plan_workflow`, `validate_workflow`,
`execute_workflow`, and `plan_and_execute_workflow`. For separate planning and
execution, pass the complete workflow returned by `plan_workflow` to the next
tool call; the server does not retain workflow state between requests.

The port is published only on the loopback interface because the server has no
authentication. Do not change `-p 127.0.0.1:8000:8000` to a network-accessible
port binding without adding authentication.

Inspect the running server with `docker logs esmflow-mcp`. Stop and remove it
with:

```bash
docker stop esmflow-mcp && docker rm esmflow-mcp
```

After local source changes, including regenerated `tools/tool_catalog.yaml`,
replace the running container before starting the rebuilt image:

```bash
docker stop esmflow-mcp && docker rm esmflow-mcp
docker build -t esmflow-mcp .
```

Then run the `docker run` command above.

For a local MCP client, configure the HTTP endpoint:

```json
{
  "mcpServers": {
    "esmflow": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Here is a sample query you can you to interact with the chatbot

```
Create a workflow to:

1. Extract global QRUNOFF from ELM for years 1985-1989 using the sample.v3.LR.historical case from ./data/e3sm. 
2. Compute the climatological area-weighted global mean
3. Produce a map visualization with statistics overlaid.

Write the workflow as yaml to the output folder
```

## Running the benchmark

From the repository root, run one of:

- `python benchmark/run_benchmark.py --mode protocol --models "GPT 5.4" --tasks task_01_obs_summary --runs 1` to plan and execute workflows using v2's supervisor/planner architecture. This requires the local sample data.
- `python benchmark/run_benchmark.py --mode single_agent --models "GPT 5.4" --tasks task_01_obs_summary --runs 1` to plan and execute workflows using a single, undelegated LLM call given the full tool catalog -- the baseline arm, reproducing v1's original single-agent architecture against v2's own tool catalog and data.

Both arms read the same task prompts, execute identically via `run_workflow_definition()`, and write a `pydantic_evals` `EvaluationReport` to `benchmark/results/{mode}_report.json`, graded inline by the shared `StructuralGrade` evaluator in `benchmark/grading.py`. See `AGENTS.md`'s "Benchmark" section for the full pipeline (manual review queue, self-debug retries).
