# Full pilot benchmark runbook

Step-by-step commands to run the complete v2 benchmark grid (3 pilot
models x 7 tasks x 4 runs x 2 modes = 168 live-LLM cases) and produce final
graded reports, structured as small, independently resumable chunks.

## Why chunked instead of one script

`run_benchmark.py --mode X --runs 4` (no `--models`/`--tasks` filter) builds
one big `pydantic_evals.Dataset` covering all pilot models and tasks, and
only writes its report JSON at the very end. If that single invocation is
interrupted -- a shell/session timeout, a lost connection, anything -- you
get **no report at all** for any of it, even for cases that individually
succeeded, because nothing is written until the whole dataset finishes.

There is also no cheap way to resume a chunk *as a whole* in place:
`run_planned_case()` (`benchmark/datasets.py`) always re-plans and
re-executes a case from scratch; nothing here skips a case just because
`run{N}.yaml` already exists on disk from a previous attempt. (Individual
failed cases *can* be retried cheaply without redoing the rest of the
chunk -- see "Retrying a partial chunk failure" under Phase 1.)

Splitting the grid into one chunk per (model, mode) -- 6 chunks total --
bounds the damage from any single interruption to one chunk (roughly
30-110 minutes of live-LLM calls per chunk, based on past runs) instead of
the whole multi-hour grid, and each chunk produces its own report file
immediately on completion.

If you're running this somewhere long-running background jobs reliably
survive session boundaries (a real server/CI, not a sandboxed dev
environment), you can safely combine steps -- e.g. loop over all 6 chunks in
one script, or drop `--models`/`--output` entirely and run the whole grid
in one `run_benchmark.py` call per mode. The chunking below is a
workaround for environments where that isn't reliable, not a hard
requirement of the benchmark itself.

## Prerequisites

Run once, before Phase 1:

```bash
cd esflow-v2   # repo root; all commands below assume this as cwd

# Confirm secrets and sample data are present -- there is no fetch step for
# either; see AGENTS.md.
test -f .env && echo ".env present" || echo "MISSING .env"
test -d data/sample/e3sm && test -d data/sample/obs && echo "sample data present" || echo "MISSING data/sample"

# Sanity-check the environment with the deterministic test suite (no live
# LLM calls). Expect "135 passed, 4 deselected" -- those 4 are known
# pre-existing live-API failures unrelated to this runbook; see AGENTS.md
# "Testing / evals".
.venv/bin/python -m pytest -q \
  --deselect tests/agents/test_planner_agents.py::test_oneshot_planner_returns_a_structured_workflow_with_test_model \
  --deselect tests/agents/test_planner_agents.py::test_multistep_planner_runs_with_test_model \
  --deselect tests/agents/test_planner_agents.py::test_oneshot_executor_planning_wrapper_updates_workflow_state \
  --deselect tests/agents/test_planner_agents.py::test_multistep_executor_planning_wrapper_returns_agent_response

# Start from an empty results directory. benchmark/results/ is gitignored;
# nothing here is meant to be preserved across a fresh full run.
rm -rf benchmark/results/*
```

## Phase 1: per-(model, mode) chunks

Run each of these 6 commands **one at a time**, waiting for each to finish
and print `wrote benchmark/results/...` before starting the next. Each
covers all 7 tasks x 4 runs (28 cases) for one model in one mode.

Expected runtime per chunk: roughly 30-110 minutes, depending on model and
mode (`single_agent` is a single LLM call per case and is usually much
faster than `protocol`, which delegates through category subagents plus
self-check tools).

Transient gateway errors (sporadic 504s from the PNNL Depot gateway,
especially with Gemini, raw HTTP read/connect timeouts, and mid-response
disconnects) are retried automatically up to 4 attempts with backoff -- you
do not need to manually retry a chunk for those. If a *few* cases still end
up in the printed "Case Failures" table / the written report's `failures`
after exhausting those 4 attempts (or a chunk is interrupted outright after
some cases already succeeded), see "Retrying a partial chunk failure"
below instead of re-running the whole chunk from scratch -- re-running a
28-case chunk to fix 1-4 failures redoes every already-successful case's
live LLM planning and execution too, which is the expensive part.

```bash
# --- protocol mode ---
.venv/bin/python -u benchmark/run_benchmark.py --mode protocol \
  --models "GPT 5.6 Terra" --runs 4 --concurrency 4 \
  --output benchmark/results/protocol_report_gpt_5_6_terra.json

.venv/bin/python -u benchmark/run_benchmark.py --mode protocol \
  --models "Claude Sonnet 5" --runs 4 --concurrency 4 \
  --output benchmark/results/protocol_report_claude_sonnet_5.json

.venv/bin/python -u benchmark/run_benchmark.py --mode protocol \
  --models "Gemini 3.7 Flash" --runs 4 --concurrency 4 \
  --output benchmark/results/protocol_report_gemini_3_7_flash.json

# --- single_agent mode ---
.venv/bin/python -u benchmark/run_benchmark.py --mode single_agent \
  --models "GPT 5.6 Terra" --runs 4 --concurrency 4 \
  --output benchmark/results/single_agent_report_gpt_5_6_terra.json

.venv/bin/python -u benchmark/run_benchmark.py --mode single_agent \
  --models "Claude Sonnet 5" --runs 4 --concurrency 4 \
  --output benchmark/results/single_agent_report_claude_sonnet_5.json

.venv/bin/python -u benchmark/run_benchmark.py --mode single_agent \
  --models "Gemini 3.7 Flash" --runs 4 --concurrency 4 \
  --output benchmark/results/single_agent_report_gemini_3_7_flash.json
```

After each command, confirm the printed summary table and the `wrote ...`
line, and that the corresponding `benchmark/results/<mode>_report_<model>.json`
file exists before moving to the next command.

**What to expect in each chunk's summary table** (from prior full runs
against these exact models): most cases will show `StructuralGrade: 1.00`
(success) or `0.500` (undetermined -- needs manual review, see Phase 3) or
`0.000` (crash). Crashes are normal here and do **not** indicate a bug in
the harness -- they were confirmed, case by case, to be genuine model
output-quality issues (e.g. a model referencing a `${settings.*}` field
that doesn't exist, inventing a tool parameter name, or omitting a
required parameter). Do not attempt to "fix" a model's crash by editing
framework code; that's the actual signal this benchmark is measuring. If
you see a *new* kind of failure not resembling "wrong/missing/hallucinated
parameter or reference," treat it as worth investigating rather than
assuming it's just model-quality noise.

### Retrying a partial chunk failure

A case lands in the written report's `failures` list (not `cases`) only
when it exhausted all 4 of its automatic per-case retry attempts -- see
`_is_transient_failure()` in `run_benchmark.py` for exactly which
exceptions qualify (gateway 5xx, read/connect timeouts, mid-response
disconnects, bare `ModelAPIError`, `UsageLimitExceeded`). These are
infra/gateway flakiness, not model-quality signal, so retrying just those
specific cases is safe and cheap -- unlike a `crash` grade (`StructuralGrade:
0.000` in `cases`, not `failures`), which is real signal and should never
be "fixed" by re-running.

1. Find the exact failed case name(s) from the chunk's own report file
   (the printed "Case Failures" table shows a truncated Case ID; the JSON
   has the full name):

   ```bash
   .venv/bin/python -c "
   import json
   d = json.load(open('benchmark/results/single_agent_report_gemini_3_7_flash.json'))
   for f in d['failures']:
       print(f['name'])
   "
   ```

2. Re-run *only* those cases with `--cases` (exact `model/task/run` names;
   cannot be combined with `--models`/`--tasks`/`--runs`), writing to a
   separate small patch file:

   ```bash
   .venv/bin/python -u benchmark/run_benchmark.py --mode single_agent \
     --cases "Gemini 3.7 Flash/task_06_water_balance/run1" \
     --output /tmp/patch_gemini_single_agent.json
   ```

   This takes seconds to minutes (one case's planning + execution) instead
   of redoing an entire 30-110 minute chunk.

3. Splice the patch back into the original chunk report in place --
   `merge_reports.py` dedupes by case name with **last-report-wins**
   semantics, so list the original chunk file first and the patch last:

   ```bash
   .venv/bin/python -u benchmark/merge_reports.py --mode single_agent \
     --reports benchmark/results/single_agent_report_gemini_3_7_flash.json \
               /tmp/patch_gemini_single_agent.json \
     --output benchmark/results/single_agent_report_gemini_3_7_flash.json
   ```

   If the patched case still lands in `failures`, repeat steps 1-3 for it.

This per-case retry loop is the preferred path for a small number of
failures. If *many* cases in a chunk failed at once (e.g. a sustained
multi-minute gateway outage affecting most of the chunk), retrying them
one-by-one has diminishing returns -- just re-run the whole chunk command
instead; it overwrites its own `--output` file from scratch.

## Phase 2: merge each mode's chunks into one report

Once all 3 chunks for a mode are done, combine them into the single
report path the rest of the pipeline expects:

```bash
.venv/bin/python -u benchmark/merge_reports.py --mode protocol \
  --reports benchmark/results/protocol_report_gpt_5_6_terra.json \
            benchmark/results/protocol_report_claude_sonnet_5.json \
            benchmark/results/protocol_report_gemini_3_7_flash.json \
  --output benchmark/results/protocol_report.json

.venv/bin/python -u benchmark/merge_reports.py --mode single_agent \
  --reports benchmark/results/single_agent_report_gpt_5_6_terra.json \
            benchmark/results/single_agent_report_claude_sonnet_5.json \
            benchmark/results/single_agent_report_gemini_3_7_flash.json \
  --output benchmark/results/single_agent_report.json
```

Each command prints `merged 3 report(s) (84 case(s)) -> ...`. 84 = 3
models x 7 tasks x 4 runs. If the case count is different, one of the
Phase 1 chunks likely didn't finish cleanly -- check its individual
report file before proceeding.

## Phase 3: grading pipeline (unchanged from the standard benchmark flow)

Run for both modes. See AGENTS.md "Benchmark" for what each step does.

```bash
for MODE in protocol single_agent; do
  .venv/bin/python -u benchmark/make_manual_queue.py \
    --report "benchmark/results/${MODE}_report.json" --mode "$MODE"

  .venv/bin/python -u benchmark/merge_grades.py \
    --report "benchmark/results/${MODE}_report.json" --mode "$MODE"
done
```

`make_manual_queue.py` writes `benchmark/results/manual_review_{mode}.csv`
with every case graded `undetermined` (score 0.5) queued for human
review -- fill in the `manual_grade` column (`success`/`silent`/`obvious`)
for each row before running `merge_grades.py`, which reads that CSV back
in and writes a resolved `final_grade` label onto each case in the report
JSON (in place; no separate output file).

## Phase 4 (optional): iterative self-debug experiment

Only run this if you want to replicate the paper's follow-on self-debug
experiment (up to 3 rounds of traceback-driven repair applied to every
crashed case). This is a separate experiment from the core single-shot
grid above, makes additional live-LLM calls, and is not required to have
a complete core benchmark result.

```bash
for MODE in protocol single_agent; do
  .venv/bin/python -u benchmark/self_debug_crashes.py \
    --report "benchmark/results/${MODE}_report.json" --mode "$MODE" --max-rounds 3
done
```

Writes `benchmark/results/self_debug_{mode}.json` (plain JSON rows, not a
`pydantic_evals` report).
