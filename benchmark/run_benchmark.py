#!/usr/bin/env python3
"""
ESFlow Benchmark Runner

Sends task prompts to multiple LLMs via OpenAI-compatible API,
saves returned YAML workflows, and auto-scores them.

API Keys:
  Store in .env (gitignored) at the repo root:
    LLM_API_KEY_BIRTHRIGHT=sk-...   # for -birthright models
    LLM_API_KEY_PROJECT=sk-...      # for -project models (e.g., Opus 4.6)

Scoring:
  S0 — YAML parseable?
  S1 — Passes --dry-run? (valid tool names, required params, wiring)
  S2 — Executes without error? (requires data, optional)
  S3 — Scientifically correct? (human review)

Available models (PNNL Depot):
  Frontier:   gpt-5.2-birthright, claude-sonnet-4-5-20250929-v1-birthright,
              claude-opus-4-6-v1-project, grok-4-birthright
  Strong:     gpt-4.1-birthright, o3-birthright, claude-sonnet-4-20250514-v1-birthright
  Efficient:  o4-mini-birthright, claude-haiku-4-5-20251001-v1-birthright,
              grok-4-fast-non-reasoning-birthright

Default models (no --models flag):
  gpt-4.1-birthright, claude-opus-4-6-v1-project,
  claude-haiku-4-5-20251001-v1-birthright

Prompt modes (--mode):
  protocol — (default) YAML workflow generation with full tool catalog
  baseline — Python code generation with data-layout system message (no tool catalog)

Usage:
  # Default: protocol mode, 3 models, single task
  python benchmark/run_benchmark.py --task benchmark/protocol/task_01_obs_summary.txt

  # Default: protocol mode, 3 models, all 7 tasks
  python benchmark/run_benchmark.py

  # Code-gen baseline: same models, same task (no tool catalog)
  python benchmark/run_benchmark.py --mode baseline --task benchmark/protocol/task_01_obs_summary.txt

  # All Depot models (frontier + strong + efficient), all tasks, 3 runs each
  python benchmark/run_benchmark.py --all --runs 3

  # Specific models
  python benchmark/run_benchmark.py --models claude-opus-4-6-v1-project gpt-4.1-birthright

  # Use LM Studio (local, no API key needed)
  python benchmark/run_benchmark.py --local

  # Use Google Gemini
  python benchmark/run_benchmark.py --gemini

  # Mix Depot + Gemini models (Gemini models auto-routed to Google API)
  python benchmark/run_benchmark.py --models claude-opus-4-6-v1-project gpt-5-birthright gemini-2.5-flash

  # Compare protocol vs baseline for one task:
  python benchmark/run_benchmark.py --mode protocol --task benchmark/protocol/task_03_et_benchmark.txt
  python benchmark/run_benchmark.py --mode baseline --task benchmark/protocol/task_03_et_benchmark.txt
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Load .env file from repo root if it exists
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "tools" / "tool_catalog.yaml"
PROMPT_TEMPLATE_PATH = REPO_ROOT / "prompts" / "workflow_prompt_template.md"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUN_WORKFLOW = REPO_ROOT / "run_workflow.py"
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

# Prompt modes for comparison experiments
#   protocol — full tool catalog + YAML workflow generation  (default)
#   baseline — code-gen system msg + same task prompts (no tool catalog)
PROMPT_MODES = ["protocol", "baseline"]

# Default PNNL Depot endpoint
DEFAULT_BASE_URL = "https://ai-incubator-api.pnnl.gov"

# Model tiers for the benchmark
# Models ending in "-birthright" use LLM_API_KEY_BIRTHRIGHT
# Models ending in "-project" use LLM_API_KEY_PROJECT
MODEL_TIERS = {
    "frontier": [
        "gpt-5.2-birthright",
        "claude-sonnet-4-5-20250929-v1-birthright",
        "claude-opus-4-6-v1-project",
        "grok-4-birthright",
    ],
    "strong": [
        "gpt-4.1-birthright",
        "o3-birthright",
        "claude-sonnet-4-20250514-v1-birthright",
    ],
    "efficient": [
        "o4-mini-birthright",
        "claude-haiku-4-5-20251001-v1-birthright",
        "grok-4-fast-non-reasoning-birthright",
    ],
}

# Quick test: one from each Depot tier
TEST_MODELS = [
    "gpt-4.1-birthright",
    "claude-opus-4-6-v1-project",
    "claude-haiku-4-5-20251001-v1-birthright",
]

# LM Studio local models
LOCAL_BASE_URL = "http://localhost:1234/v1"
LOCAL_API_KEY = "lm-studio"

LOCAL_MODELS = {
    "local-small": [
        "meta-llama-3.1-8b-instruct",
    ],
    "local-mid": [
        "google_gemma-3-27b-it-qat",
    ],
}

# Google Gemini models (via google-genai SDK, not Depot)
GEMINI_MODELS = {
    "gemini-mid": [
        "gemini-2.5-flash",
    ],
}

# The 6 benchmark models for the paper (Depot + Gemini + Local)
PAPER_MODELS = [
    "claude-opus-4-6-v1-project",       # Frontier (Anthropic)
    "gpt-5-birthright",                  # Frontier (OpenAI)
    "gemini-2.5-flash",                  # Mid-tier (Google)
    "o4-mini-birthright",                # Reasoning/small (OpenAI)
    "claude-haiku-4-5-20251001-v1-birthright",  # Small/fast (Anthropic)
    "phi-4",                             # Local 15B (Microsoft, LM Studio)
]


def get_all_models():
    """Return flat list of all Depot models."""
    models = []
    for tier_models in MODEL_TIERS.values():
        models.extend(tier_models)
    return models


def get_all_local_models():
    """Return flat list of all local models."""
    models = []
    for tier_models in LOCAL_MODELS.values():
        models.extend(tier_models)
    return models


def get_tier(model_name):
    """Return tier name for a model."""
    for tier, models in MODEL_TIERS.items():
        if model_name in models:
            return tier
    for tier, models in GEMINI_MODELS.items():
        if model_name in models:
            return tier
    for tier, models in LOCAL_MODELS.items():
        if model_name in models:
            return tier
    return "local"


def get_all_gemini_models():
    """Return flat list of all Gemini models."""
    models = []
    for tier_models in GEMINI_MODELS.values():
        models.extend(tier_models)
    return models


def is_gemini_model(model_name):
    """Check if a model should be routed through the Gemini API."""
    return model_name.startswith("gemini-")


def is_local_model(model_name):
    """Check if a model should be routed through LM Studio."""
    # Depot models end with -birthright or -project; Gemini starts with gemini-
    if model_name.endswith(("-birthright", "-project")):
        return False
    if is_gemini_model(model_name):
        return False
    return True


# ---------------------------------------------------------------------------
# Build the system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(mode="protocol"):
    """Build system prompt for the given mode.

    Modes:
      protocol — YAML workflow generation with full tool catalog (~5500 tokens)
      baseline — Python code generation with data layout hints (~650 tokens)
    """
    if mode == "protocol":
        protocol_dir = Path(__file__).resolve().parent / "protocol"
        instructions = (protocol_dir / "system_protocol.txt").read_text()
        catalog = CATALOG_PATH.read_text()
        return f"{instructions}{catalog}\n"
    else:
        # baseline uses the code-gen system prompt
        return (BASELINES_DIR / "system_codegen.txt").read_text()


def get_task_prompt(task_file, mode="protocol"):
    """Return the task prompt text for the given mode.

    Both protocol and baseline use the same full task description.
    """
    task_file = Path(task_file)
    return task_file.read_text().strip()


# ---------------------------------------------------------------------------
# API key routing
# ---------------------------------------------------------------------------

def get_api_key_for_model(model):
    """Return the correct API key based on model suffix (-birthright or -project)."""
    if model.endswith("-project"):
        key = os.environ.get("LLM_API_KEY_PROJECT", "")
        if key:
            return key
    if model.endswith("-birthright"):
        key = os.environ.get("LLM_API_KEY_BIRTHRIGHT", "")
        if key:
            return key
    # Fallback to generic key
    return os.environ.get("LLM_API_KEY", "")


# ---------------------------------------------------------------------------
# Call LLM
# ---------------------------------------------------------------------------

def call_llm(model, task_prompt, system_prompt, base_url, api_key):
    """Send prompt to LLM and return raw response text + metadata."""
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Reasoning models (Opus, o-series, GPT-5) require temperature=1
    reasoning = ("opus" in model.lower() or
                 model.startswith("o3") or model.startswith("o4") or
                 "gpt-5" in model.lower())
    temp = 1.0 if reasoning else 0.0

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ],
            temperature=temp,
        )
        elapsed = time.time() - t0
        text = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        }
        return text, elapsed, usage, None
    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, None, str(e)


def call_gemini(model, task_prompt, system_prompt):
    """Send prompt to Google Gemini via google-genai SDK."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, 0, None, "GEMINI_API_KEY not set"

    client = genai.Client(api_key=api_key)

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=task_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
            ),
        )
        elapsed = time.time() - t0
        text = response.text
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        return text, elapsed, usage, None
    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, None, str(e)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def strip_markdown_fences(text):
    """Remove ```yaml/python ... ``` fences if present."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def score_s0(yaml_text):
    """S0: Is the output valid YAML?"""
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return False, "Parsed but not a dict"
        if "steps" not in data:
            return False, "No 'steps' key"
        return True, "Valid YAML with steps"
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"


def score_s1(yaml_path):
    """S1: Does it pass --dry-run?"""
    try:
        result = subprocess.run(
            [sys.executable, str(RUN_WORKFLOW), str(yaml_path), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            return True, "Dry-run passed"
        else:
            # Extract the key error line
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            error_msg = stderr[-200:] if stderr else stdout[-200:]
            return False, f"Dry-run failed: {error_msg}"
    except subprocess.TimeoutExpired:
        return False, "Dry-run timed out"
    except Exception as e:
        return False, f"Dry-run error: {e}"


def score_python_s0(code_text):
    """S0 for code-gen: Is the output valid Python syntax?"""
    try:
        compile(code_text, "<generated>", "exec")
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"


def score_python_s1(script_path):
    """S1 for code-gen: Do all imports succeed?"""
    # Extract import lines and try to execute just those
    code = Path(script_path).read_text()
    import_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line)
    if not import_lines:
        return True, "No imports to check"
    import_code = "\n".join(import_lines)
    try:
        result = subprocess.run(
            [sys.executable, "-c", import_code],
            capture_output=True, text=True, timeout=15,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            return True, "All imports succeeded"
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
            return False, f"Import failed: {err}"
    except subprocess.TimeoutExpired:
        return False, "Import check timed out"


def score_python_s2(script_path, timeout=120):
    """S2 for code-gen: Does the script execute without error?"""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            return True, "Execution succeeded"
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
            return False, f"Runtime error: {err}"
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out ({timeout}s)"


def _rewrite_output_dir(yaml_text, new_output_dir):
    """Rewrite the output_dir in settings so each run saves to its own directory."""
    try:
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict) and "settings" in data and isinstance(data["settings"], dict):
            data["settings"]["output_dir"] = new_output_dir
            return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception:
        pass
    return yaml_text


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(models, task_files, base_url, api_key, n_runs=1,
                   is_local=False, mode="protocol"):
    """Run benchmark across models and tasks. Return results list.

    mode: "protocol" (YAML workflow) or "baseline" (Python code-gen)
    """
    system_prompt = build_system_prompt(mode)
    is_codegen = mode == "baseline"
    results = []
    total_calls = len(models) * len(task_files) * n_runs
    call_num = 0

    for task_file in task_files:
        task_name = Path(task_file).stem
        task_prompt = get_task_prompt(task_file, mode)

        for model in models:
            model_short = model.replace("-birthright", "").replace("-project", "").replace("-v1", "")
            tier = get_tier(model)

            # Pick the right API key for this model
            if is_local:
                model_api_key = api_key
            else:
                model_api_key = get_api_key_for_model(model) or api_key

            for run_idx in range(n_runs):
                call_num += 1
                run_label = f"run{run_idx + 1}" if n_runs > 1 else "run1"

                print(f"\n{'='*70}")
                print(f"[{call_num}/{total_calls}] {model_short} | {task_name} | {run_label} | {mode}")
                print(f"{'='*70}")

                # Pre-compute output directory so we can inject it into code-gen prompts
                out_dir = RESULTS_DIR / f"{model_short}_{mode}" / task_name
                run_dir = out_dir / f"{run_label}_output"
                out_dir.mkdir(parents=True, exist_ok=True)
                run_dir.mkdir(parents=True, exist_ok=True)

                # For code-gen modes, append output dir to task prompt
                actual_task_prompt = task_prompt
                if is_codegen:
                    actual_task_prompt = (
                        f"{task_prompt}\n\n"
                        f"Save all output files to: {run_dir}"
                    )

                # Call LLM — route by model type
                print(f"  Calling {model}...")
                if is_gemini_model(model):
                    raw_text, elapsed, usage, error = call_gemini(
                        model, actual_task_prompt, system_prompt
                    )
                elif is_local_model(model):
                    raw_text, elapsed, usage, error = call_llm(
                        model, actual_task_prompt, system_prompt,
                        LOCAL_BASE_URL, LOCAL_API_KEY
                    )
                else:
                    raw_text, elapsed, usage, error = call_llm(
                        model, actual_task_prompt, system_prompt, base_url, model_api_key
                    )

                if error:
                    print(f"  ERROR: {error}")
                    results.append({
                        "task": task_name,
                        "model": model_short,
                        "tier": tier,
                        "mode": mode,
                        "run": run_idx + 1,
                        "s0": False,
                        "s1": False,
                        "s0_msg": f"API error: {error}",
                        "s1_msg": "Skipped",
                        "elapsed": elapsed,
                        "tokens": None,
                        "output_file": None,
                    })
                    continue

                tokens = usage.get("total_tokens") if usage else None
                print(f"  Response: {elapsed:.1f}s, {tokens or '?'} tokens")

                # Clean up markdown fences
                cleaned = strip_markdown_fences(raw_text)

                # Save outputs
                raw_path = out_dir / f"{run_label}_raw.txt"
                raw_path.write_text(raw_text)

                if is_codegen:
                    # --- Code-gen mode: save as .py and score Python ---
                    script_path = out_dir / f"{run_label}.py"
                    script_path.write_text(cleaned)
                    print(f"  Saved: {script_path}")

                    # S0: syntax
                    s0_pass, s0_msg = score_python_s0(cleaned)
                    s0_icon = "pass" if s0_pass else "FAIL"
                    print(f"  S0 (Python syntax): {s0_icon} — {s0_msg}")

                    # S1: imports
                    s1_pass, s1_msg = False, "Skipped (S0 failed)"
                    if s0_pass:
                        s1_pass, s1_msg = score_python_s1(script_path)
                        s1_icon = "pass" if s1_pass else "FAIL"
                        print(f"  S1 (imports):       {s1_icon} — {s1_msg}")

                    output_file = str(script_path)

                else:
                    # --- Protocol mode: save as .yaml and score YAML ---
                    yaml_path = out_dir / f"{run_label}.yaml"
                    cleaned_with_output_dir = _rewrite_output_dir(cleaned, str(run_dir))
                    yaml_path.write_text(cleaned_with_output_dir)
                    print(f"  Saved: {yaml_path}")
                    print(f"  Output dir: {run_dir}")

                    # S0: YAML parse
                    s0_pass, s0_msg = score_s0(cleaned)
                    s0_icon = "pass" if s0_pass else "FAIL"
                    print(f"  S0 (YAML parse): {s0_icon} — {s0_msg}")

                    # S1: dry-run
                    s1_pass, s1_msg = False, "Skipped (S0 failed)"
                    if s0_pass:
                        s1_pass, s1_msg = score_s1(yaml_path)
                        s1_icon = "pass" if s1_pass else "FAIL"
                        print(f"  S1 (dry-run):    {s1_icon} — {s1_msg}")

                    output_file = str(yaml_path)

                results.append({
                    "task": task_name,
                    "model": model_short,
                    "tier": tier,
                    "mode": mode,
                    "run": run_idx + 1,
                    "s0": s0_pass,
                    "s1": s1_pass,
                    "s0_msg": s0_msg,
                    "s1_msg": s1_msg,
                    "elapsed": round(elapsed, 1),
                    "tokens": tokens,
                    "output_file": output_file,
                })

    return results


def print_results_table(results):
    """Print a summary table."""
    print(f"\n{'='*90}")
    print("BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(f"{'Model':<35} {'Mode':<10} {'Tier':<10} {'S0':<6} {'S1':<6} {'Time':<8} {'Tokens':<8}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")

    for r in results:
        s0 = "pass" if r["s0"] else "FAIL"
        s1 = "pass" if r["s1"] else "FAIL"
        tokens = str(r["tokens"]) if r["tokens"] else "?"
        mode = r.get("mode", "protocol")
        print(f"{r['model']:<35} {mode:<10} {r['tier']:<10} {s0:<6} {s1:<6} {r['elapsed']:<8} {tokens:<8}")

    # Summary
    total = len(results)
    s0_pass = sum(1 for r in results if r["s0"])
    s1_pass = sum(1 for r in results if r["s1"])
    print(f"\nTotal: {total} | S0 pass: {s0_pass}/{total} | S1 pass: {s1_pass}/{total}")


def save_results(results):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ESFlow Benchmark Runner")
    parser.add_argument("--task", type=str, help="Single task file to run")
    parser.add_argument("--tasks-dir", type=str, default=str(Path(__file__).parent / "protocol"),
                        help="Directory of task files")
    parser.add_argument("--all", action="store_true", help="Run all Depot models")
    parser.add_argument("--local", action="store_true",
                        help="Run local LM Studio models (no API key needed)")
    parser.add_argument("--gemini", action="store_true",
                        help="Run Google Gemini models (requires GEMINI_API_KEY)")
    parser.add_argument("--models", nargs="+", help="Specific model names")
    parser.add_argument("--runs", type=int, default=1, help="Runs per model per task")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (auto-set for --local)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (default: LLM_API_KEY env var, not needed for --local)")
    parser.add_argument("--mode", type=str, default="protocol", choices=PROMPT_MODES,
                        help="Prompt mode: protocol (YAML+catalog), baseline (Python code-gen). "
                             "Default: protocol")

    # Paper benchmark shortcuts — all 6 models, 4 runs
    paper_group = parser.add_mutually_exclusive_group()
    paper_group.add_argument("--protocol", action="store_true",
                             help="Paper benchmark: all 6 models, 4 runs, protocol mode (YAML)")
    paper_group.add_argument("--baseline", action="store_true",
                             help="Paper benchmark: all 6 models, 4 runs, code-gen baseline")
    args = parser.parse_args()

    # Paper benchmark shortcuts
    if args.protocol or args.baseline:
        mode = "protocol" if args.protocol else "baseline"
        models = PAPER_MODELS
        n_runs = args.runs if args.runs > 1 else 4
        base_url = DEFAULT_BASE_URL
        api_key = None  # per-model keys resolved later

        # Verify LM Studio is reachable for local models
        local_models = [m for m in models if not m.endswith(("-birthright", "-project"))
                        and not is_gemini_model(m)]
        if local_models:
            try:
                import openai
                client = openai.OpenAI(api_key=LOCAL_API_KEY, base_url=LOCAL_BASE_URL)
                available = [m.id for m in client.models.list().data]
                missing = [m for m in local_models if m not in available]
                if missing:
                    print(f"WARNING: Local models not loaded in LM Studio: {missing}")
                    print(f"  Available: {available}")
                    models = [m for m in models if m not in missing]
            except Exception as e:
                print(f"WARNING: Cannot connect to LM Studio ({e})")
                print(f"  Skipping local models: {local_models}")
                models = [m for m in models if m not in local_models]

        if not models:
            print("ERROR: No models available to run")
            sys.exit(1)

        label = "PROTOCOL (YAML workflow)" if args.protocol else "BASELINE (Python code-gen)"
        print(f"\nPaper benchmark: {label}")
        print(f"  Models: {len(models)}")
        print(f"  Runs:   {n_runs}")
        for m in models:
            print(f"    - {m}")

        # Determine tasks
        if args.task:
            task_files = [args.task]
        else:
            tasks_dir = Path(args.tasks_dir)
            task_files = sorted(tasks_dir.glob("task_*.txt"))

        system_prompt = build_system_prompt(mode)
        results = run_benchmark(models, task_files, base_url, api_key, n_runs,
                                is_local=False, mode=mode)
        print_results_table(results)
        save_results(results)
        return

    # Determine mode: local, gemini, or Depot
    is_local = args.local

    if args.gemini:
        # Google Gemini mode — API key from env, no base_url needed
        base_url = None
        api_key = None
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            print("ERROR: Set GEMINI_API_KEY in .env or environment")
            sys.exit(1)

        if args.models:
            models = args.models
        else:
            models = get_all_gemini_models()

        print(f"\nUsing Google Gemini API")

    elif is_local:
        base_url = args.base_url or LOCAL_BASE_URL
        api_key = args.api_key or LOCAL_API_KEY

        # Check LM Studio is running
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            available = client.models.list()
            available_ids = [m.id for m in available.data]
            print(f"\nLM Studio models available: {available_ids}")
        except Exception as e:
            print(f"ERROR: Cannot connect to LM Studio at {base_url}")
            print(f"  Make sure LM Studio is running with a model loaded.")
            print(f"  Detail: {e}")
            sys.exit(1)

        # Pick models
        if args.models:
            models = args.models
        else:
            models = get_all_local_models()

        # Check requested models are loaded
        for m in models:
            if m not in available_ids:
                print(f"\nWARNING: '{m}' not loaded in LM Studio.")
                print(f"  Available: {available_ids}")
                print(f"  Load it in LM Studio, then re-run. Or use --models to pick from available.")

        # Filter to only available models
        runnable = [m for m in models if m in available_ids]
        if not runnable:
            print("ERROR: None of the requested models are loaded in LM Studio.")
            sys.exit(1)

        if len(runnable) < len(models):
            skipped = [m for m in models if m not in available_ids]
            print(f"  Skipping unavailable: {skipped}")
            print(f"  Running with: {runnable}")
            models = runnable
        else:
            models = runnable

    else:
        base_url = args.base_url or DEFAULT_BASE_URL
        api_key = args.api_key or os.getenv("LLM_API_KEY")
        # Per-model keys (LLM_API_KEY_BIRTHRIGHT / LLM_API_KEY_PROJECT) are
        # resolved later in get_api_key_for_model().  Only error out when
        # *no* key source exists at all.
        if not api_key and not (os.getenv("LLM_API_KEY_BIRTHRIGHT") or os.getenv("LLM_API_KEY_PROJECT")):
            print("ERROR: Set LLM_API_KEY, LLM_API_KEY_BIRTHRIGHT, or LLM_API_KEY_PROJECT")
            print("  (or store them in .env at the repo root)")
            print("  Or use --local for LM Studio models (no key needed)")
            sys.exit(1)

        # Models
        if args.models:
            models = args.models
        elif args.all:
            models = get_all_models()
        else:
            models = TEST_MODELS
            print(f"Using test models (use --all for full benchmark, --local for LM Studio)")

    # Tasks
    if args.task:
        task_files = [args.task]
    else:
        tasks_dir = Path(args.tasks_dir)
        task_files = sorted(tasks_dir.glob("task_*.txt"))
        if not task_files:
            print(f"ERROR: No task files found in {tasks_dir}")
            sys.exit(1)

    print(f"\nESFlow Benchmark")
    print(f"  Endpoint: {'Local (LM Studio)' if is_local else 'Depot API'}")
    print(f"  Prompt:   {args.mode} ({'YAML workflow' if args.mode == 'protocol' else 'Python code-gen'})")
    print(f"  Models:   {len(models)}")
    print(f"  Tasks:    {len(task_files)}")
    print(f"  Runs:     {args.runs}")
    print(f"  Total:    {len(models) * len(task_files) * args.runs} calls")
    print(f"  API:      {base_url}")
    print(f"  Models:   {', '.join(models)}")

    results = run_benchmark(models, task_files, base_url, api_key, args.runs,
                            is_local=is_local, mode=args.mode)
    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
