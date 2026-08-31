"""Shared paths, model selection, and result helpers for the benchmark."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
PROTOCOL_DIR = BENCHMARK_DIR / "protocol"
REFERENCE_DIR = BENCHMARK_DIR / "reference_workflows" / "results"

TASKS = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    "task_03_et_benchmark",
    "task_04_streamflow_fdc",
    "task_05_basin_streamflow",
    "task_06_water_balance",
    "task_07_integrated_diagnostic",
]

# Current v2 equivalents of the five non-local models in the v1 paper suite.
PILOT_MODELS = [
    "Claude Opus 4.8",
    "GPT 5.4",
    "Gemini 3.5 Flash",
    "GPT o4 Mini",
    "Claude Haiku 4.5",
]

# All six E3SM-dependent tasks (task_02..task_07) reference this exact case
# name in their prompts and expect it available via ${settings.case_name} for
# tools like extract_gridded_field/match_to_grid, which require it as a
# separate param from data_dir. Settings.case_name is unset by
# agents/planner/settings.py's server-side default_settings() (a real user's
# case name can't be known in advance there), but the benchmark always
# targets this one fixed sample dataset, so it belongs here.
SAMPLE_CASE_NAME = "sample.v3.LR.historical"


def slug(value: str) -> str:
    """Return a stable filesystem-safe label."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_dir(mode: str, model: str, task: str) -> Path:
    return RESULTS_DIR / f"{slug(model)}_{mode}" / task


def output_dir(mode: str, model: str, task: str, run: int) -> Path:
    return run_dir(mode, model, task) / f"run{run}_output"


def prompt_path(mode: str, task: str) -> Path:
    """Return the task prompt file for a benchmark mode.

    Both "protocol" (agents/planner/oneshot_planner.py) and "single_agent"
    (agents/planner/single_agent_planner.py) read the exact same task
    description from benchmark/protocol/ -- the task doesn't change between
    arms, only which planning architecture receives it.
    """
    return PROTOCOL_DIR / f"{task}.txt"


def clean_artifact(text: str, language: str) -> str:
    """Remove a single Markdown fence when a model ignored output instructions."""
    stripped = text.strip()
    match = re.fullmatch(rf"```(?:{language})?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped
