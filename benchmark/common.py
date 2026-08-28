"""Shared paths, model selection, and result helpers for the benchmark."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
PROTOCOL_DIR = BENCHMARK_DIR / "protocol"
BASELINES_DIR = BENCHMARK_DIR / "baselines"
REFERENCE_DIR = BENCHMARK_DIR / "reference_workflows" / "results"
SANDBOX_DIR = BENCHMARK_DIR / "sandbox"
SANDBOX_IMAGE = "esflow-v2-benchmark-baseline"

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

# task_03 and task_07 call fetch_ilamb_data, which downloads from
# https://www.ilamb.org over plain HTTP with no offline cache mechanism.
# Protocol-mode workflows execute unsandboxed on the host and always have
# network access; baseline containers otherwise run with --network=none.
# These two tasks are a deliberate, documented exception: their baseline
# containers get full outbound network access (not a narrower domain
# allow-list) so the comparison isn't invalidated by sandboxing alone.
NETWORK_REQUIRED_TASKS = {"task_03_et_benchmark", "task_07_integrated_diagnostic"}

# Appended to every baseline task prompt (initial generation and self-debug
# repair alike) so a repaired script can't silently drop the sandbox's
# writable mount point. Keep both call sites referencing this constant.
BASELINE_OUTPUT_INSTRUCTION = "Save every output file to: /workspace/output"

# Word-boundary anchored so benign substrings (e.g. a column named
# "error_bar", or output like "0 errors found") don't trigger a false
# crash. Intentionally mirrors v1's benchmark/run_benchmark.py pattern
# (r"\b(traceback|error|exception|failed|fatal)\b") -- a plain unanchored
# version was ported here at one point and produced false positives on any
# baseline script whose benign output happened to contain one of these
# words as part of a longer token.
ERROR_RE = re.compile(r"\b(traceback|error|exception|failed|fatal)\b", re.IGNORECASE)


def slug(value: str) -> str:
    """Return a stable filesystem-safe label."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_dir(mode: str, model: str, task: str) -> Path:
    return RESULTS_DIR / f"{slug(model)}_{mode}" / task


def output_dir(mode: str, model: str, task: str, run: int) -> Path:
    return run_dir(mode, model, task) / f"run{run}_output"


def prompt_path(mode: str, task: str) -> Path:
    return (PROTOCOL_DIR if mode == "protocol" else BASELINES_DIR) / f"{task}.txt"


def clean_artifact(text: str, language: str) -> str:
    """Remove a single Markdown fence when a model ignored output instructions."""
    stripped = text.strip()
    match = re.fullmatch(rf"```(?:{language})?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped
