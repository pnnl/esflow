"""
pydantic-evals Dataset factory for structural grading.

Builds a ``Dataset`` covering every (model, task, run) combination for a
given mode, wiring up a ``StructuralGrade`` evaluator per case.

Usage::

    from evals.structural_grading_dataset import make_dataset

    dataset = make_dataset("protocol")

    def get_output_dir(label: str) -> Path:
        # label is "<model>/<task>/run<n>"
        model, task, run = label.split("/")
        return RESULTS_DIR / f"{model}_protocol" / task / f"{run}_output"

    report = dataset.evaluate_sync(get_output_dir)
"""

from pathlib import Path
from typing import Literal

from pydantic_evals import Case, Dataset

from evals.structural_grading import (
    StructuralGrade,
    find_key_csv,
    find_key_nc,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = [
    "claude-opus-4-6",
    "gpt-5",
    "gemini-2.5-flash",
    "o4-mini",
    "claude-haiku-4-5-20251001",
    "phi-4",
]
TASKS = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    "task_03_et_benchmark",
    "task_04_streamflow_fdc",
    "task_05_basin_streamflow",
    "task_06_water_balance",
    "task_07_integrated_diagnostic",
]
RUNS = [1, 2, 3, 4]

# Reference run used to locate ground-truth CSV/NC files for success detection.
REF_MODEL = "claude-opus-4-6"
REF_RUN = 2


def _resolve_ref_files(results_dir: Path) -> dict[str, dict[str, Path | None]]:
    """Return reference CSV/NC paths keyed by task."""
    ref_files: dict[str, dict[str, Path | None]] = {}
    for task in TASKS:
        ref_dir = (
            results_dir / f"{REF_MODEL}_protocol" / task / f"run{REF_RUN}_output"
        )
        ref_files[task] = {
            "csv": find_key_csv(ref_dir, task),
            "nc":  find_key_nc(ref_dir, task),
        }
    return ref_files


def make_dataset(
    mode: Literal["protocol", "baseline"],
    results_dir: Path = RESULTS_DIR,
    models: list[str] = MODELS,
    tasks: list[str] = TASKS,
    runs: list[int] = RUNS,
) -> Dataset[str, Path, None]:
    """Build a pydantic-evals ``Dataset`` for structural grading.

    Each ``Case`` represents one (model, task, run) triple.
    ``inputs`` carries the human-readable run label ``"<model>/<task>/run<n>"``
    for reporting; evaluators read from ``ctx.output`` (a ``Path`` to the run
    output directory).

    The function passed to ``dataset.evaluate`` must accept a ``str`` label
    and return a ``Path`` to the corresponding output directory::

        def get_output_dir(label: str) -> Path:
            model, task, run = label.split("/")
            return results_dir / f"{model}_{mode}" / task / f"{run}_output"

        report = dataset.evaluate_sync(get_output_dir)
    """
    ref_files = _resolve_ref_files(results_dir)

    cases: list[Case[str, Path, None]] = []
    for model in models:
        for task in tasks:
            for run in runs:
                label = f"{model}/{task}/run{run}"
                evaluators = [
                    StructuralGrade(
                        task=task,
                        mode=mode,
                        ref_csv=ref_files[task]["csv"] if mode == "protocol" else None,
                        ref_nc=ref_files[task]["nc"] if mode == "protocol" else None,
                    )
                ]
                cases.append(
                    Case(
                        name=f"{model}_{mode}_{task}_run{run}",
                        inputs=label,
                        expected_output=None,
                        evaluators=evaluators,
                    )
                )

    return Dataset(
        name=f"structural_grading_{mode}",
        cases=cases,
    )
