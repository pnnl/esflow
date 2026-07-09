"""
pydantic-evals Dataset factory for structural grading.

Builds a ``Dataset`` covering every (model, task) combination,
wiring up a ``StructuralGrade`` evaluator per case.

Usage::

    from evals.structural_grading_dataset import make_dataset

    dataset = make_dataset(
        results_dir=Path("evals/reference_workflows/results"),
        models=["model1", "model2"],
        tasks=["task_01_obs_summary", ...],
    )

    def get_output_dir(label: str) -> Path:
        # label is "<model>/<task>/run1"
        model, task, run = label.split("/")
        return results_dir / f"{model}_protocol" / task / f"{run}_output"

    report = dataset.evaluate_sync(get_output_dir)
"""

from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic_evals import Case, Dataset

from evals.structural_grading import (
    StructuralGrade,
    find_key_csv,
    find_key_nc,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _resolve_ref_files(results_dir: Path, tasks: list[str]) -> dict[str, dict[str, Path | None]]:
    """Return reference CSV/NC paths keyed by task."""
    ref_files: dict[str, dict[str, Path | None]] = {}
    for task in tasks:
        ref_dir = results_dir / task
        ref_files[task] = {
            "csv": find_key_csv(ref_dir, task),
            "nc":  find_key_nc(ref_dir, task),
        }
    return ref_files


def make_dataset(
    results_dir: Path,
    models: list[str],
    tasks: list[str],
) -> Dataset[str, Path, None]:
    """Build a pydantic-evals ``Dataset`` for structural grading in protocol mode.

    Each ``Case`` represents one (model, task) pair.
    ``inputs`` carries the human-readable run label ``"<model>/<task>/<run_id>"``
    for reporting; evaluators read from ``ctx.output`` (a ``Path`` to the run
    output directory). A unique run ID is generated automatically.

    The function passed to ``dataset.evaluate`` must accept a ``str`` label
    and return a ``Path`` to the corresponding output directory::

        def get_output_dir(label: str) -> Path:
            model, task, run = label.split("/")
            return results_dir / f"{model}_protocol" / task / f"{run}_output"

        report = dataset.evaluate_sync(get_output_dir)
    
    Args:
        results_dir: Path to results directory (e.g., evals/reference_workflows/results)
        models: List of model names
        tasks: List of task names
    """
    ref_files = _resolve_ref_files(results_dir, tasks)
    run_id = uuid4().hex[:8]

    cases: list[Case[str, Path, None]] = []
    for model in models:
        for task in tasks:
            label = f"{model}/{task}/{run_id}"
            evaluators = [
                StructuralGrade(
                    task=task,
                    mode="protocol",
                    ref_csv=ref_files[task]["csv"],
                    ref_nc=ref_files[task]["nc"],
                )
            ]
            cases.append(
                Case(
                    name=f"{model}_protocol_{task}_{run_id}",
                    inputs=label,
                    expected_output=None,
                    evaluators=evaluators,
                )
            )

    return Dataset(
        name="structural_grading_protocol",
        cases=cases,
    )
