#!/usr/bin/env python3
"""
Complete example: run workflows and grade them with structural grading.

This script demonstrates the full pipeline:
  1. Define a workflow (or load from YAML)
  2. Execute it via run_workflow_definition()
  3. Store outputs in the expected directory structure
  4. Create a structural grading Dataset
  5. Run evaluation and print results

Usage:
    python evals/example_workflow_to_grading.py
"""

from pathlib import Path
from common.workflow_runner import run_workflow_definition
from evals.structural_grading_dataset import make_dataset


# ===========================================================================
# Configuration
# ===========================================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL = "test-model"
MODE = "protocol"
TASKS = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    "task_03_et_benchmark",
    "task_04_streamflow_fdc",
    "task_05_basin_streamflow",
    "task_06_water_balance",
    "task_07_integrated_diagnostic",
]
NUM_RUNS = 2  # Run 1 and 2


# ===========================================================================
# Step 1: Run workflows and collect outputs
# ===========================================================================

def run_workflows():
    """
    Execute workflows for all tasks and runs, storing outputs in the
    expected directory structure.

    In a real scenario, you would:
      1. Read task descriptions from evals/*.txt
      2. Generate workflows (using planner agents)
      3. Execute each workflow
      4. Move outputs to RESULTS_DIR
    """
    print("=" * 70)
    print("STEP 1: Running workflows")
    print("=" * 70)

    for task in TASKS:
        for run_num in range(1, NUM_RUNS + 1):
            run_label = f"{MODEL}/{task}/run{run_num}"
            output_path = (
                RESULTS_DIR / f"{MODEL}_{MODE}" / task / f"run{run_num}_output"
            )
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"\n[{run_label}]")

            # ---------------------------------------------------------------
            # In this example, we create a dummy workflow that just outputs
            # a placeholder file. In reality, you would:
            #   1. Read the task description
            #   2. Use a planner agent to generate a workflow
            #   3. Call run_workflow_definition() with that workflow
            # ---------------------------------------------------------------

            workflow = {
                "name": f"dummy_{task}_run{run_num}",
                "settings": {
                    "output_dir": str(output_path),
                },
                "steps": [
                    {
                        "id": "dummy_step",
                        "tool": "compute_summary_stats",  # This must exist in tool_catalog.yaml
                        "params": {
                            "output_dir": str(output_path),
                        },
                        "outputs": {
                            "stats": "summary.csv",  # Task 01 expects CSV
                        },
                    }
                ],
            }

            try:
                context = run_workflow_definition(
                    workflow,
                    workflow_path=f"<memory>:{run_label}",
                    verbose=False,
                )
                print(f"  ✓ Outputs saved to {output_path}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # In a real scenario, decide whether to continue or abort


# ===========================================================================
# Step 2: Create a structural grading Dataset
# ===========================================================================

def create_grading_dataset():
    """
    Build a pydantic-evals Dataset for the runs we just executed.

    The Dataset will contain one Case per (model, task, run) triple,
    each wired with a StructuralGrade evaluator that knows which
    reference files to compare against.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Creating structural grading Dataset")
    print("=" * 70)

    # Create a subset dataset for just our model and tasks
    dataset = make_dataset(
        mode=MODE,
        results_dir=RESULTS_DIR,
        models=[MODEL],
        tasks=TASKS,
        runs=list(range(1, NUM_RUNS + 1)),
    )

    print(f"\n✓ Created Dataset with {len(dataset.cases)} cases")
    print(f"  Mode: {MODE}")
    print(f"  Model: {MODEL}")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Runs: {NUM_RUNS}")

    return dataset


# ===========================================================================
# Step 3: Run evaluation
# ===========================================================================

def evaluate_dataset(dataset):
    """
    Execute the evaluation, passing each case to our get_output_dir function.

    The evaluators will:
      1. Check if the final deliverable (CSV/PNG) exists (Step 1)
      2. If protocol mode, compare key files against the reference (Step 2)
      3. Return a score: 0.0 (crash), 0.5 (undetermined), 1.0 (success)
    """
    print("\n" + "=" * 70)
    print("STEP 3: Running evaluation")
    print("=" * 70)

    def get_output_dir(label: str) -> Path:
        """
        Map a case label to the corresponding output directory.

        label format: "<model>/<task>/run<n>"
        """
        model, task, run = label.split("/")
        return RESULTS_DIR / f"{model}_{MODE}" / task / f"{run}_output"

    print("\nEvaluating all cases...\n")
    report = dataset.evaluate_sync(get_output_dir, max_concurrency=1)

    return report


# ===========================================================================
# Step 4: Print results
# ===========================================================================

def print_results(report):
    """Print evaluation results and summary statistics."""
    print("\n" + "=" * 70)
    print("STEP 4: Results")
    print("=" * 70 + "\n")

    # Full details
    report.print(include_reasons=True)

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    scores = [case.score for case in report.cases]
    total = len(scores)
    crashes = sum(1 for s in scores if s == 0.0)
    undetermined = sum(1 for s in scores if s == 0.5)
    successes = sum(1 for s in scores if s == 1.0)

    print(f"Total cases: {total}")
    print(f"  Crash:       {crashes:3d} ({crashes/total*100:5.1f}%)")
    print(f"  Undetermined: {undetermined:3d} ({undetermined/total*100:5.1f}%)")
    print(f"  Success:     {successes:3d} ({successes/total*100:5.1f}%)")
    print(f"\nAverage score: {sum(scores) / len(scores):.3f}")

    # Per-task breakdown
    print("\nPer-task breakdown:")
    for task in TASKS:
        task_scores = [
            case.score for case in report.cases if task in case.name
        ]
        if task_scores:
            avg = sum(task_scores) / len(task_scores)
            print(f"  {task:35s}  avg={avg:.2f}")

    print()


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    """Run the full pipeline."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Workflow Execution → Structural Grading".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")

    try:
        # Step 1: Run workflows
        run_workflows()

        # Step 2: Create Dataset
        dataset = create_grading_dataset()

        # Step 3: Evaluate
        report = evaluate_dataset(dataset)

        # Step 4: Print results
        print_results(report)

        print("\n✓ Pipeline complete!")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
