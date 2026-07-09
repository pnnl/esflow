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

import asyncio
from pathlib import Path
from common.workflow_runner import run_workflow_definition
from common.workflow import Settings
from evals.structural_grading_dataset import make_dataset
from agents.planner.oneshot_planner import plan_workflow_one_shot


# ===========================================================================
# Configuration
# ===========================================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL = "Gemini 3.5 Flash"
TASKS = [
    "task_01_obs_summary",
    # "task_02_seasonal_runoff",
    # "task_03_et_benchmark",
    # "task_04_streamflow_fdc",
    # "task_05_basin_streamflow",
    # "task_06_water_balance",
    # "task_07_integrated_diagnostic",
]


# ===========================================================================
# Step 1: Run workflows and collect outputs
# ===========================================================================

def run_workflows():
    """
    Execute workflows for all tasks by reading prompts and using the oneshot planner.

    For each task:
      1. Read the task prompt from evals/protocol_prompts/{task}.txt
      2. Use the oneshot planner to generate a workflow
      3. Execute the workflow
      4. Store outputs in the expected directory structure
    """
    print("=" * 70)
    print("STEP 1: Running workflows")
    print("=" * 70)

    for task in TASKS:
        run_label = f"{MODEL}/{task}/run1"
        output_path = (
            RESULTS_DIR / f"{MODEL}_protocol" / task / "run1_output"
        )
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[{run_label}]")

        # Read task prompt
        prompt_file = Path(__file__).resolve().parent / "protocol_prompts" / f"{task}.txt"
        try:
            with open(prompt_file) as f:
                user_goal = f.read()
        except FileNotFoundError:
            print(f"  ✗ Prompt file not found: {prompt_file}")
            continue

        # Use oneshot planner to generate workflow
        settings = Settings(output_dir=str(output_path))
        try:
            workflow = asyncio.run(plan_workflow_one_shot(user_goal, settings))
            print(f"  ✓ Workflow generated with {len(workflow.steps)} steps")
        except Exception as e:
            print(f"  ✗ Planning failed: {e}")
            continue

        # Execute the workflow
        try:
            context = run_workflow_definition(
                workflow,
                workflow_path=f"<planner>:{run_label}",
                verbose=False,
            )
            print(f"  ✓ Outputs saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Execution failed: {e}")


# ===========================================================================
# Step 2: Create a structural grading Dataset
# ===========================================================================

def create_grading_dataset():
    """
    Build a pydantic-evals Dataset for the runs we just executed.

    A unique ID is generated automatically by make_dataset(),
    so repeated evaluations won't collide even with the same dataset.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Creating structural grading Dataset")
    print("=" * 70)

    dataset = make_dataset(
        results_dir=RESULTS_DIR,
        models=[MODEL],
        tasks=TASKS,
    )

    print(f"\n✓ Created dataset with {len(dataset.cases)} cases")
    print(f"  Model: {MODEL}")
    print(f"  Tasks: {len(TASKS)}")

    return dataset


# ===========================================================================
# Step 3: Run evaluation
# ===========================================================================

def evaluate_dataset(dataset, repeat: int = 2):
    """
    Execute the evaluation for the dataset with repetitions.

    The evaluators will:
      1. Check if the final deliverable (CSV/PNG) exists (Step 1)
      2. If protocol mode, compare key files against the reference (Step 2)
      3. Return a score: 0.0 (crash), 0.5 (undetermined), 1.0 (success)
    
    Args:
        dataset: The pydantic-evals Dataset to evaluate
        repeat: Number of times to repeat the evaluation (default 2)
    """
    print("\n" + "=" * 70)
    print("STEP 3: Running evaluation")
    print("=" * 70)

    def get_output_dir(label: str) -> Path:
        """
        Map a case label to the corresponding output directory.

        label format: "<model>/<task>/<run_id>"
        """
        model, task, run = label.split("/")
        return RESULTS_DIR / f"{model}_protocol" / task / f"{run}_output"

    print("\nEvaluating all cases...\n")
    report = dataset.evaluate_sync(get_output_dir, repeat=repeat, max_concurrency=1)

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
    print("\n" + "-" * 70)
    print("Summary")
    print("-" * 70)

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

        # Step 3: Evaluate (with repetitions)
        report = evaluate_dataset(dataset, repeat=2)

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
