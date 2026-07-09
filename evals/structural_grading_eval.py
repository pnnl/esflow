from pathlib import Path
from common.workflow_runner import run_workflow_definition
from evals.structural_grading_dataset import make_dataset

MODEL = "claude-opus-4-6"
MODE = "protocol"
TASKS = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    # ... etc
]

# Run each task, storing outputs in a consistent location
results_dir = Path("evals/results")
for task in TASKS:
    for run in [1, 2, 3, 4]:
        output_path = results_dir / f"{MODEL}_{MODE}" / task / f"run{run}_output"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and run the workflow (your task file)
        task_file = Path(f"evals/{task}.txt")
        # ... read task description and create workflow ...
        
        context = run_workflow_definition(
            workflow_dict,
            workflow_path=task_file,
        )
        
        # Move outputs to the expected location
        temp_output = Path(context["output_dir"])
        for file in temp_output.glob("*"):
            if file.is_file():
                file.rename(output_path / file.name)

# Now evaluate
dataset = make_dataset("protocol", models=[MODEL])

def get_output_dir(label: str) -> Path:
    model, task, run = label.split("/")
    return results_dir / f"{model}_protocol" / task / f"{run}_output"

report = dataset.evaluate_sync(get_output_dir)