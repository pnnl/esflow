from dataclasses import dataclass
import asyncio
import tempfile
from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from agents.planner.settings import default_settings
from common.workflow import Workflow
from agents.planner.oneshot_planner import plan_workflow_one_shot
from common.workflow_validation import validate_workflow


@dataclass
class IsValidWorkflow(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason | bool:
        workflow: Workflow = ctx.output
        
        # Write workflow to a temp file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        workflow.write_to_file(tmp_path)
        
        # Validate the workflow
        errors = validate_workflow(workflow.to_yaml_dict())
        
        if errors:
            return EvaluationReason(
                value=False,
                reason=f"Validation failed:\n" + "\n".join(f"  - {e}" for e in errors),
            )
        else:
            return True

# Create a dataset with test cases
dataset = Dataset(
    name='validation',
    cases=[
        Case(
            name='observation_summary',
            inputs='./evals/task_01_obs_summary.txt',
        ),
        Case(
            name='seasonal_runoff',
            inputs='./evals/task_02_seasonal_runoff.txt',
        ),
        Case(
            name='et_benchmark',
            inputs='./evals/task_03_et_benchmark.txt',
        ),
        Case(
            name='streamflow_fdc',
            inputs='./evals/task_04_streamflow_fdc.txt',
        ),
        Case(
            name='basin_streamflow',
            inputs='./evals/task_05_basin_streamflow.txt',
        ),
        Case(
            name='water_balance',
            inputs='./evals/task_06_water_balance.txt',
        ),
        Case(
            name='integrated_diagnostic',
            inputs='./evals/task_07_integrated_diagnostic.txt',
        ),
    ],
    evaluators=[IsValidWorkflow()],
)

# Define the function to evaluate
def plan_workflow(task_file: str) -> Workflow:
    with open(task_file, 'r') as f:
        task_description = f.read()
    return asyncio.run(plan_workflow_one_shot(task_description, default_settings()))

# Run the evaluation
report = dataset.evaluate_sync(plan_workflow, max_concurrency=2)
# Print the results
report.print(include_reasons=True)