from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from common import WorkflowState
from common.workflow import Settings, Workflow
from common.workflow_validation import _catalog_singletons


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_DIR = REPO_ROOT / "data" / "sample"
REFERENCE_RESULTS_DIR = REPO_ROOT / "benchmark" / "reference_workflows" / "results"


@pytest.fixture(autouse=True)
def _reset_catalog_singletons():
    """Ensure catalog state from one test cannot affect another."""
    _catalog_singletons.clear()
    yield
    _catalog_singletons.clear()


@pytest.fixture
def sample_data_dir() -> Path:
    if not SAMPLE_DATA_DIR.is_dir():
        pytest.skip("data/sample is not available in this checkout")
    return SAMPLE_DATA_DIR


@pytest.fixture
def reference_results_dir() -> Path:
    if not REFERENCE_RESULTS_DIR.is_dir():
        pytest.skip("reference workflow results are not available in this checkout")
    return REFERENCE_RESULTS_DIR


@pytest.fixture
def workflow_state(tmp_path) -> WorkflowState:
    return WorkflowState(
        workflow=Workflow(
            name="Test workflow",
            description="Workflow used by unit tests",
            settings=Settings(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output")),
            steps=[],
        )
    )


@pytest.fixture
def run_context(workflow_state) -> RunContext[WorkflowState]:
    return RunContext(deps=workflow_state, model=TestModel(), usage=RunUsage())
