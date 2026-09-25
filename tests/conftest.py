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

#: Files that describe the shipped system. A test that rewrites one of these has
#: escaped its sandbox, and whatever it wrote gets committed and shipped.
PROTECTED_REPO_FILES = (
    REPO_ROOT / "extensions" / "registry.yaml",
    REPO_ROOT / "tools" / "tool_catalog.yaml",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "mutates_repo_state: test legitimately rewrites extensions/registry.yaml "
        "or tools/tool_catalog.yaml and restores them itself",
    )


@pytest.fixture(autouse=True)
def _reset_catalog_singletons():
    """Ensure catalog state from one test cannot affect another."""
    _catalog_singletons.clear()
    yield
    _catalog_singletons.clear()


@pytest.fixture(autouse=True)
def _protect_committed_repo_state(request):
    """Fail any test that leaves the real registry or catalog modified.

    This is not paranoia; it caught a real shipped defect. ``TestModel`` calls
    *every* tool an agent exposes with dummy arguments. Because the planner
    exposes ``delegate_to_onboarding``, which runs the onboarding agent with the
    caller's model, a planner smoke test reached the onboarding *write* tools and
    executed ``create_subagent(name='a')`` against the real
    ``extensions/registry.yaml``. That junk subagent then became a live planner
    category and a ``call_a`` tool in every session, while the suite stayed green
    because nothing asserted on the committed file.

    Writes are reverted as well as reported, so one leaky test cannot corrupt the
    working tree or cascade into later tests.
    """
    if request.node.get_closest_marker("mutates_repo_state"):
        yield
        return

    before = {
        path: path.read_bytes() if path.is_file() else None
        for path in PROTECTED_REPO_FILES
    }

    yield

    dirtied = []
    for path, original in before.items():
        current = path.read_bytes() if path.is_file() else None
        if current == original:
            continue
        dirtied.append(path.relative_to(REPO_ROOT).as_posix())
        # Restore, so the working tree survives and later tests see clean state.
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)

    assert not dirtied, (
        f"this test modified committed repo state: {', '.join(dirtied)} "
        "(now restored). Point the write at tmp_path, stub the write function, or "
        "mark the test with @pytest.mark.mutates_repo_state if the mutation is "
        "the behaviour under test."
    )


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
