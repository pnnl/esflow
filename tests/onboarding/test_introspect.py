"""Tests for AST-based introspection of user code into CapabilityDrafts.

``scan_source`` must never import the file it scans, so these tests deliberately
include a module whose imports do not exist.
"""

from pathlib import Path

import pytest

from onboarding.introspect import category_for_subagent, scan_source

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def test_scan_does_not_import_the_module(tmp_path):
    source = _write(
        tmp_path,
        "unimportable.py",
        '''"""Module that cannot be imported."""

import a_package_that_does_not_exist  # noqa: F401


def compute_thing(input_file: str) -> "pd.DataFrame":
    """Compute a thing.

    Args:
        input_file: Path to the input CSV.
    """
    raise NotImplementedError
''',
    )

    result = scan_source(source)

    assert [d.tool_name for d in result.drafts] == ["compute_thing"]


def test_private_helpers_and_async_functions_are_skipped(tmp_path):
    source = _write(
        tmp_path,
        "mixed.py",
        '''"""Mixed module."""


def _helper(x: int) -> int:
    """Private."""
    return x


async def fetch_remote(url: str) -> str:
    """Async fetch."""
    return url


def compute_public(input_file: str) -> "pd.DataFrame":
    """Public entry point."""
    raise NotImplementedError
''',
    )

    result = scan_source(source)

    assert [d.tool_name for d in result.drafts] == ["compute_public"]
    assert any("_helper" in s and "private" in s for s in result.skipped)
    assert any("fetch_remote" in s and "async" in s for s in result.skipped)


def test_selecting_a_named_function_bypasses_the_private_filter(tmp_path):
    source = _write(
        tmp_path,
        "private_only.py",
        '''"""Private only."""


def _compute(input_file: str) -> "pd.DataFrame":
    """Compute privately."""
    raise NotImplementedError
''',
    )

    result = scan_source(source, function_name="_compute")

    # The leading underscore is stripped so the tool name is a clean identifier.
    assert [d.tool_name for d in result.drafts] == ["compute"]


def test_missing_function_name_is_an_error(tmp_path):
    source = _write(tmp_path, "empty.py", '"""Nothing here."""\n')

    with pytest.raises(ValueError, match="not found"):
        scan_source(source, function_name="nope")


def test_non_python_and_missing_sources_are_rejected(tmp_path):
    text = _write(tmp_path, "notes.txt", "hello")

    with pytest.raises(ValueError, match="Only .py files"):
        scan_source(text)
    with pytest.raises(FileNotFoundError):
        scan_source(tmp_path / "ghost.py")


def test_parameter_types_defaults_and_requiredness(tmp_path):
    source = _write(
        tmp_path,
        "params.py",
        '''"""Params module."""


def compute_metrics(
    input_file: str,
    window: int = 3,
    scale: float = 1.5,
    normalize: bool = False,
    labels: list[str] = ["a"],
    value_column: str | None = None,
    mystery: SomeCustomType = None,
    output_dir: str = "./out",
) -> "pd.DataFrame":
    """Compute metrics.

    Args:
        input_file: Path to the input CSV.
        window: Rolling window length.
    """
    raise NotImplementedError
''',
    )

    draft = scan_source(source).drafts[0]
    inputs = {p.name: p for p in draft.inputs}

    # output_dir is runtime plumbing, never a planner-visible parameter.
    assert "output_dir" not in inputs

    # A file-ish name is promoted to 'path' even though it is annotated `str`.
    assert inputs["input_file"].type == "path"
    assert inputs["input_file"].required is True
    assert inputs["input_file"].default is None
    assert inputs["input_file"].description == "Path to the input CSV."

    assert (inputs["window"].type, inputs["window"].default) == ("int", 3)
    assert inputs["window"].required is False
    assert (inputs["scale"].type, inputs["scale"].default) == ("float", 1.5)
    assert (inputs["normalize"].type, inputs["normalize"].default) == ("bool", False)
    assert inputs["labels"].type == "list[str]"
    # `X | None` unwraps to the inner type.
    assert inputs["value_column"].type == "str"
    # An unrecognized annotation falls back to str and is flagged for review.
    assert inputs["mystery"].type == "str"
    assert any("mystery" in note for note in draft.notes)


def test_dataframe_return_becomes_a_csv_output(tmp_path):
    source = _write(
        tmp_path,
        "frames.py",
        '''"""Frames."""

import pandas as pd


def compute_table(input_file: str) -> pd.DataFrame:
    """Compute a table."""
    raise NotImplementedError
''',
    )

    draft = scan_source(source).drafts[0]

    assert [(o.name, o.type) for o in draft.outputs] == [("table_file", "csv")]


def test_returned_dict_literal_drives_multiple_outputs(tmp_path):
    source = _write(
        tmp_path,
        "multi.py",
        '''"""Multi-output."""

import pandas as pd


def compute_many(input_file: str) -> dict:
    """Compute several things."""
    frame = pd.DataFrame()
    return {
        "index_file": frame,
        "n_events": 3,
        "worst": -1.5,
    }
''',
    )

    draft = scan_source(source).drafts[0]
    outputs = {o.name: o.type for o in draft.outputs}

    assert outputs["index_file"] == "csv"
    assert outputs["n_events"] == "int"
    assert outputs["worst"] == "float"


def test_figure_return_suggests_the_visualization_subagent(tmp_path):
    source = _write(
        tmp_path,
        "plots.py",
        '''"""Plots."""

from matplotlib.figure import Figure


def plot_heatmap(input_file: str) -> Figure:
    """Draw a heatmap."""
    raise NotImplementedError
''',
    )

    draft = scan_source(source).drafts[0]

    assert draft.subagent == "visualization"
    assert draft.category == "plotters"
    assert [(o.name, o.type) for o in draft.outputs] == [("plot_file", "png")]


@pytest.mark.parametrize(
    "func_name, expected",
    [
        ("extract_basin_subset", "extraction"),
        ("fetch_reference_dataset", "data_discovery"),
        ("compute_basin_water_balance", "water_cycle"),
        ("compute_some_index", "diagnostics"),
    ],
)
def test_subagent_is_suggested_from_naming_hints(tmp_path, func_name, expected):
    source = _write(
        tmp_path,
        f"{func_name}.py",
        f'''"""Hints."""

import pandas as pd


def {func_name}(input_file: str) -> pd.DataFrame:
    """Do the thing."""
    raise NotImplementedError
''',
    )

    draft = scan_source(source).drafts[0]

    assert draft.subagent == expected
    assert draft.category == category_for_subagent(expected)


def test_default_to_diagnostics_is_flagged_for_confirmation(tmp_path):
    source = _write(
        tmp_path,
        "vague.py",
        '''"""Vague."""

import pandas as pd


def compute_thing(input_file: str) -> pd.DataFrame:
    """Do something unclassifiable."""
    raise NotImplementedError
''',
    )

    draft = scan_source(source).drafts[0]

    assert draft.subagent == "diagnostics"
    assert any("defaulted to 'diagnostics'" in note for note in draft.notes)


def test_missing_docstring_borrows_the_module_summary_and_notes_it(tmp_path):
    source = _write(
        tmp_path,
        "nodoc.py",
        '''"""Module level summary line."""

import pandas as pd


def compute_thing(input_file: str) -> pd.DataFrame:
    raise NotImplementedError
''',
    )

    draft = scan_source(source).drafts[0]

    assert draft.description == "Module level summary line."
    assert any("no function docstring" in note for note in draft.notes)


def test_scanning_a_shipped_exemplar_produces_a_complete_draft():
    """The exemplar user code must be onboardable without manual repair."""

    result = scan_source(REPO_ROOT / "examples" / "user_code" / "drought_indices.py")
    draft = next(d for d in result.drafts if d.source_function.startswith("compute"))

    assert draft.source_module == "examples.user_code.drought_indices"
    assert draft.description
    assert draft.outputs
    assert any(p.type == "path" for p in draft.inputs)
