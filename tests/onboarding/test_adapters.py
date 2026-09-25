"""Tests for the user-function bridge in ``tools/core/adapters.py``.

This is the layer where a plain user function meets the ESMFlow tool contract,
so the interesting cases are all about *mismatch*: config keys the function does
not accept, required arguments nobody declared, and return values that have to
become files on disk.

``spec`` is duck-typed throughout ``adapters.py`` (only ``.name``, ``.inputs``
and ``.outputs`` are touched), so these tests use a small stub rather than
importing ``ToolSpec``. That deliberately avoids the ``tools/`` dual-module-
instance trap described in AGENTS.md.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pytest

from tools.core.adapters import (
    AdapterError,
    call_user_function,
    load_user_function,
    materialize_result,
    output_type_of,
    run_user_capability,
)


@dataclass
class SpecStub:
    """Minimal stand-in for ``ToolSpec`` (adapters only need these three)."""

    name: str = "my_tool"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)


USER_SOURCE = '''
"""Throwaway user module used by the adapter tests."""


def compute(input_file, window=3):
    return {"input_file": input_file, "window": window}


def writes_here(input_file, output_dir):
    return {"input_file": input_file, "output_dir": output_dir}


def takes_anything(input_file, **kwargs):
    return {"input_file": input_file, "kwargs": kwargs}


def needs_more(input_file, required_but_undeclared):
    return required_but_undeclared
'''


@pytest.fixture
def user_module(tmp_path, monkeypatch) -> str:
    """Write an importable throwaway user module and return its dotted path."""

    module_dir = tmp_path / "pkg"
    module_dir.mkdir()
    (module_dir / "user_code.py").write_text(USER_SOURCE)
    monkeypatch.syspath_prepend(str(module_dir))
    # Guard against a stale entry from another test in the same process.
    sys.modules.pop("user_code", None)
    return "user_code"


# ---------------------------------------------------------------------------
# load_user_function
# ---------------------------------------------------------------------------


def test_load_user_function_returns_the_callable(user_module):
    assert load_user_function(user_module, "compute").__name__ == "compute"


def test_missing_function_raises_a_helpful_adapter_error(user_module):
    with pytest.raises(AdapterError) as excinfo:
        load_user_function(user_module, "not_there")
    assert "has no function 'not_there'" in str(excinfo.value)


def test_unimportable_module_raises_an_adapter_error():
    with pytest.raises(AdapterError) as excinfo:
        load_user_function("a_module_that_does_not_exist", "compute")
    assert "Could not import user module" in str(excinfo.value)


# ---------------------------------------------------------------------------
# call_user_function
# ---------------------------------------------------------------------------


def test_only_declared_inputs_the_function_accepts_are_forwarded(user_module):
    spec = SpecStub(inputs={"input_file": None, "window": None, "unused": None})
    config = {
        "input_file": "a.csv",
        "window": 5,
        "unused": "declared but not in the signature",
        "output_dir": "/tmp/out",  # ESMFlow plumbing; `compute` must not see it
    }

    result = call_user_function(user_module, "compute", spec, config)

    assert result == {"input_file": "a.csv", "window": 5}


def test_none_values_fall_through_to_the_functions_own_default(user_module):
    spec = SpecStub(inputs={"input_file": None, "window": None})

    result = call_user_function(
        user_module, "compute", spec, {"input_file": "a.csv", "window": None}
    )

    assert result == {"input_file": "a.csv", "window": 3}


def test_output_dir_is_passed_only_when_the_function_asks_for_it(user_module):
    spec = SpecStub(inputs={"input_file": None})

    result = call_user_function(
        user_module,
        "writes_here",
        spec,
        {"input_file": "a.csv", "output_dir": "/tmp/out"},
    )

    assert result == {"input_file": "a.csv", "output_dir": "/tmp/out"}


def test_a_kwargs_function_receives_every_declared_input(user_module):
    spec = SpecStub(inputs={"input_file": None, "extra": None})

    result = call_user_function(
        user_module, "takes_anything", spec, {"input_file": "a.csv", "extra": 7}
    )

    assert result == {"input_file": "a.csv", "kwargs": {"extra": 7}}


def test_an_undeclared_required_argument_is_reported_before_calling(user_module):
    spec = SpecStub(inputs={"input_file": None})

    with pytest.raises(AdapterError) as excinfo:
        call_user_function(user_module, "needs_more", spec, {"input_file": "a.csv"})

    message = str(excinfo.value)
    assert "requires argument(s) ['required_but_undeclared']" in message
    assert "not declared as tool inputs" in message


# ---------------------------------------------------------------------------
# output_type_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared, expected",
    [
        ("csv", "csv"),  # ToolSpec's terse form
        ({"type": "png", "description": "a plot"}, "png"),  # catalog's rich form
        ({"description": "no type given"}, ""),
        (None, ""),
    ],
)
def test_output_type_of_accepts_both_declaration_shapes(declared, expected):
    assert output_type_of(declared) == expected


# ---------------------------------------------------------------------------
# materialize_result
# ---------------------------------------------------------------------------


def test_a_dataframe_is_written_as_csv(tmp_path):
    pd = pytest.importorskip("pandas")
    spec = SpecStub(outputs={"table_file": {"type": "csv"}})
    frame = pd.DataFrame({"value": [1, 2, 3]})

    result = materialize_result(spec, frame, {"output_dir": str(tmp_path)})

    written = Path(result["table_file"])
    assert written.exists()
    assert written.name == "my_tool_table_file.csv"
    assert "value" in written.read_text()


def test_an_xarray_dataset_is_written_as_netcdf(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    spec = SpecStub(outputs={"output_file": {"type": "netcdf"}})
    dataset = xr.Dataset({"t": ("x", [1.0, 2.0])})

    result = materialize_result(spec, dataset, {"output_dir": str(tmp_path)})

    written = Path(result["output_file"])
    assert written.exists() and written.suffix == ".nc"


def test_a_matplotlib_figure_is_saved_as_png(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    spec = SpecStub(outputs={"plot_file": {"type": "png"}})
    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])

    result = materialize_result(spec, figure, {"output_dir": str(tmp_path)})

    written = Path(result["plot_file"])
    assert written.exists()
    assert written.read_bytes()[:4] == b"\x89PNG"


def test_an_already_written_file_is_copied_to_the_declared_location(tmp_path):
    source = tmp_path / "elsewhere" / "precomputed.csv"
    source.parent.mkdir()
    source.write_text("a,b\n1,2\n")
    spec = SpecStub(outputs={"table_file": {"type": "csv"}})
    out_dir = tmp_path / "out"

    result = materialize_result(spec, str(source), {"output_dir": str(out_dir)})

    written = Path(result["table_file"])
    assert written.parent == out_dir
    assert written.read_text() == "a,b\n1,2\n"
    assert source.exists()  # the user's own file is left alone


def test_a_path_output_that_does_not_exist_is_an_error(tmp_path):
    spec = SpecStub(outputs={"table_file": {"type": "csv"}})

    with pytest.raises(AdapterError) as excinfo:
        materialize_result(
            spec, str(tmp_path / "never_written.csv"), {"output_dir": str(tmp_path)}
        )
    assert "no such file exists" in str(excinfo.value)


def test_the_runner_injected_filename_is_honoured(tmp_path):
    """``output_<key>`` comes from the workflow's ``outputs:`` mapping.

    Writing straight to it makes the runner's post-hoc rename a no-op.
    """

    pd = pytest.importorskip("pandas")
    spec = SpecStub(outputs={"index_file": {"type": "csv"}})
    target = tmp_path / "spi_table.csv"

    result = materialize_result(
        spec,
        pd.DataFrame({"v": [1]}),
        {"output_dir": str(tmp_path), "output_index_file": str(target)},
    )

    assert Path(result["index_file"]) == target
    assert target.exists()


def test_scalars_pass_through_without_touching_the_filesystem(tmp_path):
    spec = SpecStub(outputs={"n_events": {"type": "int"}, "worst": {"type": "float"}})

    result = materialize_result(
        spec, {"n_events": 28, "worst": -2.72}, {"output_dir": str(tmp_path)}
    )

    assert result == {"n_events": 28, "worst": -2.72}
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_files_and_scalars_are_split_correctly(tmp_path):
    pd = pytest.importorskip("pandas")
    spec = SpecStub(
        outputs={"index_file": {"type": "csv"}, "n_events": {"type": "int"}}
    )

    result = materialize_result(
        spec,
        {"index_file": pd.DataFrame({"v": [1, 2]}), "n_events": 2},
        {"output_dir": str(tmp_path)},
    )

    assert Path(result["index_file"]).exists()
    assert result["n_events"] == 2


def test_a_single_file_output_can_be_returned_bare_alongside_scalars(tmp_path):
    """One file output + optional scalars: a bare return maps to the file."""

    pd = pytest.importorskip("pandas")
    spec = SpecStub(
        outputs={"index_file": {"type": "csv"}, "n_events": {"type": "int"}}
    )

    result = materialize_result(
        spec, pd.DataFrame({"v": [1]}), {"output_dir": str(tmp_path)}
    )

    assert Path(result["index_file"]).exists()
    assert "n_events" not in result  # scalar simply not provided


def test_a_missing_file_output_is_an_error(tmp_path):
    spec = SpecStub(
        outputs={"index_file": {"type": "csv"}, "plot_file": {"type": "png"}}
    )
    # index_file is satisfied, so the failure must be about the absent plot_file.
    index = tmp_path / "given.csv"
    index.write_text("v\n1\n")

    with pytest.raises(AdapterError) as excinfo:
        materialize_result(
            spec, {"index_file": str(index)}, {"output_dir": str(tmp_path)}
        )
    assert "did not provide required output 'plot_file'" in str(excinfo.value)


def test_a_declared_type_mismatch_is_reported(tmp_path):
    pd = pytest.importorskip("pandas")
    spec = SpecStub(outputs={"plot_file": {"type": "png"}})

    with pytest.raises(AdapterError) as excinfo:
        materialize_result(spec, pd.DataFrame({"v": [1]}), {"output_dir": str(tmp_path)})
    assert "returned a pandas object (expected 'csv')" in str(excinfo.value)


def test_unmaterializable_values_are_reported_with_their_type(tmp_path):
    spec = SpecStub(outputs={"table_file": {"type": "csv"}})

    with pytest.raises(AdapterError) as excinfo:
        materialize_result(spec, object(), {"output_dir": str(tmp_path)})
    assert "cannot materialize output 'table_file'" in str(excinfo.value)


def test_extra_returned_keys_are_surfaced_for_logging(tmp_path):
    spec = SpecStub(outputs={"n_events": {"type": "int"}})

    result = materialize_result(
        spec, {"n_events": 1, "debug_note": "converged"}, {"output_dir": str(tmp_path)}
    )

    assert result == {"n_events": 1, "debug_note": "converged"}


def test_run_user_capability_calls_then_materializes(user_module, tmp_path):
    spec = SpecStub(
        name="compute",
        inputs={"input_file": None, "window": None},
        outputs={"input_file": {"type": "str"}, "window": {"type": "int"}},
    )

    result = run_user_capability(
        user_module,
        "compute",
        spec,
        {"input_file": "a.csv", "window": 5, "output_dir": str(tmp_path)},
    )

    assert result == {"input_file": "a.csv", "window": 5}
