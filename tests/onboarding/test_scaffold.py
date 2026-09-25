"""Tests for adapter rendering and the generated-file safety rails.

Three things here are load-bearing and easy to break silently:

1. The generated adapter must use the repo's ``sys.path`` + ``from core.base``
   import convention. If it ever imports ``tools.core.base`` instead, it lands in
   a *different* module instance and ``tools/generate_catalog.py`` will not see
   the tool in ``TOOL_REGISTRY`` -- the tool vanishes from the catalog with no
   error anywhere.
2. ``outputs`` must be rendered in the canonical dict form. ``@esmflow_tool``
   calls ``out_spec.get('type', '')``, so a plain string value raises
   ``AttributeError`` at *run* time, long after registration appears to succeed.
3. The workflow snippet is copy-pasted by users, so it has to match the runner's
   reference syntax exactly.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from onboarding.models import CapabilityDraft, OutputDraft, ParamDraft
from onboarding.scaffold import (
    GENERATED_MARKER,
    ScaffoldError,
    adapter_path_for,
    relative_adapter_path,
    remove_adapter,
    render_adapter,
    render_workflow_snippet,
    validate_draft,
    write_adapter,
)


def _draft(**overrides) -> CapabilityDraft:
    """A complete, valid draft; override single fields per test."""

    payload: Dict[str, Any] = dict(
        tool_name="compute_my_index",
        description="Compute a standardized index from a timeseries.",
        subagent="diagnostics",
        category="analyzers",
        source_module="examples.user_code.my_index",
        source_function="compute_my_index",
        inputs=[
            ParamDraft(
                name="input_file",
                type="path",
                required=True,
                description="Input timeseries CSV.",
            ),
            ParamDraft(name="window", type="int", default=3),
        ],
        outputs=[
            OutputDraft(name="index_file", type="csv", description="The index table."),
            OutputDraft(name="n_events", type="int"),
        ],
    )
    payload.update(overrides)
    return CapabilityDraft(**payload)


# ---------------------------------------------------------------------------
# validate_draft
# ---------------------------------------------------------------------------


def test_a_complete_draft_has_no_problems():
    assert validate_draft(_draft()) == []


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        ({"tool_name": "not an identifier"}, "valid Python identifier"),
        ({"description": "   "}, "description is empty"),
        ({"source_function": ""}, "source_module and source_function"),
        ({"outputs": []}, "at least one output"),
    ],
)
def test_incomplete_drafts_are_rejected(overrides, expected_fragment):
    problems = validate_draft(_draft(**overrides))
    assert any(expected_fragment in problem for problem in problems), problems


def test_duplicate_input_and_output_names_are_reported():
    draft = _draft(
        inputs=[ParamDraft(name="input_file"), ParamDraft(name="input_file")],
        outputs=[OutputDraft(name="index_file"), OutputDraft(name="index_file")],
    )
    problems = validate_draft(draft)
    assert any("duplicate input name" in p for p in problems), problems
    assert any("duplicate output name" in p for p in problems), problems


def test_declaring_output_dir_as_an_input_is_rejected():
    # @esmflow_tool injects output_dir; declaring it would shadow the injection.
    problems = validate_draft(_draft(inputs=[ParamDraft(name="output_dir")]))
    assert any("output_dir" in p for p in problems), problems


def test_render_refuses_an_invalid_draft():
    with pytest.raises(ScaffoldError) as excinfo:
        render_adapter(_draft(outputs=[]))
    assert "at least one output" in str(excinfo.value)


# ---------------------------------------------------------------------------
# render_adapter
# ---------------------------------------------------------------------------


def test_adapter_uses_the_repo_sys_path_import_convention():
    source = render_adapter(_draft())

    assert "sys.path.insert(0, str(Path(__file__).parent.parent))" in source
    assert "from core.base import Param, ToolSpec, esmflow_tool" in source
    assert "from core.adapters import call_user_function, materialize_result" in source
    # The bug this guards against: importing through the `tools` package.
    assert "from tools.core" not in source
    assert "import tools." not in source


def test_adapter_is_marked_as_generated_and_records_its_provenance():
    source = render_adapter(_draft())

    assert GENERATED_MARKER in source
    assert "examples.user_code.my_index.compute_my_index" in source
    assert "USER_MODULE = 'examples.user_code.my_index'" in source
    assert "USER_FUNCTION = 'compute_my_index'" in source


def test_adapter_renders_inputs_with_types_requiredness_and_defaults():
    source = render_adapter(_draft())

    assert (
        "'input_file': Param(type='path', required=True, "
        "description='Input timeseries CSV.')," in source
    )
    assert "'window': Param(type='int', default=3)," in source


def test_adapter_renders_outputs_in_the_canonical_dict_form():
    source = render_adapter(_draft())

    # Dict form, not `'index_file': 'csv'` -- see the module docstring.
    assert "'index_file': {'type': 'csv', 'description': 'The index table.'}," in source
    assert "'n_events': {'type': 'int'}," in source


def test_no_inputs_renders_an_empty_mapping():
    source = render_adapter(_draft(inputs=[]))
    assert "inputs={}," in source


def test_notes_are_carried_into_the_docstring_for_the_user_to_review():
    source = render_adapter(_draft(notes=["output 'n_events': type guessed as int"]))

    assert "Onboarding notes:" in source
    assert "- output 'n_events': type guessed as int" in source


def test_rendered_adapter_is_syntactically_valid_python():
    compile(render_adapter(_draft()), "adapter.py", "exec")


# ---------------------------------------------------------------------------
# write / remove
# ---------------------------------------------------------------------------


def test_write_adapter_creates_the_package_and_refuses_to_clobber(tmp_path):
    draft = _draft()

    path = write_adapter(draft, tools_root=tmp_path)

    assert path == adapter_path_for(draft, tools_root=tmp_path)
    assert path == tmp_path / "analyzers" / "compute_my_index.py"
    assert (tmp_path / "analyzers" / "__init__.py").exists()
    assert GENERATED_MARKER in path.read_text()

    with pytest.raises(ScaffoldError) as excinfo:
        write_adapter(draft, tools_root=tmp_path)
    assert "overwrite=True" in str(excinfo.value)

    # ...and succeeds when explicitly allowed.
    assert write_adapter(draft, tools_root=tmp_path, overwrite=True) == path


def test_remove_adapter_deletes_generated_files_only(tmp_path):
    draft = _draft()
    path = write_adapter(draft, tools_root=tmp_path)

    assert remove_adapter("compute_my_index", "analyzers", tools_root=tmp_path) == path
    assert not path.exists()

    # Removing something absent is a no-op, not an error.
    assert remove_adapter("compute_my_index", "analyzers", tools_root=tmp_path) is None


def test_remove_adapter_refuses_a_hand_written_tool(tmp_path):
    hand_written = tmp_path / "analyzers" / "precious_tool.py"
    hand_written.parent.mkdir(parents=True)
    hand_written.write_text('"""A carefully hand-written tool."""\n')

    with pytest.raises(ScaffoldError) as excinfo:
        remove_adapter("precious_tool", "analyzers", tools_root=tmp_path)
    assert "not an onboarding-generated adapter" in str(excinfo.value)
    assert hand_written.exists()


def test_relative_adapter_path_is_repo_relative_and_posix():
    draft = _draft()

    assert (
        relative_adapter_path(adapter_path_for(draft))
        == "tools/analyzers/compute_my_index.py"
    )
    # Paths outside the repo are returned unchanged rather than raising.
    outside = Path("/tmp/elsewhere/tool.py")
    assert relative_adapter_path(outside) == "/tmp/elsewhere/tool.py"


# ---------------------------------------------------------------------------
# render_workflow_snippet
# ---------------------------------------------------------------------------


def test_snippet_declares_file_outputs_with_extensions_and_a_dotted_reference():
    snippet = render_workflow_snippet(_draft(), step_id="my_index")
    lines = snippet.splitlines()

    assert lines[0] == "- id: my_index"
    assert "  type: diagnostics" in lines
    assert "  tool: compute_my_index" in lines
    # Only the csv output is a file; the int scalar stays in the step result.
    assert "    index_file: my_index_index_file.csv" in lines
    assert not any("n_events" in line for line in lines)
    # The reference form the runner actually resolves.
    assert (
        "# Reference this from a later step as ${my_index.outputs.index_file}" in lines
    )


def test_snippet_lists_required_params_as_placeholders():
    lines = render_workflow_snippet(_draft()).splitlines()

    assert "  params:" in lines
    assert "    input_file: <fill me>" in lines
    # `window` has a default, so the planner need not supply it.
    assert not any(line.startswith("    window:") for line in lines)


def test_snippet_defaults_its_step_id_to_the_tool_name():
    assert render_workflow_snippet(_draft()).startswith("- id: compute_my_index_step")


def test_snippet_handles_no_params_and_no_file_outputs():
    draft = _draft(inputs=[], outputs=[OutputDraft(name="count", type="int")])
    lines = render_workflow_snippet(draft, step_id="s").splitlines()

    assert "    {}" in lines
    assert not any("outputs:" in line for line in lines)
    assert not any(line.startswith("# Reference") for line in lines)
