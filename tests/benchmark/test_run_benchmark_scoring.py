"""Deterministic tests for the pure Python scoring helpers in
benchmark/run_benchmark.py.

No Docker or live LLM calls anywhere in this module. score_python_s1() does
spawn a subprocess to check that imports resolve, but that subprocess is
just `python -c <imports>` against the current interpreter/environment --
no network access and no Docker required.
"""

from benchmark.run_benchmark import score_python_s0, score_python_s1


def test_score_python_s0_passes_for_valid_syntax():
    ok, msg = score_python_s0("print('hello world')\n")
    assert ok is True
    assert "compiles" in msg.lower()


def test_score_python_s0_fails_for_invalid_syntax():
    ok, msg = score_python_s0("def broken(:\n    pass\n")
    assert ok is False
    assert "syntax error" in msg.lower()


def test_score_python_s1_passes_with_no_imports():
    ok, msg = score_python_s1("x = 1 + 1\nprint(x)\n")
    assert ok is True
    assert "no imports" in msg.lower()


def test_score_python_s1_passes_for_resolvable_stdlib_imports():
    ok, msg = score_python_s1("import os\nimport sys\nfrom pathlib import Path\n")
    assert ok is True
    assert "resolve" in msg.lower()


def test_score_python_s1_fails_for_unresolvable_import():
    ok, msg = score_python_s1("import this_module_definitely_does_not_exist_xyz\n")
    assert ok is False
    assert "modulenotfounderror" in msg.lower() or "no module named" in msg.lower()


def test_score_python_s1_only_inspects_import_lines_not_full_execution():
    """A script with resolvable imports but broken runtime logic should
    still pass s1 -- s1 is a lightweight import-only check, distinct from
    s2 (full sandboxed execution)."""
    code = (
        "import os\n"
        "raise RuntimeError('this would fail at full execution, not at s1')\n"
    )
    ok, msg = score_python_s1(code)
    assert ok is True
