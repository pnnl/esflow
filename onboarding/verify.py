"""Post-registration checks for an onboarded capability.

Registration touches three coupled artifacts (adapter module, registry entry,
generated catalog) and two caches that are populated at import time
(``common.workflow``'s ``Literal`` allow-lists and
``common.workflow_validation``'s catalog singleton).  Verification therefore
runs in a **fresh subprocess** so the checks see the on-disk truth rather than
the state the onboarding process happened to import at startup.

Checks performed for a tool:

1. the adapter module imports and exposes ``SPEC`` / ``run``
2. ``SPEC.name`` matches the registered ``tool_name``
3. the tool appears in ``tools/tool_catalog.yaml``
4. the tool is accepted by its subagent's Step class ``Literal`` allow-list
5. a probe workflow containing one step for the tool passes
   ``validate_workflow`` and ``check_completeness``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from onboarding.models import VerificationReport
from onboarding.registry import find_capability

REPO_ROOT = Path(__file__).resolve().parent.parent

# Executed in a fresh interpreter; communicates via a JSON blob on stdout.
_PROBE_SOURCE = r'''
import importlib
import json
import sys

result = {"checks": [], "failures": []}
tool_name, subagent, adapter_module = sys.argv[1], sys.argv[2], sys.argv[3]


def ok(msg):
    result["checks"].append(msg)


def fail(msg):
    result["failures"].append(msg)


spec = None
try:
    module = importlib.import_module(adapter_module)
    ok(f"adapter module '{adapter_module}' imports cleanly")
except Exception as exc:
    fail(f"adapter module '{adapter_module}' failed to import: {exc!r}")
    module = None

if module is not None:
    spec = getattr(module, "SPEC", None)
    run = getattr(module, "run", None)
    if spec is None:
        fail("adapter does not define a module-level SPEC")
    if run is None or not callable(run):
        fail("adapter does not define a callable run(config)")
    if spec is not None:
        if getattr(spec, "name", None) == tool_name:
            ok(f"SPEC.name == '{tool_name}'")
        else:
            fail(f"SPEC.name is {getattr(spec, 'name', None)!r}, expected {tool_name!r}")
        if getattr(spec, "description", ""):
            ok("SPEC has a description for the planner")
        else:
            fail("SPEC.description is empty; the planner cannot choose this tool")
        if getattr(spec, "outputs", None):
            ok(f"SPEC declares outputs: {sorted(spec.outputs)}")
        else:
            fail("SPEC declares no outputs")

try:
    from common.workflow_validation import load_raw_catalog
    catalog = load_raw_catalog()
    names = {t.get("name") for t in (catalog.get("tools") or [])}
    if tool_name in names:
        ok("tool is present in tools/tool_catalog.yaml")
    else:
        fail(
            "tool is missing from tools/tool_catalog.yaml; run "
            "'python tools/generate_catalog.py --overwrite'"
        )
except Exception as exc:
    fail(f"could not read the tool catalog: {exc!r}")

step_class = None
try:
    from common.workflow import STEP_CLASSES
    step_class = STEP_CLASSES.get(subagent)
    if step_class is None:
        fail(f"no Step class is registered for subagent '{subagent}'")
    else:
        ok(f"subagent '{subagent}' maps to Step class {step_class.__name__}")
except Exception as exc:
    fail(f"could not load Step classes: {exc!r}")

if step_class is not None:
    try:
        step_class(id="probe", type=subagent, tool=tool_name, params={}, outputs={})
        ok(f"{step_class.__name__} accepts tool='{tool_name}'")
    except Exception as exc:
        fail(
            f"{step_class.__name__} rejects tool='{tool_name}' "
            f"(allow-list not widened): {exc!r}"
        )

if spec is not None and not result["failures"]:
    try:
        from common.workflow_validation import check_completeness, validate_workflow

        params = {}
        for key, param in (spec.inputs or {}).items():
            default = getattr(param, "default", None)
            required = getattr(param, "required", False)
            if required or default is None:
                ptype = getattr(param, "type", "str")
                params[key] = 0 if ptype in ("int", "float") else "probe_value"
        workflow = {
            "name": f"probe_{tool_name}",
            "description": f"Probe workflow exercising {tool_name}.",
            "settings": {"data_dir": "data", "output_dir": "outputs"},
            "steps": [
                {
                    "id": "probe",
                    "type": subagent,
                    "tool": tool_name,
                    "params": params,
                    "outputs": {
                        name: f"probe_{name}" for name in (spec.outputs or {})
                    },
                }
            ],
        }
        errors = validate_workflow(workflow)
        if errors:
            fail("probe workflow failed validation: " + "; ".join(errors))
        else:
            ok("a single-step probe workflow passes validate_workflow")
        gaps = check_completeness(workflow)
        if gaps:
            result["checks"].append(
                "check_completeness notes (expected for placeholder params): "
                + "; ".join(gaps)
            )
        else:
            ok("probe workflow is complete")
    except Exception as exc:
        fail(f"probe workflow could not be built: {exc!r}")

print("@@JSON@@" + json.dumps(result))
'''


def _adapter_module_name(adapter_path: str) -> str:
    return adapter_path.removesuffix(".py").replace("/", ".")


def verify_capability(
    tool_name: str,
    registry_file: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> VerificationReport:
    """Run all checks for ``tool_name`` and return a structured report."""

    entry = find_capability(tool_name, registry_file)
    if entry is None:
        return VerificationReport(
            tool_name=tool_name,
            failures=[f"'{tool_name}' is not present in the extension registry"],
        )

    root = repo_root or REPO_ROOT
    module_name = _adapter_module_name(entry.adapter_path)

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SOURCE, tool_name, entry.subagent, module_name],
        cwd=str(root),
        capture_output=True,
        text=True,
    )

    marker = "@@JSON@@"
    if marker in proc.stdout:
        payload = json.loads(proc.stdout.split(marker, 1)[1].strip())
        return VerificationReport(
            tool_name=tool_name,
            checks=payload.get("checks", []),
            failures=payload.get("failures", []),
        )

    detail = (proc.stderr or proc.stdout or "").strip()
    return VerificationReport(
        tool_name=tool_name,
        failures=[f"verification subprocess crashed: {detail[-1500:]}"],
    )


def verify_all(
    registry_file: Optional[Path] = None, repo_root: Optional[Path] = None
) -> List[VerificationReport]:
    """Verify every registered capability."""

    from onboarding.registry import list_capabilities

    return [
        verify_capability(entry.tool_name, registry_file, repo_root)
        for entry in list_capabilities(registry_file)
    ]
