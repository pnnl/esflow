"""A repeatable, self-cleaning onboarding demonstration.

The onboarding agent uses this module to *show* a user what onboarding does,
end to end, using one bundled exemplar
(``examples/user_code/demo_growing_degree_days.py``):

1. **scan**    -- introspect the exemplar into a draft (no writes)
2. **start**   -- register it: write the adapter, record the registry entry,
                  regenerate the catalog
3. **run**     -- execute a real one-step workflow that calls the new tool on
                  synthesized sample data
4. **end**     -- un-register it and delete the generated adapter, so the next
                  demonstration starts from the same clean state

Two properties make this safe to run from a chat session at any time:

*Idempotence.* ``end_demo`` tolerates a partially-completed demo -- a registry
entry with no adapter, an adapter with no registry entry, or nothing at all --
and always converges on "not registered". ``start_demo`` refuses to clobber a
*real* capability of the same name that this module did not create.

*Fresh-subprocess execution.* The run stage shells out to a new interpreter.
Registration changes two things that are computed at **import** time: the
``Literal`` tool allow-lists on the Step classes in ``common.workflow`` and the
cached catalog singleton in ``common.workflow_validation``. The process that
performed the registration therefore cannot see its own new tool. This is the
same reason ``onboarding/verify.py`` uses a subprocess, and it is precisely the
"you must restart to *use* a newly onboarded tool" limitation the demo exists to
make concrete -- the subprocess *is* the restart.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from onboarding.introspect import scan_source
from onboarding.models import CapabilityDraft
from onboarding.registry import (
    RegistryError,
    find_capability,
    list_capabilities,
    regenerate_catalog,
    register_capability,
    remove_capability,
)
from onboarding.scaffold import (
    adapter_path_for,
    relative_adapter_path,
    render_workflow_snippet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The bundled exemplar. Repo-relative so it appears verbatim in chat replies.
DEMO_SOURCE = "examples/user_code/demo_growing_degree_days.py"
DEMO_FUNCTION = "compute_growing_degree_days"
DEMO_TOOL = "compute_growing_degree_days"
DEMO_MODULE = "examples.user_code.demo_growing_degree_days"

#: Marks the exemplar's own sample CSV so it is recognisable in a temp dir.
SAMPLE_CSV_NAME = "demo_daily_temperature.csv"


class DemoError(RuntimeError):
    """Raised when the demonstration cannot proceed safely."""


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class DemoStatus(BaseModel):
    """Where the demonstration currently stands, from on-disk evidence only."""

    registered: bool = False
    adapter_exists: bool = False
    adapter_path: str = ""
    in_catalog: bool = False
    #: True when a capability of this name exists but did NOT come from the
    #: bundled exemplar -- i.e. a real user capability we must not touch.
    foreign_capability: bool = False

    @property
    def clean(self) -> bool:
        """No trace of the demo anywhere."""

        return not (self.registered or self.adapter_exists or self.in_catalog)

    def summary(self) -> str:
        if self.foreign_capability:
            return (
                f"A capability named '{DEMO_TOOL}' is registered but was NOT "
                "created by the demonstration; refusing to manage it."
            )
        if self.clean:
            return "Demo is not active: nothing registered, no adapter, not in the catalog."
        bits = [
            f"registry entry: {'yes' if self.registered else 'no'}",
            f"adapter file: {'yes' if self.adapter_exists else 'no'}"
            + (f" ({self.adapter_path})" if self.adapter_path else ""),
            f"in tool catalog: {'yes' if self.in_catalog else 'no'}",
        ]
        return "Demo is currently active -- " + "; ".join(bits)


def _catalog_tool_names() -> set:
    """Tool names in ``tools/tool_catalog.yaml``, read fresh from disk."""

    catalog_file = REPO_ROOT / "tools" / "tool_catalog.yaml"
    if not catalog_file.exists():
        return set()
    try:
        import yaml

        with catalog_file.open() as stream:
            catalog = yaml.safe_load(stream) or {}
    except Exception:  # pragma: no cover - unreadable catalog
        return set()
    return {t.get("name") for t in (catalog.get("tools") or []) if isinstance(t, dict)}


def demo_status(registry_path: Optional[Path] = None) -> DemoStatus:
    """Inspect the repository for traces of the demonstration."""

    draft = demo_draft()
    adapter = adapter_path_for(draft)
    entry = find_capability(DEMO_TOOL, registry_path)

    foreign = entry is not None and entry.source_module != DEMO_MODULE
    return DemoStatus(
        registered=entry is not None and not foreign,
        adapter_exists=adapter.exists(),
        adapter_path=relative_adapter_path(adapter) if adapter.exists() else "",
        in_catalog=DEMO_TOOL in _catalog_tool_names(),
        foreign_capability=foreign,
    )


# ---------------------------------------------------------------------------
# Stage 1: scan
# ---------------------------------------------------------------------------


def demo_draft() -> CapabilityDraft:
    """Introspect the bundled exemplar. Pure read; nothing is imported."""

    result = scan_source(DEMO_SOURCE, function_name=DEMO_FUNCTION)
    if not result.drafts:  # pragma: no cover - defensive
        raise DemoError(f"{DEMO_SOURCE} no longer defines {DEMO_FUNCTION}()")
    return result.drafts[0]


def demo_source_text() -> str:
    """The exemplar's source, for showing in chat."""

    return (REPO_ROOT / DEMO_SOURCE).read_text()


# ---------------------------------------------------------------------------
# Stage 2: sample data
# ---------------------------------------------------------------------------


def synthesize_sample_data(target_dir: Path, days: int = 180) -> Path:
    """Write a deterministic daily-temperature CSV the demo tool can consume.

    The repository ships no ``data/`` directory (it is gitignored), so the
    demonstration must bring its own input. A smooth seasonal sinusoid is used
    rather than random noise so the numbers a user sees are reproducible.
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / SAMPLE_CSV_NAME

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "tas_degc"])
        for day in range(days):
            # Peaks mid-summer at ~24 C, bottoms in winter at ~-4 C.
            temperature = 10.0 + 14.0 * math.sin(2 * math.pi * (day - 80) / 365.0)
            writer.writerow([_date_for(day), round(temperature, 3)])
    return path


def _date_for(day_offset: int) -> str:
    """ISO date ``day_offset`` days after 2001-01-01, without pandas."""

    from datetime import date, timedelta

    return (date(2001, 1, 1) + timedelta(days=day_offset)).isoformat()


def demo_workflow(input_file: Path, output_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """The one-step workflow the run stage executes."""

    draft = demo_draft()
    return {
        "name": "onboarding_demo",
        "description": (
            "Single-step demonstration workflow exercising the freshly "
            f"onboarded '{DEMO_TOOL}' tool."
        ),
        "settings": {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
        },
        "steps": [
            {
                "id": "growing_degree_days",
                "type": draft.subagent,
                "tool": DEMO_TOOL,
                "params": {
                    "input_file": str(input_file),
                    "temperature_column": "tas_degc",
                    "base_temperature": 10.0,
                    "upper_cutoff": 30.0,
                },
                "outputs": {"gdd_file": "growing_degree_days.csv"},
            }
        ],
    }


def demo_workflow_snippet() -> str:
    """The copy-pasteable snippet form, for teaching workflow authoring."""

    return render_workflow_snippet(demo_draft(), step_id="growing_degree_days")


# ---------------------------------------------------------------------------
# Stage 3: start (register)
# ---------------------------------------------------------------------------


class DemoStartResult(BaseModel):
    tool_name: str
    subagent: str
    adapter_path: str
    catalog_output: str = ""
    already_active: bool = False
    notes: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        lead = (
            "Demo capability was already registered"
            if self.already_active
            else "Registered the demo capability"
        )
        lines = [
            f"{lead}: {self.tool_name}  [{self.subagent}]",
            f"  generated adapter: {self.adapter_path}",
            f"  source function:   {DEMO_MODULE}.{DEMO_FUNCTION}",
        ]
        if self.notes:
            lines.append("  introspector notes:")
            lines.extend(f"    - {note}" for note in self.notes)
        return "\n".join(lines)


def start_demo(registry_path: Optional[Path] = None) -> DemoStartResult:
    """Register the exemplar as a real capability. Idempotent."""

    status = demo_status(registry_path)
    if status.foreign_capability:
        raise DemoError(status.summary())

    draft = demo_draft()
    if status.registered and status.adapter_exists and status.in_catalog:
        return DemoStartResult(
            tool_name=draft.tool_name,
            subagent=draft.subagent,
            adapter_path=status.adapter_path,
            already_active=True,
            notes=list(draft.notes),
        )

    # Any partial state is repaired by re-registering over the top.
    entry, catalog_output = register_capability(
        draft, path=registry_path, overwrite=True, regenerate=True
    )
    return DemoStartResult(
        tool_name=entry.tool_name,
        subagent=entry.subagent,
        adapter_path=entry.adapter_path,
        catalog_output=catalog_output,
        notes=list(draft.notes),
    )


# ---------------------------------------------------------------------------
# Stage 4: run (in a fresh interpreter)
# ---------------------------------------------------------------------------

# Executed by a new interpreter; communicates via a JSON blob on stdout so the
# parent never has to import the tool it just generated.
_RUN_PROBE = r'''
import json
import sys

payload = {"ok": False, "errors": [], "step": {}, "artifacts": []}
workflow_json, output_dir = sys.argv[1], sys.argv[2]

try:
    workflow = json.loads(workflow_json)

    from common.workflow import STEP_CLASSES
    from common.workflow_validation import validate_workflow

    step = workflow["steps"][0]
    step_class = STEP_CLASSES.get(step["type"])
    if step_class is None:
        payload["errors"].append(
            "no Step class for category '%s'" % step["type"]
        )
    else:
        # Proves the planner could now emit this step, not just that the
        # tool function runs.
        step_class(
            id=step["id"], type=step["type"], tool=step["tool"],
            params=step["params"], outputs=step["outputs"],
        )
        payload["planner_accepts_tool"] = True

    problems = validate_workflow(workflow, None)
    if problems:
        payload["errors"].extend(problems)

    if not payload["errors"]:
        from common.workflow_runner import run_workflow_definition

        context = run_workflow_definition(workflow, workflow_path="<demo>")
        if context is None:
            payload["errors"].append("workflow execution returned no context")
        else:
            entry = context.get(step["id"], {})
            result = entry.get("result") or {}
            payload["step"] = {
                "id": step["id"],
                "tool": step["tool"],
                "result": {k: v for k, v in result.items()},
                "outputs": entry.get("outputs") or {},
            }
            if "error" in result:
                payload["errors"].append(str(result["error"]))
            else:
                payload["ok"] = True

    import os
    if os.path.isdir(output_dir):
        payload["artifacts"] = sorted(os.listdir(output_dir))
except Exception as exc:
    import traceback
    payload["errors"].append("%r" % (exc,))
    payload["traceback"] = traceback.format_exc()

print("@@JSON@@" + json.dumps(payload, default=str))
'''


class DemoRunResult(BaseModel):
    ok: bool = False
    planner_accepts_tool: bool = False
    tool_name: str = DEMO_TOOL
    input_file: str = ""
    output_dir: str = ""
    scalars: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            lines = [f"Demo run FAILED for '{self.tool_name}':"]
            lines.extend(f"  - {err}" for err in self.errors)
            return "\n".join(lines)
        lines = [
            f"Ran a real one-step workflow using '{self.tool_name}'.",
            f"  input:  {self.input_file}  (synthesized, 180 days)",
            f"  output_dir: {self.output_dir}",
        ]
        if self.planner_accepts_tool:
            lines.append(
                "  the planner's Step allow-list accepted the tool, so it is "
                "selectable in a new session"
            )
        if self.scalars:
            lines.append("  scalar results (returned inline in the step result):")
            for key, value in self.scalars.items():
                lines.append(f"    {key}: {value}")
        if self.outputs:
            lines.append("  file artifacts (written into output_dir):")
            for key, value in self.outputs.items():
                lines.append(f"    {key}: {value}")
        return "\n".join(lines)


def run_demo(
    work_dir: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    timeout: int = 600,
) -> DemoRunResult:
    """Execute the demo tool through the real workflow runner.

    Runs in a fresh interpreter so the newly generated adapter, the regenerated
    catalog and the widened Step allow-lists are all visible.
    """

    status = demo_status(registry_path)
    if not status.registered:
        raise DemoError(
            "the demo capability is not registered yet; run start_demo() first"
        )

    temporary = work_dir is None
    base = Path(tempfile.mkdtemp(prefix="esmflow_demo_")) if temporary else Path(work_dir)
    try:
        data_dir = base / "data"
        output_dir = base / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        sample = synthesize_sample_data(data_dir)
        workflow = demo_workflow(sample, output_dir, data_dir)

        proc = subprocess.run(
            [sys.executable, "-c", _RUN_PROBE, json.dumps(workflow), str(output_dir)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        payload = _decode_probe(proc)
        step = payload.get("step") or {}
        outputs = step.get("outputs") or {}
        # A file-valued output appears twice in the runner's bookkeeping: as the
        # adapter's raw path in `result`, and as the renamed artifact in
        # `outputs`. Show it once, under artifacts.
        scalars = {
            key: value
            for key, value in (step.get("result") or {}).items()
            if key not in outputs
        }
        return DemoRunResult(
            ok=bool(payload.get("ok")),
            planner_accepts_tool=bool(payload.get("planner_accepts_tool")),
            input_file=str(sample),
            output_dir=str(output_dir),
            scalars=scalars,
            outputs=outputs,
            artifacts=payload.get("artifacts") or [],
            errors=payload.get("errors") or [],
        )
    finally:
        if temporary:
            shutil.rmtree(base, ignore_errors=True)


def _decode_probe(proc: subprocess.CompletedProcess) -> Dict[str, Any]:
    """Pull the ``@@JSON@@`` blob out of a probe's stdout."""

    for line in (proc.stdout or "").splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@") :])
    detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return {
        "ok": False,
        "errors": [
            "the demo subprocess produced no result blob "
            f"(exit code {proc.returncode})",
            detail[-2000:] or "<no output>",
        ],
    }


# ---------------------------------------------------------------------------
# Stage 5: end (offboard)
# ---------------------------------------------------------------------------


class DemoEndResult(BaseModel):
    removed_capability: bool = False
    removed_adapter: bool = False
    regenerated_catalog: bool = False
    was_already_clean: bool = False
    adapter_path: str = ""
    leftovers: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        if self.was_already_clean:
            return (
                "Nothing to offboard -- the demo capability was already absent, "
                "so the next demonstration starts clean."
            )
        lines = ["Offboarded the demo capability:"]
        if self.removed_capability:
            lines.append(f"  - removed the registry entry for {DEMO_TOOL}")
        if self.removed_adapter:
            lines.append(f"  - deleted the generated adapter {self.adapter_path}")
        if self.regenerated_catalog:
            lines.append("  - regenerated tools/tool_catalog.yaml without it")
        lines.append(
            "  - the exemplar source under examples/user_code/ is left in place; "
            "it is the teaching material, not a generated artifact"
        )
        if self.leftovers:
            lines.append("  WARNING -- leftovers still present:")
            lines.extend(f"    - {item}" for item in self.leftovers)
        return "\n".join(lines)


def end_demo(registry_path: Optional[Path] = None) -> DemoEndResult:
    """Return the repository to its pre-demo state. Idempotent.

    Handles every partial state: registry entry only, adapter only, both, or
    neither. A real (non-demo) capability of the same name is never touched.
    """

    status = demo_status(registry_path)
    if status.foreign_capability:
        raise DemoError(status.summary())

    if status.clean:
        return DemoEndResult(was_already_clean=True)

    result = DemoEndResult(adapter_path=status.adapter_path)
    adapter = adapter_path_for(demo_draft())

    if status.registered:
        try:
            remove_capability(
                DEMO_TOOL, path=registry_path, delete_adapter=True, regenerate=True
            )
        except RegistryError as exc:  # pragma: no cover - surfaced to the agent
            raise DemoError(f"could not un-register the demo capability: {exc}") from exc
        result.removed_capability = True
        result.removed_adapter = not adapter.exists()
        result.regenerated_catalog = True
    elif status.adapter_exists:
        # Orphaned adapter (e.g. the process died between the two writes).
        adapter.unlink(missing_ok=True)
        result.removed_adapter = True
        regenerate_catalog()
        result.regenerated_catalog = True
    elif status.in_catalog:
        regenerate_catalog()
        result.regenerated_catalog = True

    after = demo_status(registry_path)
    if after.registered:
        result.leftovers.append("registry entry still present")
    if after.adapter_exists:
        result.leftovers.append(f"adapter file still present: {after.adapter_path}")
    if after.in_catalog:
        result.leftovers.append("tool is still listed in tools/tool_catalog.yaml")
    return result


# ---------------------------------------------------------------------------
# The guidance the chat shows alongside the mechanics
# ---------------------------------------------------------------------------

# NOTE: a plain (non-f, non-.format) string. It contains literal ``${...}``
# workflow references and YAML braces, so the snippet is spliced in with a
# straight replace rather than any brace-aware formatting.
WHERE_CODE_GOES = """\
Where your code goes, and what the system generates for you
===========================================================

You write exactly one file. Everything else is generated.

  examples/user_code/your_module.py     <-- YOU WRITE THIS. Plain Python: your
                                           function, ordinary type hints, a
                                           docstring. No ESMFlow imports, no
                                           decorators, no tool contract.
                                           (Any importable package in the repo
                                           works; examples/user_code/ is the
                                           conventional home for user code.)

  tools/<category>/<tool_name>.py       <-- GENERATED adapter. A thin SPEC +
                                           run() wrapper that calls your
                                           function. Never hand-edit it; it is
                                           rewritten on every registration.

  extensions/registry.yaml             <-- GENERATED ledger. Records that your
                                           capability exists and which planner
                                           category it belongs to.

  tools/tool_catalog.yaml              <-- GENERATED catalog. What the planner
                                           reads to decide which tools exist.

  workflows (YAML the planner emits)   <-- your tool appears as one step:

{snippet}

  In a step, `params:` are your function's arguments and `outputs:` maps each
  file-valued result to the filename it takes inside `settings.output_dir`.
  Scalar results are not written to disk -- they come back in the step result
  and can be referenced by later steps. Your function never opens an output
  file itself: return a DataFrame/Dataset/Figure and ESMFlow materializes it.

  Data flows between steps by reference: a later step consumes
  `${growing_degree_days.outputs.gdd_file}` rather than a hard-coded path.

One caveat worth knowing: a newly onboarded tool becomes selectable by the
planner only in a **new** session, because the planner's tool allow-lists are
built when the process starts. The demonstration's run stage sidesteps this by
executing in a fresh interpreter.
"""


def where_code_goes() -> str:
    """Directory-and-workflow orientation, with the live exemplar's snippet."""

    snippet = "\n".join(
        "    " + line for line in demo_workflow_snippet().splitlines()
    )
    return WHERE_CODE_GOES.replace("{snippet}", snippet)


def demo_overview() -> str:
    """What the demonstration will do, before it does it."""

    draft = demo_draft()
    return "\n".join(
        [
            "Onboarding demonstration -- what will happen:",
            f"  1. scan   {DEMO_SOURCE}",
            f"            -> proposes tool '{draft.tool_name}' in the "
            f"'{draft.subagent}' category",
            "  2. start  writes the generated adapter, records the registry "
            "entry, regenerates the catalog",
            "  3. run    executes a real one-step workflow on synthesized "
            "sample data, in a fresh interpreter",
            "  4. end    un-registers the capability and deletes the generated "
            "adapter, leaving no trace",
            "",
            "Step 4 is what makes this repeatable: the demonstration can be "
            "shown again from an identical starting state. Your own capability "
            "would simply stop after step 3.",
        ]
    )
