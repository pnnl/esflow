"""Pydantic models describing onboarding drafts, registry entries and reports.

These are the structures the onboarding agent reads and mutates. Keeping them
typed means the agent's tool signatures are self-documenting and every guess the
introspector makes is visible to the user before anything is written to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Mirrors tools/core/base.Param's coercible types.
ParamType = Literal["str", "int", "float", "bool", "path", "list[str]", "list[int]"]

# Mirrors ToolSpec output types; the file-backed ones drive artifact writing.
OutputType = Literal["csv", "png", "netcdf", "int", "float", "str", "dict"]


class ParamDraft(BaseModel):
    """A proposed tool input, derived from one user-function argument."""

    name: str
    type: ParamType = "str"
    required: bool = False
    default: Optional[Any] = None
    description: str = ""

    def to_spec_kwargs(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
        }


class OutputDraft(BaseModel):
    """A proposed tool output."""

    name: str
    type: OutputType = "csv"
    description: str = ""


class CapabilityDraft(BaseModel):
    """A candidate capability awaiting user confirmation and registration."""

    tool_name: str
    description: str = ""
    category: str = "analyzers"
    subagent: str = "diagnostics"
    source_module: str
    source_function: str
    inputs: List[ParamDraft] = Field(default_factory=list)
    outputs: List[OutputDraft] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-screen rendering used in agent replies."""

        lines = [
            f"tool_name:  {self.tool_name}",
            f"source:     {self.source_module}.{self.source_function}",
            f"category:   {self.category}  (directory tools/{self.category}/)",
            f"subagent:   {self.subagent}",
            f"description:{' ' + self.description if self.description else ' <missing>'}",
            "inputs:",
        ]
        for param in self.inputs:
            flag = "required" if param.required else f"default={param.default!r}"
            lines.append(f"  - {param.name}: {param.type} ({flag}) {param.description}")
        lines.append("outputs:")
        for output in self.outputs:
            lines.append(f"  - {output.name}: {output.type} {output.description}")
        if self.notes:
            lines.append("notes (please confirm):")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


class SubagentDraft(BaseModel):
    """A proposed new planner category/subagent."""

    name: str
    display_name: str = ""
    description: str = ""
    step_class: str = ""
    result_label: str = ""
    call_tool_name: str = ""


class ScanResult(BaseModel):
    """Everything a source scan found."""

    source: str
    drafts: List[CapabilityDraft] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Scanned {self.source}: {len(self.drafts)} candidate function(s)."]
        for draft in self.drafts:
            lines.append(
                f"  - {draft.source_function} -> tool '{draft.tool_name}' "
                f"({draft.subagent}/{draft.category})"
            )
        for skip in self.skipped:
            lines.append(f"  (skipped) {skip}")
        return "\n".join(lines)


class RegistryEntry(BaseModel):
    """A persisted capability record in ``extensions/registry.yaml``."""

    tool_name: str
    category: str
    subagent: str
    source_module: str
    source_function: str
    adapter_path: str
    onboarded_at: str = ""


class VerificationReport(BaseModel):
    """Result of post-registration checks for one capability."""

    tool_name: str
    checks: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"[{status}] verification for '{self.tool_name}'"]
        lines.extend(f"  ok   {check}" for check in self.checks)
        lines.extend(f"  FAIL {failure}" for failure in self.failures)
        return "\n".join(lines)
