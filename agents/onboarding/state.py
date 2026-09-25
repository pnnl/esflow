"""Deps (shared mutable state) for the onboarding agent.

The agent works on *drafts* held in memory and only touches the filesystem when
the user explicitly approves registration. This mirrors the planner's
``WorkflowState``: one dataclass, mutated in place by tool functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from onboarding.models import CapabilityDraft


@dataclass
class OnboardingState:
    """Drafts under review, keyed by tool name."""

    drafts: Dict[str, CapabilityDraft] = field(default_factory=dict)
    registered: List[str] = field(default_factory=list)
    last_source: Optional[str] = None

    def add_drafts(self, drafts: List[CapabilityDraft]) -> None:
        for draft in drafts:
            self.drafts[draft.tool_name] = draft

    def get(self, tool_name: str) -> CapabilityDraft:
        if tool_name not in self.drafts:
            known = ", ".join(sorted(self.drafts)) or "(none)"
            raise KeyError(
                f"no draft named '{tool_name}'. Scanned drafts: {known}. "
                "Call scan_user_code first."
            )
        return self.drafts[tool_name]

    def rename(self, old: str, new: str) -> None:
        draft = self.drafts.pop(old)
        draft.tool_name = new
        self.drafts[new] = draft

    def context_summary(self) -> str:
        """Short state recap injected into the agent's instructions."""

        if not self.drafts:
            return (
                "No capability drafts are loaded yet. Start by asking the user for a "
                "path to their Python module, then call scan_user_code."
            )
        lines = ["Drafts currently under review:"]
        for name, draft in self.drafts.items():
            status = "registered" if name in self.registered else "pending"
            lines.append(
                f"- {name} [{status}] {draft.subagent}/{draft.category} "
                f"<- {draft.source_module}.{draft.source_function} "
                f"({len(draft.notes)} open note(s))"
            )
        return "\n".join(lines)
