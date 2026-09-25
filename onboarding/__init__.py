"""Onboarding system: turn user Python into ESMFlow capabilities.

Pipeline::

    scan_source()  ->  CapabilityDraft(s)      (onboarding.introspect)
    write_adapter() ->  tools/<cat>/<name>.py  (onboarding.scaffold)
    register_capability() -> extensions/registry.yaml + regenerated catalog
                                               (onboarding.registry)
    verify_capability() -> import/catalog/Literal/validation checks
                                               (onboarding.verify)

The same operations are exposed as agent tools (``agents/onboarding``) and as a
CLI (``python -m onboarding.cli``).
"""

from onboarding.models import (
    CapabilityDraft,
    OutputDraft,
    ParamDraft,
    RegistryEntry,
    ScanResult,
    SubagentDraft,
    VerificationReport,
)

__all__ = [
    "CapabilityDraft",
    "OutputDraft",
    "ParamDraft",
    "RegistryEntry",
    "ScanResult",
    "SubagentDraft",
    "VerificationReport",
]
