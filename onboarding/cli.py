"""Command-line entrypoint for the onboarding machinery.

The onboarding *agent* wraps these same functions as pydantic-ai tools; the CLI
exists so the pipeline can be driven (and scripted, and tested) without a model
in the loop.

    python -m onboarding.cli scan examples/user_code/drought_indices.py
    python -m onboarding.cli register examples/user_code/drought_indices.py \
        --function compute_standardized_anomaly_index --subagent diagnostics
    python -m onboarding.cli list
    python -m onboarding.cli verify compute_standardized_anomaly_index
    python -m onboarding.cli remove compute_standardized_anomaly_index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onboarding.introspect import category_for_subagent, scan_source  # noqa: E402
from onboarding.models import CapabilityDraft  # noqa: E402
from onboarding.registry import (  # noqa: E402
    RegistryError,
    register_capability,
    register_subagent,
    registry_summary,
    remove_capability,
    subagent_draft_for,
)
from onboarding.scaffold import (  # noqa: E402
    render_adapter,
    render_workflow_snippet,
    validate_draft,
)
from onboarding.verify import verify_capability  # noqa: E402


def _cmd_scan(args: argparse.Namespace) -> int:
    result = scan_source(args.source, function_name=args.function)
    print(result.summary())
    print()
    for draft in result.drafts:
        print(draft.summary())
        print("-" * 72)
    if args.show_adapter:
        for draft in result.drafts:
            problems = validate_draft(draft)
            if problems:
                print(f"# cannot render {draft.tool_name}: {'; '.join(problems)}")
                continue
            print(render_adapter(draft))
            print("-" * 72)
    return 0


def _apply_overrides(draft: CapabilityDraft, args: argparse.Namespace) -> CapabilityDraft:
    if args.tool_name:
        draft.tool_name = args.tool_name
    if args.description:
        draft.description = args.description
    if args.subagent:
        draft.subagent = args.subagent
        draft.category = args.category or category_for_subagent(args.subagent)
    elif args.category:
        draft.category = args.category
    return draft


def _cmd_register(args: argparse.Namespace) -> int:
    result = scan_source(args.source, function_name=args.function)
    if not result.drafts:
        print(f"No onboardable functions found in {args.source}.", file=sys.stderr)
        return 1
    if len(result.drafts) > 1 and not args.all:
        print(
            f"{args.source} defines {len(result.drafts)} candidate functions; pass "
            "--function NAME to pick one or --all to register every one.",
            file=sys.stderr,
        )
        print(result.summary(), file=sys.stderr)
        return 1

    drafts: List[CapabilityDraft] = result.drafts
    if len(drafts) == 1:
        drafts = [_apply_overrides(drafts[0], args)]
    else:
        drafts = [_apply_overrides(draft, args) for draft in drafts]

    if args.new_subagent:
        for draft in drafts:
            draft.subagent = args.new_subagent
            draft.category = args.category or args.new_subagent
        register_subagent(
            subagent_draft_for(args.new_subagent, description=args.subagent_description)
        )
        print(f"Registered new subagent '{args.new_subagent}'.")

    exit_code = 0
    for draft in drafts:
        try:
            entry, catalog_output = register_capability(
                draft, overwrite=args.overwrite, regenerate=not args.no_catalog
            )
        except RegistryError as exc:
            print(f"[FAIL] {draft.tool_name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"[ok] registered '{entry.tool_name}' -> {entry.adapter_path}")
        if catalog_output:
            print(catalog_output)
        if not args.no_verify:
            report = verify_capability(entry.tool_name)
            print(report.summary())
            if not report.ok:
                exit_code = 1
        print()
        print("Example workflow step:")
        print(render_workflow_snippet(draft))
    return exit_code


def _cmd_list(args: argparse.Namespace) -> int:
    print(registry_summary())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    if args.tool_name:
        reports = [verify_capability(args.tool_name)]
    else:
        from onboarding.verify import verify_all

        reports = verify_all()
    if not reports:
        print("Nothing to verify: no capabilities are registered.")
        return 0
    failed = False
    for report in reports:
        print(report.summary())
        failed = failed or not report.ok
    return 1 if failed else 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Drive the self-cleaning onboarding demonstration."""

    from onboarding import demo as demo_mod

    action = args.action
    try:
        if action == "show":
            print(demo_mod.demo_overview())
            print()
            print(demo_mod.where_code_goes())
            print()
            print(demo_mod.demo_draft().summary())
            return 0
        if action == "status":
            print(demo_mod.demo_status().summary())
            return 0
        if action == "start":
            print(demo_mod.start_demo().summary())
            return 0
        if action == "run":
            result = demo_mod.run_demo()
            print(result.summary())
            return 0 if result.ok else 1
        if action == "end":
            print(demo_mod.end_demo().summary())
            return 0
        if action == "all":
            # The whole point of the demo is that it leaves no trace, so the
            # offboarding step runs even if the run stage fails.
            print(demo_mod.demo_overview())
            print()
            print(demo_mod.start_demo().summary())
            print()
            try:
                result = demo_mod.run_demo()
                print(result.summary())
            finally:
                print()
                print(demo_mod.end_demo().summary())
            return 0 if result.ok else 1
    except demo_mod.DemoError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[FAIL] unknown demo action: {action}", file=sys.stderr)  # pragma: no cover
    return 1


def _cmd_remove(args: argparse.Namespace) -> int:
    try:
        entry = remove_capability(
            args.tool_name,
            delete_adapter=not args.keep_adapter,
            regenerate=not args.no_catalog,
        )
    except RegistryError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[ok] removed '{entry.tool_name}' ({entry.adapter_path})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m onboarding.cli",
        description="Onboard user code as ESMFlow capabilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="introspect a module without writing anything")
    scan.add_argument("source", help="path to a .py file or an importable module path")
    scan.add_argument("--function", help="only scan this function")
    scan.add_argument(
        "--show-adapter",
        action="store_true",
        help="also print the adapter source that would be generated",
    )
    scan.set_defaults(func=_cmd_scan)

    reg = sub.add_parser("register", help="scan, scaffold, register and verify")
    reg.add_argument("source", help="path to a .py file or an importable module path")
    reg.add_argument("--function", help="function to onboard")
    reg.add_argument("--all", action="store_true", help="onboard every public function")
    reg.add_argument("--tool-name", help="override the generated tool name")
    reg.add_argument("--description", help="override the tool description")
    reg.add_argument("--subagent", help="planner category to attach the tool to")
    reg.add_argument("--category", help="tools/<category>/ directory override")
    reg.add_argument("--new-subagent", help="create this subagent before registering")
    reg.add_argument("--subagent-description", default="", help="new subagent blurb")
    reg.add_argument("--overwrite", action="store_true", help="replace an existing tool")
    reg.add_argument("--no-catalog", action="store_true", help="skip catalog regeneration")
    reg.add_argument("--no-verify", action="store_true", help="skip verification")
    reg.set_defaults(func=_cmd_register)

    listing = sub.add_parser("list", help="show the extension registry")
    listing.set_defaults(func=_cmd_list)

    ver = sub.add_parser("verify", help="check one or all registered capabilities")
    ver.add_argument("tool_name", nargs="?", help="omit to verify everything")
    ver.set_defaults(func=_cmd_verify)

    dem = sub.add_parser(
        "demo",
        help="run the self-cleaning onboarding demonstration",
        description=(
            "Demonstrate onboarding end to end with the bundled exemplar: "
            "register a tool, run it on synthesized data, then un-register it "
            "so the demonstration is repeatable."
        ),
    )
    dem.add_argument(
        "action",
        choices=["show", "status", "start", "run", "end", "all"],
        help=(
            "show: explain the format and where code goes (no writes); "
            "status: report demo state; start: register; run: execute; "
            "end: un-register; all: the full cycle, always cleaning up"
        ),
    )
    dem.set_defaults(func=_cmd_demo)

    rm = sub.add_parser("remove", help="un-register a capability")
    rm.add_argument("tool_name")
    rm.add_argument("--keep-adapter", action="store_true", help="leave the .py in place")
    rm.add_argument("--no-catalog", action="store_true", help="skip catalog regeneration")
    rm.set_defaults(func=_cmd_remove)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
