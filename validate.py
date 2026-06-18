from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import yaml

from workflow import Workflow


def _resolve_validation_cli() -> tuple[Path, Path] | None:
    """Resolve run_workflow.py location and its working directory."""
    this_repo_root = Path(__file__).resolve().parent
    workspace_root = this_repo_root.parent
    cli_path = workspace_root / "esflow-gmd-v1" / "run_workflow.py"
    return cli_path, cli_path.parent


def _extract_validation_errors(stdout: str) -> List[str]:
    errors: List[str] = []
    lines = stdout.splitlines()
    capture = False

    for line in lines:
        stripped = line.strip()
        if stripped == "VALIDATION ERRORS:":
            capture = True
            continue

        if capture:
            if stripped.startswith("-"):
                errors.append(stripped.lstrip("-").strip())
            elif stripped.startswith("Workflow validation passed"):
                break
            elif stripped == "":
                continue
            elif stripped.startswith("["):
                continue
            else:
                # Any other content after entering capture ends the error block.
                break

    return errors


def validate_with_cli(workflow: Workflow) -> Tuple[bool, List[str]]:
    """Validate a workflow by serializing to YAML and invoking run_workflow.py --dry-run."""
    cli_path, cli_repo_root = _resolve_validation_cli()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        yaml.safe_dump(
            workflow.to_yaml_dict(),
            tmp_file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    try:
        proc = subprocess.run(
            [sys.executable, str(cli_path), str(tmp_path), "--dry-run"],
            cwd=str(cli_repo_root),
            capture_output=True,
            text=True,
        )

        errors = _extract_validation_errors(proc.stdout)
        if errors:
            return False, errors

        if proc.returncode != 0:
            stderr_text = proc.stderr.strip()
            if stderr_text:
                return False, [stderr_text]
            return False, ["Workflow validation failed with non-zero exit code."]

        return True, []
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
