"""Tests for the tool-only MCP server (tool_mcp/server.py).

Standalone suite -- run via ``python -m pytest tool_mcp/tests`` from
``esflow-v2/``, not wired into the main app's ``pytest.ini`` testpaths.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOL_MCP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOL_MCP_DIR.parent
TOOLS_DIR = REPO_ROOT / "tools"

if str(TOOL_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_MCP_DIR))

import server  # noqa: E402


CATEGORIES = ["fetchers", "loaders", "matchers", "extractors", "analyzers", "plotters"]


def _tool_files_on_disk() -> list[str]:
    names = []
    for category in CATEGORIES:
        category_dir = TOOLS_DIR / category
        if not category_dir.exists():
            continue
        for py_file in category_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            names.append(py_file.stem)
    return names


def test_all_tools_on_disk_are_registered():
    """Every non-private .py file under a category dir becomes an MCP tool.

    Computed from disk rather than hardcoded, so it can't silently drift the
    way a hardcoded count (like esflow-tool-mcp's ``assert len(TOOLS) == 24``)
    can when a tool is added or removed.
    """
    disk_names = set(_tool_files_on_disk())
    tools = asyncio.run(server.mcp.list_tools())
    registered_names = {t.name for t in tools}

    assert registered_names == disk_names


def test_every_tool_has_a_populated_input_schema():
    tools = asyncio.run(server.mcp.list_tools())
    assert tools, "expected at least one registered tool"

    for tool in tools:
        mcp_tool = tool.to_mcp_tool()
        assert mcp_tool.name == tool.name
        schema = mcp_tool.inputSchema
        assert schema, f"{tool.name} has an empty inputSchema"
        assert "output_dir" in schema.get("properties", {}), (
            f"{tool.name} is missing the trailing output_dir parameter"
        )


def test_tool_name_matches_module_stem():
    disk_names = set(_tool_files_on_disk())
    tools = asyncio.run(server.mcp.list_tools())
    for tool in tools:
        assert tool.name in disk_names


def test_compute_metrics_end_to_end(tmp_path):
    """Smoke-test that the generated wrapper actually threads args into run().

    compute_metrics's own logic is already unit-tested under
    esflow-v2/tests/tools; this only proves the dynamic MCP wrapper built by
    _signature.build_tool_function correctly maps kwargs -> config -> run().
    """
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    sim = pd.DataFrame({"g1": range(24)}, index=idx)
    obs = pd.DataFrame({"g1": range(1, 25)}, index=idx)

    sim_file = tmp_path / "sim.csv"
    obs_file = tmp_path / "obs.csv"
    sim.to_csv(sim_file)
    obs.to_csv(obs_file)

    result = asyncio.run(
        server.mcp.call_tool(
            "compute_metrics",
            {
                "sim_file": str(sim_file),
                "obs_file": str(obs_file),
                "output_dir": str(tmp_path),
            },
        )
    )

    assert result.structured_content["n_gauges"] == 1
    metrics_file = Path(result.structured_content["metrics_file"])
    assert metrics_file.exists()
    assert metrics_file.parent == tmp_path


def test_compute_metrics_error_path(tmp_path):
    """Bad input raises rather than returning a silent error dict."""
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(Exception):
        asyncio.run(
            server.mcp.call_tool(
                "compute_metrics",
                {
                    "sim_file": str(missing_file),
                    "obs_file": str(missing_file),
                    "output_dir": str(tmp_path),
                },
            )
        )
