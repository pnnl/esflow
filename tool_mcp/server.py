"""Tool-only FastMCP server exposing every esmflow tool as an individual MCP tool.

Unlike ``mcp_server.py`` at the repo root (which exposes planning/validation/
execution over MCP), this server has no concept of workflows, planning, or
execution -- it is a thin, auto-generated MCP wrapper directly over
``tools/{fetchers,loaders,matchers,extractors,analyzers,plotters}/*.py``, one
MCP tool per esmflow tool.

Tools are *not* hand-ported or individually registered. Every tool module's
``ToolSpec`` (declared via ``@esmflow_tool`` in ``tools/core/base.py`` -- the
same source of truth ``tools/generate_catalog.py`` reads to build
``tools/tool_catalog.yaml``) is auto-discovered, and ``_signature.build_tool_function``
synthesizes a real, introspectable Python function for each one so FastMCP
can build a proper per-argument JSON schema. Adding a new tool under
``tools/<category>/`` requires no edits here -- it appears automatically the
next time this server starts.

Docker configures FastMCP to serve HTTP; direct execution retains FastMCP's
default stdio transport.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"

# Same category list ``tools/generate_catalog.py`` scans.
TOOL_CATEGORIES = ["fetchers", "loaders", "matchers", "extractors", "analyzers", "plotters"]

# Tool modules do `sys.path.insert(0, str(Path(__file__).parent.parent))` and
# `from core.base import ...`, i.e. they expect `tools/` itself on sys.path
# (not the repo root) so `core.base` resolves to a single module instance.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _signature import build_tool_function  # noqa: E402  (needs sys.path tweak above)


def discover_tool_modules() -> None:
    """Import every tool module so it registers its ``ToolSpec`` in ``TOOL_REGISTRY``."""
    for category in TOOL_CATEGORIES:
        category_dir = TOOLS_DIR / category
        if not category_dir.exists():
            continue
        for py_file in sorted(category_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"{category}.{py_file.stem}"
            importlib.import_module(module_name)


discover_tool_modules()

from core.base import TOOL_REGISTRY  # noqa: E402  (populated by discover_tool_modules above)


mcp = FastMCP(
    "ESFlow Tools (v2)",
    instructions=(
        "Direct, stateless access to esmflow's Earth System Model (ESM) analysis "
        "tools: fetch/load data, match gauges to grids, extract time series and "
        "gridded fields, compute validation metrics and diagnostics, and produce "
        "plots. This server has no planning or workflow concept -- each MCP tool "
        "maps 1:1 to one esmflow tool's run() function."
    ),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Liveness probe for HTTP transport; not part of the MCP protocol itself."""
    return JSONResponse({"status": "ok"})


def _register_all_tools() -> None:
    for name, spec in TOOL_REGISTRY.items():
        category_module = None
        for category in TOOL_CATEGORIES:
            if (TOOLS_DIR / category / f"{name}.py").exists():
                category_module = f"{category}.{name}"
                break
        if category_module is None:
            continue

        module = importlib.import_module(category_module)
        run_fn = getattr(module, "run")
        wrapper = build_tool_function(name, spec, run_fn)
        mcp.tool(wrapper, name=name, description=spec.description)


_register_all_tools()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
