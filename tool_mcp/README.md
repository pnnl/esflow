# tool_mcp

A tool-only [FastMCP](https://github.com/jlowin/fastmcp) server exposing
every esmflow tool under `tools/{fetchers,loaders,matchers,extractors,
analyzers,plotters}/` as an individual MCP tool.

Unlike `mcp_server.py` at the repo root (planning, validation, and
execution over MCP), this server has **no workflow concept at all** -- one
MCP tool maps 1:1 to one esmflow tool's `run()` function, called directly.

## How tools are exposed

Tools are **not** hand-ported or individually registered here. Every tool
module's `ToolSpec` (declared via `@esmflow_tool` in `tools/core/base.py` --
the same source of truth `tools/generate_catalog.py` reads to produce
`tools/tool_catalog.yaml`) is auto-discovered at server startup, and
`_signature.build_tool_function` synthesizes a real, introspectable Python
function for each one so FastMCP builds a proper per-argument JSON schema
(rather than an opaque `config: dict`).

**Adding a tool under `tools/<category>/` requires no edits in `tool_mcp/`**
-- it appears automatically the next time this server starts.

## Run locally

```bash
python tool_mcp/server.py
```

Set `FASTMCP_TRANSPORT=http FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8000` to serve
over HTTP instead of the default stdio transport (this is what the Docker
image does).

## Test

Standalone suite, independent of the main app's `pytest.ini`:

```bash
python -m pytest tool_mcp/tests
```

## Docker

The build context must be the `esflow-v2/` repo root (not `tool_mcp/`
itself), since the image needs both `tool_mcp/` and the sibling `tools/`
directory:

```bash
docker build -f tool_mcp/Dockerfile -t esflow-v2-tool-mcp .
docker run -p 8000:8000 esflow-v2-tool-mcp
```

Only `tools/` and `tool_mcp/` are copied into the image -- no `agents/`,
`common/`, `app.py`, or `mcp_server.py` -- so this image is tool-only, with
no planning, execution, or workflow code present.
