# Tool-only MCP server for esmflow's tool catalog.
#
# Build context must be the esflow-v2/ repo root (not tool_mcp/), since it
# needs both tool_mcp/ and the sibling tools/ directory:
#
#   docker build -f tool_mcp/Dockerfile -t esflow-v2-tool-mcp .
#   docker run -p 8000:8000 esflow-v2-tool-mcp
#
# Only tools/ and tool_mcp/ are copied in -- no agents/, common/, app.py, or
# mcp_server.py -- so this image exposes tools only, with no planning,
# execution, or workflow concepts.

FROM python:3.10-slim

WORKDIR /app

COPY tool_mcp/requirements.txt ./tool_mcp/requirements.txt
RUN pip install --no-cache-dir -r tool_mcp/requirements.txt

COPY tools ./tools
COPY tool_mcp ./tool_mcp

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/output \
    && chown -R appuser:appuser /app

ENV FASTMCP_TRANSPORT=http \
    FASTMCP_HOST=0.0.0.0 \
    FASTMCP_PORT=8000

EXPOSE 8000

USER appuser

ENTRYPOINT ["python", "tool_mcp/server.py"]
