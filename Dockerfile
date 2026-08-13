FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/output \
    && chown -R appuser:appuser /app

ENV FASTMCP_TRANSPORT=http \
    FASTMCP_HOST=0.0.0.0 \
    FASTMCP_PORT=8000

EXPOSE 8000

USER appuser

ENTRYPOINT ["python", "mcp_server.py"]
