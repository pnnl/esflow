# Local development instructions

Start the webapp by activating the virtual environment and then running `uvicorn app:app --env-file .env --host 127.0.0.1 --port 7932`

Set `WEB_AGENT_MODE` in `.env` to choose which chat agent the web app serves:

- `WEB_AGENT_MODE=planner` for the interactive multistep planner
- `WEB_AGENT_MODE=planner_executor` for the interactive multistep planner-executor

## Running the MCP server

Build the MCP server image from the repository root:

```bash
docker build -t esmflow-mcp .
```

Create the local directories that the container will access:

```bash
mkdir -p data output
```

The container runs as a non-root user with UID 1000. If your host user's UID
is not 1000 (check with `id -u`), run `chmod o+rwx data output` so the container
can write to these bind-mounted directories; the directories contain only local
datasets and generated outputs.

Run the MCP server as a detached HTTP service, passing the same `.env`
configuration used by the web app and mounting local data and output directories:

```bash
docker run -d --name esmflow-mcp \
  --env-file .env \
  -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  esmflow-mcp
```

The `.env` file must include `AI_INCUBATOR_KEY`. The server listens at
`http://127.0.0.1:8000/mcp` and exposes `plan_workflow`, `validate_workflow`,
`execute_workflow`, and `plan_and_execute_workflow`. For separate planning and
execution, pass the complete workflow returned by `plan_workflow` to the next
tool call; the server does not retain workflow state between requests.

The port is published only on the loopback interface because the server has no
authentication. Do not change `-p 127.0.0.1:8000:8000` to a network-accessible
port binding without adding authentication.

Inspect the running server with `docker logs esmflow-mcp`. Stop and remove it
with:

```bash
docker stop esmflow-mcp && docker rm esmflow-mcp
```

After local source changes, including regenerated `tools/tool_catalog.yaml`,
replace the running container before starting the rebuilt image:

```bash
docker stop esmflow-mcp && docker rm esmflow-mcp
docker build -t esmflow-mcp .
```

Then run the `docker run` command above.

For a local MCP client, configure the HTTP endpoint:

```json
{
  "mcpServers": {
    "esmflow": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Here is a sample query you can you to interact with the chatbot

```
Create a workflow to:

1. Extract global QRUNOFF from ELM for years 1985-1989 using the sample.v3.LR.historical case from ./data/e3sm. 
2. Compute the climatological area-weighted global mean
3. Produce a map visualization with statistics overlaid.

Write the workflow as yaml to the output folder
```

## Running evals

From the repository root, run one of:

- `python -m evals.validate_workflow_eval` to assess catalog-valid workflow generation.
- `python evals/workflow_execution_numerical_tolerance_eval.py` to plan, execute, and numerically grade workflows. This requires the local sample data.
