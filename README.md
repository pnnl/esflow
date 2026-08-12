# Local development instructions

Start the webapp by activating the virtual environment and then running `uvicorn app:app --env-file .env --host 127.0.0.1 --port 7932`

Set `WEB_AGENT_MODE` in `.env` to choose which chat agent the web app serves:

- `WEB_AGENT_MODE=planner` for the interactive multistep planner
- `WEB_AGENT_MODE=planner_executor` for the interactive multistep planner-executor

## Running the MCP server

Install the application dependencies, including `fastmcp`, then run the stateless
stdio server from the repository root:

```bash
.venv/bin/python mcp_server.py
```

The server requires the same `.env` configuration as the web app, including
`AI_INCUBATOR_KEY`. It exposes `plan_workflow`, `validate_workflow`,
`execute_workflow`, and `plan_and_execute_workflow`. For separate planning and
execution, pass the complete workflow returned by `plan_workflow` to the next
tool call; the server does not retain workflow state between requests.

For a local MCP client, configure the server as a stdio subprocess:

```json
{
  "mcpServers": {
    "esmflow": {
      "command": "/path/to/esflow-v2/.venv/bin/python",
      "args": ["/path/to/esflow-v2/mcp_server.py"],
      "cwd": "/path/to/esflow-v2"
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
