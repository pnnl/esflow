# Local development instructions

Start the webapp with by activating the virtual environment and then running `uvicorn app:app --env-file .env --host 127.0.0.1 --port 7932`

Here is a sample query you can you to interact with the chatbot

```
Create a workflow to:

1. Extract global QRUNOFF from ELM for years 1985-1989 using the sample.v3.LR.historical case from ./data/e3sm. 
2. Compute the climatological area-weighted global mean
3. Produce a map visualization with statistics overlaid.

Write the workflow as yaml to the output folder
```

Running evals

From the root run `python -m evals.evals`