# Lesson 2.2 CODE — MLflow Tracking

The Lesson 2.1 run records now live in an MLflow tracking store instead
of a hand-rolled JSONL file. Every flight-delay training run is logged
with its parameters, metrics, a content-hash fingerprint of the input
CSV, and the code + environment stamps. Runs are compared and picked
from the MLflow UI or via the same `compare` command.

## What's new

```text
src/flight_delays/
  tracking.py    — MLflow-backed record_run + compare_runs; fingerprints stay
tests/
  test_tracking.py — fingerprints + MLflow logging/searching
```

`runs/runs.jsonl` and `RunRecorder` are gone. MLflow stores run metadata
in the local SQLite database `mlflow.db`, and trained artifacts under
`mlruns/` (both gitignored). The `run` and `compare` commands behave
identically to Lesson 2.1, but the compare table is now powered by
MLflow's `search_runs`.

## Run and verify

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked mypy src
```

Expect `78 passed` and full quality gates green.

To see the tracking milestone, run the pipeline twice with different
seeds, then compare:

```bash
uv run flight-delays run
uv run flight-delays run --random-state 7
uv run flight-delays compare
```

Point a browser at each logged run: start the MLflow UI from the same
directory and open http://localhost:5000:

```bash
uv run --locked mlflow server --backend-store-uri sqlite:///mlflow.db
```

## What changed from Lesson 2.1

| Before | After |
|---|---|
| A JSON file, one line per run (`runs/runs.jsonl`) | An MLflow tracking store (`mlflow.db` + `mlruns/`), browsable in the UI |
| Run ids were timestamps (`20260911_120340_694107`) | MLflow assigns each run a 32-character id |
| Comparing meant a hand-written reader | `compare` uses MLflow's `search_runs` query API |
| No default UI | `mlflow server` gives every run a page: params, metrics, tags, artifact |