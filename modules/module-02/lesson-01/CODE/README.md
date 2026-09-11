# Lesson 2.1 CODE — Tracking Fundamentals

The tested Lesson 1.5 pipeline now records every training run with
parameters, metrics, a content-hash fingerprint of the input CSV, and the
Python + dependency versions. A new compare command shows runs sorted by
their metric.

## What's new

```text
src/flight_delays/
  tracking.py    — data/code fingerprints, RunRecorder, compare_runs
tests/
  test_tracking.py — fingerprints, recording, ordering, run recovery
```

`cli.py` gained two command-line flags (`--max-iter`, `--random-state`)
and a `compare` subcommand. The default command is still `run`, which
calls the pipeline, saves the production artifact, and records a run.

## Run and verify

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked mypy src
```

Expect `77 passed` and full quality gates green.

To see the tracking milestone, run the pipeline twice with different
seeds — the same accuracy and F1 as Lesson 1.5, then a second table
row with different numbers:

```bash
uv run flight-delays run
uv run flight-delays run --random-state 7
uv run flight-delays compare
```

## What changed from Lesson 1.5

| Before | After |
|---|---|
| Metrics printed and forgotten | Metrics, params, data fingerprint, and code version recorded as JSON |
| Model overwritten every run | Production artifact unchanged; run copy lives under its own run id |
| No way to compare runs | `compare` prints a table sorted by F1 or accuracy |
| CLI had one hardcoded behaviour | `--max-iter` and `--random-state` flags, plus `compare` subcommand |
