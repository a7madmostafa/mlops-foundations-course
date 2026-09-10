# Lesson 1.4 CODE — Robustness & Logging

The Lesson 1.3 project made robust: expected breakage fails loudly with a
clear message, and prints are replaced by stdlib logging with levels and
timestamps, so every run leaves a diagnosable trail. Same pipeline, same
metrics.

## What's here

```
main.py            — entry point: guarded, logged orchestrator
src/
  config.py        — paths, features, target, training defaults
  load.py          — load_csv(path) -> DataFrame
  clean.py         — clean_flights(df) -> DataFrame
  features.py      — build_model(...) -> Pipeline (all preprocessing inside)
  train.py         — train_model(...), save_model(...)
  evaluate.py      — evaluate_model(...) -> dict
  predict.py       — FlightDelayModel: load() + predict_delay()
  validate.py      — ensure_file_exists / ensure_columns / ensure_nonempty
  log_setup.py     — setup_logging(): one format for every logger
```

## Install & run

```bash
uv sync                    # create venv + install all deps (incl. dev tools)
uv run python main.py      # same metrics as L1.3, but as timestamped log lines
```

Expected output (timestamps vary):

```
2026-09-10 16:17:16 | INFO | __main__ | Pipeline started
2026-09-10 16:17:16 | INFO | __main__ | Loading data from .../data/flight_delays_2025_01.csv
2026-09-10 16:17:19 | INFO | __main__ | Modeling rows: 522,269
2026-09-10 16:17:23 | INFO | __main__ | accuracy: 0.8952
2026-09-10 16:17:23 | INFO | __main__ | f1:       0.7415
2026-09-10 16:17:23 | INFO | __main__ | Probability of delay for first test flight: 0.0550
2026-09-10 16:17:23 | INFO | __main__ | Pipeline finished
```

## Expected breakage fails loudly

Point the pipeline at a data directory with no CSV:

```bash
# macOS / Linux
FLIGHT_DATA_DIR=/no/such/dir uv run python main.py

# Windows PowerShell
$env:FLIGHT_DATA_DIR="C:\no\such\dir"; uv run python main.py
```

You get one ERROR line (no traceback) and a non-zero exit code:

```
INFO  | __main__ | Pipeline started
ERROR | __main__ | Pipeline failed: Data file not found: ... . Download the flight-delays CSV first, then run again. See the download command in data/README.md.
```

`echo $?` (macOS/Linux) or `$LASTEXITCODE` (PowerShell) prints `1`.

## Checks — the milestone

```bash
uv run ruff format --check src main.py   # formatting is consistent
uv run ruff check src main.py            # linting is clean
uv run mypy src main.py                  # types are statically verified
```

All three green. The metrics are the L1.3 numbers: accuracy `0.8952`,
F1 `0.7415` — this lesson changed failure behaviour and output, not the
model or the data.

## What changed from Lesson 1.3

| Before (L1.3) | After (L1.4) |
|---|---|
| Missing CSV → raw Python traceback | One clear ERROR line + exit code 1 |
| No checks on columns or row count | `ensure_columns`, `ensure_nonempty` raise `ValueError` |
| `print()` for every message | `logging` with levels, timestamps, logger names |
| No single logging configuration | `setup_logging()` in `log_setup.py`, one format |