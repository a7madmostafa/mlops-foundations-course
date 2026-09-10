# Lesson 1.4 CODE — Robustness and Logging

The installed Lesson 1.3 package now checks expected failure conditions and
uses standard-library logging with levels and timestamps. Successful runs keep
the same model and metrics; expected failures produce one useful error and a
non-zero exit code.

## What's here

```text
src/
  flight_delays/
    __init__.py
    __main__.py        — supports `python -m flight_delays`
    cli.py             — guarded, logged command and orchestrator
    config.py          — paths, features, target, raw schema, defaults
    load.py            — load_csv(path) -> DataFrame
    clean.py           — clean_flights(df, target) -> DataFrame
    features.py        — build_model(...) -> Pipeline
    train.py           — train_model(...), save_model(...)
    evaluate.py        — evaluate_model(...) -> dict
    predict.py         — FlightDelayModel: load() + predict_delay()
    validate.py        — file, raw-column, and non-empty checks
    log_setup.py       — one logging format for the package
```

## Install and run

```bash
uv sync
uv run flight-delays
```

The equivalent module command is `uv run python -m flight_delays`.

Expected output (timestamps vary):

```text
2026-09-10 16:17:16 | INFO | flight_delays.cli | Pipeline started
2026-09-10 16:17:16 | INFO | flight_delays.cli | Loading data from .../data/flight_delays_2025_01.csv
2026-09-10 16:17:19 | INFO | flight_delays.cli | Modeling rows: 522,269
2026-09-10 16:17:23 | INFO | flight_delays.cli | Training finished
2026-09-10 16:17:23 | INFO | flight_delays.cli | accuracy: 0.8952
2026-09-10 16:17:23 | INFO | flight_delays.cli | f1:       0.7415
2026-09-10 16:17:23 | INFO | flight_delays.cli | Saved pipeline to .../models/model_2025_01.joblib
2026-09-10 16:17:23 | INFO | flight_delays.cli | Probability of delay for first test flight: 0.0550
2026-09-10 16:17:23 | INFO | flight_delays.cli | Pipeline finished
```

## Expected breakage fails loudly

Point the pipeline at a directory with no CSV:

```bash
# macOS / Linux
FLIGHT_DATA_DIR=/no/such/dir uv run flight-delays

# Windows PowerShell
$env:FLIGHT_DATA_DIR="C:\no\such\dir"; uv run flight-delays
```

You get one `ERROR` line without a traceback and an exit code of `1`. Run
`echo $?` on macOS/Linux or `$LASTEXITCODE` in PowerShell to inspect it.

## Checks — the milestone

```bash
uv run ruff format --check src
uv run ruff check src
uv run mypy src
```

All three must be green. This lesson changed failure behavior and output, not
the model or data.

## What changed from Lesson 1.3

| Before (L1.3) | After (L1.4) |
|---|---|
| Missing CSV produces a raw traceback | One clear `ERROR` line + exit code `1` |
| Cleaning can fail before a useful schema message | Raw columns are checked before cleaning |
| No check on the cleaned row count | `ensure_nonempty()` raises a targeted `ValueError` |
| `print()` for every message | Logging with levels, timestamps, and logger names |
| No single logging configuration | `setup_logging()` defines one package-wide format |
