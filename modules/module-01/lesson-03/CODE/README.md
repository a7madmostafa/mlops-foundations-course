# Lesson 1.3 CODE — Clean, Typed, Consistent

The installed Lesson 1.2 package is now refactored: single-responsibility
modules, one pipeline containing all preprocessing, a model object for
prediction, and formatting plus typing enforced by Ruff and mypy. The model
behavior and metrics stay the same.

## What's here

```text
src/
  flight_delays/
    __init__.py
    __main__.py        — supports `python -m flight_delays`
    cli.py             — installed command and pipeline orchestrator
    config.py          — paths, features, target, training defaults (typed)
    load.py            — load_csv(path) -> DataFrame
    clean.py           — clean_flights(df, target) -> DataFrame
    features.py        — build_model(...) -> Pipeline
    train.py           — train_model(...), save_model(...)
    evaluate.py        — evaluate_model(...) -> dict
    predict.py         — FlightDelayModel: load() + predict_delay()
```

## Install and run

```bash
uv sync
uv run flight-delays
```

The equivalent module command is `uv run python -m flight_delays`.

Expected output:

```text
accuracy: 0.8952
f1:       0.7415

Probability of delay for first test flight: 0.0550
```

## Checks — the milestone

```bash
uv run ruff format --check src
uv run ruff check src
uv run mypy src
```

All three must be green. Nothing changed the numbers: accuracy `0.8952` and
F1 `0.7415`; the refactor changed structure, not behavior.

## Development dependencies

`ruff`, `mypy`, and `pandas-stubs` live in the `[dependency-groups]` `dev`
group in `pyproject.toml`. They are development tools, not runtime
requirements. `uv run <command>` runs them inside the project environment
without a manual `activate` step.

## Customize the data path

```bash
FLIGHT_DATA_DIR=/path/to/data uv run flight-delays
```

## What changed from Lesson 1.2

| Before (L1.2) | After (L1.3) |
|---|---|
| `build_preprocessor()` + `build_model()` as two calls | `build_model()` owns all preprocessing |
| `joblib.dump` inside `cli.py` | `save_model()` in `train.py` |
| Free-form prediction functions | `FlightDelayModel` wraps the fitted pipeline |
| Target repeated as a string | `TARGET` is passed from `cli.py` into cleaning and training |
| No typing | Every function annotated; `mypy` passes |
| No formatting/linting | `ruff format` + `ruff check` pass |
