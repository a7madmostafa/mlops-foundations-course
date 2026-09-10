# Lesson 1.3 CODE — Clean, Typed, Consistent

The Lesson 1.2 project, refactored: single-responsibility modules, a
pipeline that bundles all preprocessing, a model object for prediction,
and formatting + typing enforced by Ruff and mypy. Same pipeline, same
metrics.

## What's here

```
main.py            — entry point: orchestrates the full pipeline
src/
  config.py        — paths, features, target, training defaults (typed)
  load.py          — load_csv(path) -> DataFrame
  clean.py         — clean_flights(df) -> DataFrame
  features.py      — build_model(...) -> Pipeline (all preprocessing inside)
  train.py         — train_model(...), save_model(...)
  evaluate.py      — evaluate_model(...) -> dict
  predict.py       — FlightDelayModel: load() + predict_delay()
```

## Install & run

```bash
uv sync                    # create venv + install all deps (incl. dev tools)
uv run python main.py      # same metrics as L1.2
```

Expected output:

```
accuracy: 0.8952
f1:       0.7415

Probability of delay for first test flight: 0.0550
```

## Checks — the milestone

```bash
uv run ruff format --check src main.py   # formatting is consistent
uv run ruff check src main.py            # linting is clean
uv run mypy src main.py                  # types are statically verified
```

All three must be green. Nothing changed the numbers: accuracy `0.8952`,
F1 `0.7415` — the refactor only touched structure, not behavior.

## Dev dependencies

`ruff`, `mypy`, and `pandas-stubs` live in the `[dependency-groups]` `dev`
group in `pyproject.toml` — they are development tools, not runtime
requirements. `uv run <command>` runs them inside the project environment
without a manual `activate`.

## Customise the data path

```bash
FLIGHT_DATA_DIR=/path/to/data uv run python main.py
```

## What changed from Lesson 1.2

| Before (L1.2) | After (L1.3) |
|---|---|
| `build_preprocessor()` + `build_model()` as two calls | `build_model()` — one pipeline owns all preprocessing (DRY) |
| `joblib.dump` inline in `main.py` | `save_model()` in `train.py`, `FlightDelayModel.load()` for prediction |
| Free-form `predict.py` functions | `FlightDelayModel` class wrapping the fitted `Pipeline` |
| Magic strings in `main.py` | `DATA_FILE`, `MODEL_FILE`, `TARGET`, `FEATURES` in `config.py` |
| No typing | Every function annotated; `mypy` passes |
| No formatting/linting | `ruff format` + `ruff check` pass |