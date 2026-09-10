# Lesson 1.5 CODE — Automated Testing

The Lesson 1.4 pipeline now has a full `pytest` suite: cleaning units, schema
checks, the model's prediction contract, and a pipeline integration run on
synthetic data — backed by fixtures, parametrized cases, and a coverage
report. The pipeline itself is unchanged: same commands, same metrics.

## What's here

```text
src/
  flight_delays/            — the pipeline from Lesson 1.4, unchanged
tests/                      — the new test suite
  conftest.py               — shared fixtures (raw frame, synthetic set, fitted model)
  test_clean.py             — cleaning units: filters, the derived hour column
  test_config.py            — the feature/schema lists are internally consistent
  test_features.py          — build_model returns the promised pipeline shape
  test_load.py              — load_csv reads a CSV into a DataFrame
  test_model.py             — save/load round-trip, prediction contract, metrics
  test_pipeline.py          — clean → split → train → evaluate → save → load → predict
  test_train.py             — train_model fits a model object
  test_validate.py          — file, column, and non-empty checks (pytest.raises)
```

## Install and run

```bash
uv sync
uv run flight-delays       # same metrics as Lesson 1.4
```

## The milestone — automated tests

```bash
uv run pytest
```

Expect `43 passed` plus a coverage table. Add `-v` to see every test name and
its parametrized cases. Coverage and the tests re-run in one command because
`pyproject.toml` sets `--cov=flight_delays --cov-report=term-missing` as the
pytest default.

The big headline number is `56%` — but read the table carefully. Every module
that owns pipeline *logic* is `100%`; the `0%` rows are the entry point
(`cli.py`), the logging setup, and `__main__.py` — code that reads environment
variables and writes to disk. Coverage counts lines that ran, not behavior
that is verified. No test asserts the exact accuracy value, because the model
is retrained; the suite pins behavior, not numbers.

## Checks

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
```

All must be green. Ruff now covers the tests too — test files are code.

## What changed from Lesson 1.4

| Before (L1.4) | After (L1.5) |
|---|---|
| No tests — every change was a gamble | 43 pytest tests, all green in under a second |
| No fixtures or reusable test data | `tests/conftest.py` shares a raw frame, a synthetic modeling set, and a fitted model |
| Each edge case asserted by hand | `@pytest.mark.parametrize` runs the same check over many inputs |
| No coverage report | `uv run pytest` prints a per-module coverage table |
| Ruff + mypy on `src` only | Ruff also checks `tests/` |