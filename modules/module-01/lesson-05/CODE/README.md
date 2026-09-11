# Lesson 1.5 CODE — Automated Testing

The Lesson 1.4 pipeline now has unit, integration, and command-boundary tests.
The suite uses synthetic raw flight rows, so it is fast and does not depend on
the downloaded course CSV.

## What's tested

```text
tests/
  conftest.py       — raw rows, a synthetic training set, and a fitted model
  test_clean.py     — cohort filtering
  test_features.py  — HHMM-to-hour derivation and pipeline shape
  test_config.py    — raw-input and engineered-feature contracts
  test_load.py      — CSV loading
  test_model.py     — metrics, save/load, and class-1 probabilities
  test_pipeline.py  — raw data → clean → split → train → save → load → predict
  test_cli.py       — real temporary CSV/model paths and failure boundaries
  test_train.py     — model fitting
  test_validate.py  — files, columns, rows, and valid HHMM values
```

## Run and verify

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked mypy src
```

Expect `64 passed` and about `98%` total coverage. The simple
`__main__.py` launcher is the only intentionally uncovered module; CLI
orchestration and logging setup are exercised. Coverage says which lines ran,
not whether the assertions are useful, so review both the percentage and the
test behavior.

The integration test keeps one synthetic dataset from raw input through the
loaded model. It does not replace cleaned rows with a separate fixture halfway
through. The model test also verifies that `predict_delay()` returns the
probability for target class `1`, not merely a number between zero and one.

## What changed from Lesson 1.4

| Before | After |
|---|---|
| Manual reruns only | 64 repeatable tests |
| No reusable test data | Function-scoped raw rows and session-scoped trained fixtures |
| Edge cases checked one at a time | Parametrized valid and invalid cases |
| No command-boundary checks | Temporary paths test success, expected failure, and unexpected errors |
| No coverage feedback | Per-module coverage with missing lines |
