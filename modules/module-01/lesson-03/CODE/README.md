# Lesson 1.3 CODE — Clean, Typed, Consistent

Lesson 1.2 is now formatted, linted, fully annotated, and checked with mypy.
The saved model pipeline also owns the scheduled-departure-hour derivation, so
training and prediction accept the same raw input columns.

## Install, run, and verify

```bash
uv sync --locked
uv run --locked flight-delays
uv run --locked ruff format --check src
uv run --locked ruff check src
uv run --locked mypy src
```

Expected model output:

```text
accuracy: 0.8952
f1:       0.7415
Probability of delay for first test flight: 0.0550
```

The checks must report no formatting changes, `All checks passed!`, and
`Success: no issues found in 10 source files`.

## What changed from Lesson 1.2

| Before | After |
|---|---|
| Training derives `scheduled_departure_hour` before the model | The saved pipeline derives it from raw `CRSDepTime` |
| Callers pass engineered `FEATURES` | Callers pass documented `MODEL_INPUT_COLUMNS` |
| Inline paths and defaults | Typed constants in `config.py` |
| Loose prediction functions | `FlightDelayModel` holds the fitted pipeline |
| No automated style or type checks | Ruff and mypy pass |

DRY here means the scheduled-time rule has one authoritative implementation.
It does not mean every pair of small functions must be merged.

To use another data directory, set `FLIGHT_DATA_DIR` before running the command.
