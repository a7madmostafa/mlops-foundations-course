# Lesson 1.4 CODE — Robustness and Logging

The command now validates expected input failures and reports them with
standard-library logging. Valid runs retain the same model behavior; expected
bad inputs produce one useful error and exit code `1`.

## Install and run

```bash
uv sync --locked
uv run --locked flight-delays
```

Logs include a timestamp, severity, and logger name. They are written to the
console; this lesson does not yet create persistent run history.

## Expected failures

`DataValidationError` represents input problems the learner can fix: missing
columns, no usable rows, or invalid scheduled times. `CRSDepTime` must be a
whole HHMM value from `0000` through `2359`, with minutes below `60`;
values such as `1260`, `2400`, and `2500` are rejected rather than
silently clipped.

```bash
# macOS / Linux
FLIGHT_DATA_DIR=/no/such/dir uv run --locked flight-delays

# Windows PowerShell
$env:FLIGHT_DATA_DIR="C:\no\such\dir"; uv run --locked flight-delays
```

The command catches only missing-file and data-validation failures. Unexpected
programming or model errors still show a traceback, preserving debugging
information.

## Verify

```bash
uv run --locked ruff format --check src
uv run --locked ruff check src
uv run --locked mypy src
```

`main(data_dir=..., model_dir=...)` also accepts explicit paths and returns
its metrics. That small boundary makes Lesson 1.5 tests independent of the
real course data and model directories.

## What changed from Lesson 1.3

| Before | After |
|---|---|
| Missing CSV produces a raw traceback | One clear error and exit code `1` |
| Raw columns and times are trusted | Schema, row count, and HHMM values are validated |
| Expected and unexpected `ValueError`s can look alike | Only `DataValidationError` is handled as user-correctable |
| `print()` is the only output | Logs add time, severity, and source |
