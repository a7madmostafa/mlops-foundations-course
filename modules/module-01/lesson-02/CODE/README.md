# Lesson 1.2 CODE — Notebook to Installed Package

The flight-delay notebook is now an installable Python package. Its modules
can be imported without depending on the current working directory, and one
installed command reproduces the baseline.

## What's here

```text
pyproject.toml
src/
  flight_delays/       — the import package
    __init__.py
    __main__.py        — supports `python -m flight_delays`
    cli.py             — orchestrates the full pipeline
    config.py          — paths, feature lists, training defaults
    load.py            — load_csv()
    clean.py           — clean_flights()
    features.py        — build_preprocessor(), build_model()
    train.py           — train_model()
    evaluate.py        — evaluate_model()
    predict.py         — load_model(), predict_delay()
```

`src/` is the source-layout boundary; `flight_delays/` is the package you
import. Installing the project makes imports predictable, gives learners a
stable command, and prepares the same code for later tests and services.

## Install and run

Make sure the shared data is present (see `data/README.md` at the repository
root). Then run these commands from this folder:

```bash
uv sync --locked
uv run --locked flight-delays
```

`uv sync --locked` installs the project itself as well as its dependencies
without rewriting the committed lockfile. The
`flight-delays` command is declared in `pyproject.toml`. This equivalent form
also works:

```bash
uv run python -m flight_delays
```

You should see:

```text
accuracy: 0.8952
f1:       0.7415

Predicted delay probability for the first test flight: 0.0550
```

These match the notebook baseline from Lesson 1.1. The last line proves the
saved artifact loads back and predicts.

## Customize the data path

If your data lives elsewhere, set the environment variable for this command:

```bash
FLIGHT_DATA_DIR=/path/to/data uv run --locked flight-delays
```

## What changed from Lesson 1.1

| Before (notebook) | After (package) |
|---|---|
| One `.ipynb` file | Importable modules under `src/flight_delays/` |
| Hardcoded path `../../../../data/...` | Configurable through `config.py` or an environment variable |
| No dependency record | `pyproject.toml` + `uv.lock` |
| Cannot run without Jupyter | `uv run flight-delays` |
