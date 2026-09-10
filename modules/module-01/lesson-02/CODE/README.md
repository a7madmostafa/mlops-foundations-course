# Lesson 1.2 CODE — Notebook to Project

The flight-delay notebook has been split into importable modules. One
command reproduces the baseline metrics on any machine with Python and
`uv` installed.

## What's here

```
main.py            — entry point: orchestrates the full pipeline
src/
  config.py        — paths, feature lists, training defaults
  load.py          — load_csv()
  clean.py         — clean_flights()
  features.py      — build_preprocessor(), build_model()
  train.py         — train_model()
  evaluate.py      — evaluate_model()
```

## Install & run

Make sure the data pool is present (see `data/README.md` at the repo
root). Then, from this folder:

```bash
uv sync           # create venv + install dependencies
uv run python main.py
```

You should see:

```
accuracy: 0.8952
f1:       0.7415
```

These match the notebook baseline from Lesson 1.1 — same data, same
pipeline, same random seed.

## Customise the data path

If your data lives somewhere else, set the environment variable:

```bash
FLIGHT_DATA_DIR=/path/to/data uv run python main.py
```

## What changed from Lesson 1.1

| Before (notebook) | After (project) |
|---|---|
| One `.ipynb` file | 7 `.py` modules |
| Hardcoded path `../../../../data/...` | Configurable via `config.py` or env var |
| No dependency record | `pyproject.toml` + `uv.lock` |
| Cannot run without Jupyter | `uv run python main.py` |
