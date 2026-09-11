# Lesson 1.1 CODE — Your Starting Point

This is our reference notebook &#8212; one way to build a baseline model for
the flight-delay project. It is not something you should have produced;
it is a reference you compare against after building your own.

## What's here

- `flight_delays.ipynb` &#8212; our baseline. It explores the data, records
  insights from four EDA views, and trains a logistic-regression model with
  a leakage-safe scikit-learn preprocessing pipeline. It also carries the
  five operational problems the course will teach you to fix (see the
  reading page for the labeled list).
- `flight_delays.html` &#8212; browser-readable HTML export of the notebook.
- The data does **not** live in `CODE/` &#8212; it lives in the course data pool
  at the repo root (`data/`, see `data/README.md`). This lesson reads
  `data/flight_delays_2025_01.csv` (full January 2025, BTS On-Time
  Performance, arrival-delay flag `ArrDel15`).

## Run it

First make sure the data pool is present (see `data/README.md`). Then:

```
jupyter notebook flight_delays.ipynb
```

Restart the kernel, then **Run All**. Run it from the `CODE/` folder so
the relative data path resolves. If needed, install its complete stack with

```bash
python -m pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib
```

Lesson 1.2 replaces this informal setup with a recorded, locked environment.

The baseline is: **accuracy 0.8952, F1 0.7415** on 522,269 clean rows. The
prediction is made shortly after departure, so actual departure delay is an
available feature. The saved artifact contains the fitted imputers,
`StandardScaler`, `OneHotEncoder`, and classifier together.

> Note: running it creates a `models/` folder &#8212; that generated artifact is
> git-ignored here, as it would be in a real project.
