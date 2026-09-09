# Lesson 1.1 CODE — Your Starting Point

This is our reference notebook &#8212; one way to build a baseline model for
the flight-delay project. It is not something you should have produced;
it is a reference you compare against after building your own.

## What's here

- `flight_delays.ipynb` &#8212; our baseline. It runs end-to-end and prints
  accuracy and F1. It also carries every problem the course will teach
  you to fix (see the reading page for the labeled list).
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
the relative data path resolves. The notebook has no dependencies beyond
the usual scientific stack (pandas, scikit-learn, matplotlib, joblib).

The baseline is: **accuracy 0.9184, F1 0.7587** on 522,269 clean rows.

> Note: running it creates a `models/` folder &#8212; that generated artifact is
> git-ignored here, as it would be in a real project.
