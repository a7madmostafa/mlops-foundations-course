"""Shared fixtures for the flight-delays test suite.

The fixtures build small or seeded synthetic DataFrames instead of reading the
real CSV. The real data is large and lives separately from the code, so tests
that depend on a download would fail on a clean clone. Synthetic data keeps the
suite fast, deterministic, and runnable anywhere.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from flight_delays.config import (
    CATEGORICAL_FEATURES,
    MODEL_INPUT_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)
from flight_delays.features import build_model
from flight_delays.train import train_model


@pytest.fixture
def raw_flights() -> pd.DataFrame:
    """A tiny raw frame mirroring the real CSV's shape.

    Six rows: one cancelled (index 2), the rest completed and non-diverted.
    Columns are exactly the raw schema the pipeline validates against.
    """
    return pd.DataFrame(
        {
            "Cancelled": [0, 0, 1, 0, 0, 0],
            "Diverted": [0, 0, 0, 0, 0, 0],
            "CRSDepTime": [900, 2130, 600, 5, 2300, 1505],
            "DepDelay": [-3, 12, 5, 0, -7, 30],
            "Distance": [180, 900, 340, 120, 2500, 55],
            "CRSElapsedTime": [40, 150, 75, 30, 320, 15],
            "DayOfWeek": [1, 2, 3, 4, 5, 6],
            "Reporting_Airline": ["AA", "DL", "UA", "AA", "WN", "DL"],
            "Origin": ["JFK", "ATL", "ORD", "JFK", "LAX", "SFO"],
            "Dest": ["LAX", "MIA", "DFW", "BOS", "ORD", "SEA"],
            TARGET: [0, 1, 0, 0, 0, 1],
        }
    )


@pytest.fixture
def raw_columns(raw_flights: pd.DataFrame) -> list[str]:
    """The exact column list required of a raw flight-delays frame."""
    return list(raw_flights.columns)


@pytest.fixture(scope="session")
def raw_training_flights() -> pd.DataFrame:
    """A deterministic raw dataset large enough for an end-to-end model run.

    It contains raw ``CRSDepTime`` values, not the derived hour. This lets the
    integration test prove that raw input really flows through feature
    derivation, preprocessing, training, saving, loading, and prediction.
    """
    rng = np.random.default_rng(seed=RANDOM_STATE)
    n_rows = 300
    departure_delay = rng.integers(-10, 90, n_rows)
    scheduled_hour = rng.integers(0, 24, n_rows)
    scheduled_minute = rng.integers(0, 60, n_rows)
    target = (departure_delay + rng.normal(0, 25, n_rows) > 20).astype(int)

    return pd.DataFrame(
        {
            "Cancelled": np.zeros(n_rows, dtype=int),
            "Diverted": np.zeros(n_rows, dtype=int),
            "CRSDepTime": scheduled_hour * 100 + scheduled_minute,
            "DepDelay": departure_delay,
            "Distance": rng.integers(80, 2000, n_rows),
            "CRSElapsedTime": rng.integers(25, 340, n_rows),
            "DayOfWeek": rng.integers(1, 8, n_rows),
            "Reporting_Airline": rng.choice(["AA", "DL", "UA", "WN", "B6"], n_rows),
            "Origin": rng.choice(["JFK", "ATL", "ORD", "LAX", "DFW"], n_rows),
            "Dest": rng.choice(["MIA", "BOS", "SEA", "PHX", "SFO"], n_rows),
            TARGET: target,
        }
    )


@pytest.fixture(scope="session")
def flights_xy(raw_training_flights: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw model inputs and labels from the shared synthetic dataset."""
    return (
        raw_training_flights[MODEL_INPUT_COLUMNS].copy(),
        raw_training_flights[TARGET].copy(),
    )


@pytest.fixture(scope="session")
def fitted_model(flights_xy: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    """A fitted logistic-regression pipeline on the synthetic set."""
    X, y = flights_xy
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    return train_model(model, X, y)


@pytest.fixture
def required_feature_columns() -> list[str]:
    """The exact raw columns a trained model expects as input."""
    return MODEL_INPUT_COLUMNS
