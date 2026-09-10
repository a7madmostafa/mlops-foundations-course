"""Pipeline integration — the module chain composes as one, on synthetic data.

The full CLi path that usually needs the real CSV runs here on a tiny raw
frame plus a seeded synthetic modeling set: clean, validate, split, build,
train, evaluate, save, load, predict — every link of the chain, together.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from flight_delays.clean import clean_flights
from flight_delays.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from flight_delays.evaluate import evaluate_model
from flight_delays.features import build_model
from flight_delays.predict import FlightDelayModel
from flight_delays.train import save_model, train_model
from flight_delays.validate import ensure_columns, ensure_nonempty


def test_pipeline_clean_to_predict_on_synthetic_data(
    raw_flights: pd.DataFrame,
    flights_xy: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    cleaned = clean_flights(raw_flights, target=TARGET)
    ensure_columns(cleaned, FEATURES + [TARGET])
    ensure_nonempty(cleaned, target=TARGET)
    assert len(cleaned) > 0

    X, y = flights_xy
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    fitted = train_model(model, X_train, y_train)

    metrics = evaluate_model(fitted, X_test, y_test)
    assert set(metrics) == {"accuracy", "f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0

    model_path = save_model(fitted, tmp_path / "model.joblib")
    served = FlightDelayModel.load(model_path)

    probabilities = served.predict_delay(X_test.iloc[:5])
    assert probabilities.shape == (5,)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
