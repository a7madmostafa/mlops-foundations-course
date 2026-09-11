"""Pipeline integration — one raw dataset flows through the complete chain.

The seeded synthetic rows start with raw ``CRSDepTime`` and continue through
clean, validate, split, derive features, train, evaluate, save, load, and
predict. No pre-engineered dataset replaces them halfway through.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from flight_delays.clean import clean_flights
from flight_delays.config import (
    CATEGORICAL_FEATURES,
    MODEL_INPUT_COLUMNS,
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
    raw_training_flights: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cleaned = clean_flights(raw_training_flights, target=TARGET)
    ensure_columns(cleaned, MODEL_INPUT_COLUMNS + [TARGET])
    ensure_nonempty(cleaned, target=TARGET)

    X = cleaned[MODEL_INPUT_COLUMNS]
    y = cleaned[TARGET]
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
