"""Flight-delay pipeline — entry point.

Usage:
    uv run flight-delays
"""

import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from flight_delays.clean import clean_flights
from flight_delays.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DATA_FILE,
    MAX_ITER,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_INPUT_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from flight_delays.evaluate import evaluate_model
from flight_delays.features import build_model
from flight_delays.load import load_csv
from flight_delays.predict import FlightDelayModel
from flight_delays.train import save_model, train_model


def main() -> None:
    # --- Load + clean ---------------------------------------------------------
    csv_path = Path(os.environ.get("FLIGHT_DATA_DIR", DATA_DIR)) / DATA_FILE
    print(f"Loading {csv_path} ...")
    flights = clean_flights(load_csv(csv_path), target=TARGET)
    print(f"Modeling rows: {len(flights):,}")

    # --- Split ----------------------------------------------------------------
    X = flights[MODEL_INPUT_COLUMNS]
    y = flights[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --- Build + train + evaluate ---------------------------------------------
    model = build_model(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    fitted = train_model(model, X_train, y_train)

    metrics = evaluate_model(fitted, X_test, y_test)
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1:       {metrics['f1']:.4f}")

    # --- Save + predict -------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = save_model(fitted, MODEL_DIR / MODEL_FILE)
    print(f"\nSaved pipeline to {model_path}")

    delay_probability = FlightDelayModel.load(model_path).predict_delay(X_test.iloc[:1])
    first_flight_probability = float(delay_probability[0])
    print(
        f"\nProbability of delay for first test flight: {first_flight_probability:.4f}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
