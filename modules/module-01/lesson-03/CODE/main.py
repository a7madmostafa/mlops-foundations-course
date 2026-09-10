"""Flight-delay pipeline — entry point.

Usage:
    uv run python main.py
"""

import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.clean import clean_flights
from src.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DATA_FILE,
    FEATURES,
    MAX_ITER,
    MODEL_DIR,
    MODEL_FILE,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.evaluate import evaluate_model
from src.features import build_model
from src.load import load_csv
from src.predict import FlightDelayModel
from src.train import save_model, train_model


def main() -> None:
    # --- Load + clean ---------------------------------------------------------
    csv_path = Path(os.environ.get("FLIGHT_DATA_DIR", DATA_DIR)) / DATA_FILE
    print(f"Loading {csv_path} ...")
    flights = clean_flights(load_csv(csv_path))
    print(f"Modeling rows: {len(flights):,}")

    # --- Split ----------------------------------------------------------------
    X = flights[FEATURES]
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
