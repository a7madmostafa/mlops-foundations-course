"""Flight-delay pipeline — entry point.

Usage:
    uv run python main.py
"""

import os
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.clean import clean_flights
from src.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    MAX_ITER,
    MODEL_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluate import evaluate_model
from src.features import build_model, build_preprocessor
from src.load import load_csv
from src.predict import load_model, predict_delay
from src.train import train_model


def main():
    # --- Load ----------------------------------------------------------------
    csv_path = Path(os.environ.get("FLIGHT_DATA_DIR", DATA_DIR)) / "flight_delays_2025_01.csv"
    print(f"Loading {csv_path} ...")
    df = load_csv(csv_path)

    # --- Clean ---------------------------------------------------------------
    flights = clean_flights(df)
    print(f"Modeling rows: {len(flights):,}")

    # --- Split ---------------------------------------------------------------
    X = flights[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = flights["ArrDel15"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --- Build + Train -------------------------------------------------------
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model = build_model(preprocessor, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    train_model(model, X_train, y_train)

    # --- Evaluate ------------------------------------------------------------
    metrics = evaluate_model(model, X_test, y_test)
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1:       {metrics['f1']:.4f}")

    # --- Save ----------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model_2025_01.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved pipeline to {model_path}")

    # --- Predict -------------------------------------------------------------
    loaded_model = load_model(model_path)
    delay_probability = predict_delay(loaded_model, X_test.iloc[:1])
    print(f"\nPredicted delay probability for the first test flight: {delay_probability[0]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
