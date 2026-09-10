"""Flight-delay pipeline — entry point.

Usage:
    uv run python main.py
"""

import logging
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
from src.log_setup import setup_logging
from src.predict import FlightDelayModel
from src.train import save_model, train_model
from src.validate import ensure_columns, ensure_file_exists, ensure_nonempty

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Pipeline started")

    # --- Load + validate + clean ---------------------------------------------
    csv_path = Path(os.environ.get("FLIGHT_DATA_DIR", DATA_DIR)) / DATA_FILE
    ensure_file_exists(csv_path)
    logger.info("Loading data from %s", csv_path)
    flights = clean_flights(load_csv(csv_path))

    ensure_columns(flights, FEATURES + [TARGET])
    ensure_nonempty(flights)
    logger.info("Modeling rows: %s", f"{len(flights):,}")

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
    logger.info("Training finished")

    metrics = evaluate_model(fitted, X_test, y_test)
    logger.info("accuracy: %.4f", metrics["accuracy"])
    logger.info("f1:       %.4f", metrics["f1"])

    # --- Save + predict -------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = save_model(fitted, MODEL_DIR / MODEL_FILE)
    logger.info("Saved pipeline to %s", model_path)

    delay_probability = FlightDelayModel.load(model_path).predict_delay(X_test.iloc[:1])
    first_flight_probability = float(delay_probability[0])
    logger.info(
        "Probability of delay for first test flight: %.4f",
        first_flight_probability,
    )

    logger.info("Pipeline finished")


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        logger.error("Pipeline failed: %s", error)
        raise SystemExit(1) from error
