"""Flight-delay pipeline — command line, train and track runs with MLflow.

Usage:
    uv run flight-delays [run] [--max-iter N] [--random-state N]
    uv run flight-delays compare [--sort-by f1|accuracy]
    uv run --locked mlflow server --backend-store-uri sqlite:///mlflow.db
        (then browse http://localhost:5000)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import cast

from sklearn.model_selection import train_test_split

from flight_delays.clean import clean_flights
from flight_delays.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DATA_FILE,
    MAX_ITER,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_INPUT_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REQUIRED_RAW_COLUMNS,
    TARGET,
    TEST_SIZE,
)
from flight_delays.evaluate import evaluate_model
from flight_delays.features import build_model
from flight_delays.load import load_csv
from flight_delays.log_setup import setup_logging
from flight_delays.predict import FlightDelayModel
from flight_delays.tracking import (
    code_fingerprint,
    compare_runs,
    data_fingerprint,
    record_run,
    snapshot_params,
)
from flight_delays.train import save_model, train_model
from flight_delays.validate import (
    DataValidationError,
    ensure_columns,
    ensure_file_exists,
    ensure_nonempty,
    ensure_scheduled_departure_times,
)

logger = logging.getLogger(__name__)


def main(
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    tracking_uri: str | Path | None = None,
    max_iter: int = MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Run the pipeline, log the run with MLflow, and return its metrics."""
    logger.info(
        "Pipeline started (max_iter=%s, random_state=%s)", max_iter, random_state
    )

    # --- Load + validate + clean ---------------------------------------------
    resolved_data_dir = data_dir or Path(os.environ.get("FLIGHT_DATA_DIR", DATA_DIR))
    resolved_model_dir = model_dir or MODEL_DIR
    resolved_tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    csv_path = resolved_data_dir / DATA_FILE
    ensure_file_exists(csv_path)
    logger.info("Loading data from %s", csv_path)
    raw_flights = load_csv(csv_path)
    ensure_columns(raw_flights, REQUIRED_RAW_COLUMNS)
    ensure_scheduled_departure_times(raw_flights, column="CRSDepTime")
    flights = clean_flights(raw_flights, target=TARGET)
    ensure_columns(flights, MODEL_INPUT_COLUMNS + [TARGET])
    ensure_nonempty(flights, target=TARGET)
    logger.info("Modeling rows: %s", f"{len(flights):,}")

    # --- Split ----------------------------------------------------------------
    X = flights[MODEL_INPUT_COLUMNS]
    y = flights[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )

    # --- Build + train + evaluate ---------------------------------------------
    model = build_model(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        max_iter=max_iter,
        random_state=random_state,
    )
    fitted = train_model(model, X_train, y_train)
    logger.info("Training finished")

    raw_metrics = evaluate_model(fitted, X_test, y_test)
    metrics = {name: float(value) for name, value in raw_metrics.items()}
    logger.info("accuracy: %.4f", metrics["accuracy"])
    logger.info("f1:       %.4f", metrics["f1"])

    # --- Save + record --------------------------------------------------------
    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_model(fitted, resolved_model_dir / MODEL_FILE)
    logger.info("Saved pipeline to %s", model_path)

    run_id = record_run(
        tracking_uri=resolved_tracking_uri,
        params=snapshot_params(
            max_iter=max_iter, random_state=random_state, test_size=TEST_SIZE
        ),
        metrics=metrics,
        artifact=model_path,
        data=data_fingerprint(csv_path),
        code=code_fingerprint(),
    )
    logger.info(
        "Recorded run %s in experiment %s (tracking uri: %s)",
        run_id,
        MLFLOW_EXPERIMENT_NAME,
        resolved_tracking_uri,
    )

    delay_probability = FlightDelayModel.load(model_path).predict_delay(X_test.iloc[:1])
    first_flight_probability = float(delay_probability[0])
    logger.info(
        "Probability of delay for first test flight: %.4f",
        first_flight_probability,
    )

    logger.info("Pipeline finished")
    return metrics


def print_compare_table(tracking_uri: str | Path, sort_by: str) -> None:
    """Print runs from the MLflow tracking store, best metric first."""
    runs = compare_runs(tracking_uri=tracking_uri, sort_by=sort_by)
    if not runs:
        print(
            "No recorded runs yet in experiment "
            f"'{MLFLOW_EXPERIMENT_NAME}'. Train once with `uv run flight-delays run`."
        )
        return
    print(
        f"{'run_id':<38}{'max_iter':>9} {'random_state':>13} {'accuracy':>10} {'f1':>8}"
    )
    for run in runs:
        params = cast(dict[str, str | int | float], run["params"])
        metrics = cast(dict[str, float], run["metrics"])
        print(
            f"{str(run['run_id']):<38}"
            f"{float(params['max_iter']):>9.0f} "
            f"{float(params['random_state']):>13.0f} "
            f"{float(metrics['accuracy']):>10.4f} "
            f"{float(metrics['f1']):>8.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line interface: run (default), compare, mlflow-server."""
    parser = argparse.ArgumentParser(
        prog="flight-delays",
        description="Train, record, and compare flight-delay model runs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="train the pipeline and record the run (default command)"
    )
    run_parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help="solver iterations (default: %(default)s)",
    )
    run_parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="random seed for the train/test split and model (default: %(default)s)",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="list recorded runs, best metric first"
    )
    compare_parser.add_argument(
        "--sort-by",
        choices=["f1", "accuracy"],
        default="f1",
        help="metric used to order the runs (default: %(default)s)",
    )
    return parser


def run(
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    tracking_uri: str | Path | None = None,
    argv: list[str] | None = None,
) -> None:
    """Parse the command line, then train and record a run or print the table."""
    setup_logging()
    args = build_parser().parse_args(argv)

    if args.command == "compare":
        print_compare_table(tracking_uri or MLFLOW_TRACKING_URI, sort_by=args.sort_by)
        return

    try:
        main(
            data_dir=data_dir,
            model_dir=model_dir,
            tracking_uri=tracking_uri or MLFLOW_TRACKING_URI,
            max_iter=getattr(args, "max_iter", MAX_ITER),
            random_state=getattr(args, "random_state", RANDOM_STATE),
        )
    except (FileNotFoundError, DataValidationError) as error:
        logger.error("Pipeline failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    run()
