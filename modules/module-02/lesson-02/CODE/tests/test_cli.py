"""Command boundary tests — paths, artifacts, MLflow recording, and logging."""

import logging
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from flight_delays.cli import main, run
from flight_delays.config import DATA_FILE, MLFLOW_EXPERIMENT_NAME, MODEL_FILE, TARGET
from flight_delays.log_setup import setup_logging


def write_course_csv(rows: pd.DataFrame, directory: Path) -> None:
    """Write rows under the filename the command expects."""
    rows.to_csv(directory / DATA_FILE, index=False)


def sqlite_uri(stem: Path) -> str:
    """A tracking-store URI pointing at a SQLite database next to ``stem``."""
    return "sqlite:///" + (stem / "mlflow.db").as_posix()


def search_runs(tracking_uri: str) -> list[object]:
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME], output_format="list"
    )


def test_main_runs_from_raw_csv_to_saved_model(
    raw_training_flights: pd.DataFrame,
    tmp_path: Path,
) -> None:
    write_course_csv(raw_training_flights, tmp_path)
    model_dir = tmp_path / "models"
    tracking_uri = sqlite_uri(tmp_path)

    metrics = main(data_dir=tmp_path, model_dir=model_dir, tracking_uri=tracking_uri)

    assert set(metrics) == {"accuracy", "f1"}
    assert (model_dir / MODEL_FILE).is_file()
    assert len(search_runs(tracking_uri)) == 1


def test_main_records_a_run_with_params_and_metrics(
    raw_training_flights: pd.DataFrame,
    tmp_path: Path,
) -> None:
    write_course_csv(raw_training_flights, tmp_path)
    tracking_uri = sqlite_uri(tmp_path)

    metrics = main(
        data_dir=tmp_path, model_dir=tmp_path / "models", tracking_uri=tracking_uri
    )

    runs = search_runs(tracking_uri)
    assert len(runs) == 1
    data = runs[0].data  # type: ignore[attr-defined]
    assert float(data.metrics["accuracy"]) == metrics["accuracy"]
    assert float(data.metrics["f1"]) == metrics["f1"]
    assert data.params["max_iter"] == "500"
    assert data.params["random_state"] == "42"
    assert data.tags["data_sha256"]


def test_main_records_different_runs_for_different_random_state(
    raw_training_flights: pd.DataFrame,
    tmp_path: Path,
) -> None:
    write_course_csv(raw_training_flights, tmp_path)
    tracking_uri = sqlite_uri(tmp_path)

    first = main(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        tracking_uri=tracking_uri,
        random_state=42,
    )
    second = main(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        tracking_uri=tracking_uri,
        random_state=7,
    )

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        order_by=["start_time ASC"],
        output_format="list",
    )
    assert len(runs) == 2
    assert runs[0].info.run_id != runs[1].info.run_id
    first_metrics = runs[0].data.metrics
    second_metrics = runs[1].data.metrics
    assert float(first_metrics["accuracy"]) == first["accuracy"]
    assert float(first_metrics["f1"]) == first["f1"]
    assert float(second_metrics["accuracy"]) == second["accuracy"]
    assert float(second_metrics["f1"]) == second["f1"]


def test_run_compare_prints_table(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    run(tracking_uri=sqlite_uri(tmp_path), argv=["compare"])

    output = capsys.readouterr().out
    assert "No recorded runs yet" in output


def test_run_logs_expected_input_failure_and_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(
            data_dir=tmp_path,
            model_dir=tmp_path / "models",
            tracking_uri=sqlite_uri(tmp_path),
            argv=[],
        )

    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "ERROR" in output
    assert "Download the flight-delays CSV" in output


def test_run_preserves_unexpected_value_error_traceback(
    raw_training_flights: pd.DataFrame,
    tmp_path: Path,
) -> None:
    one_class = raw_training_flights.copy()
    one_class[TARGET] = 0
    write_course_csv(one_class, tmp_path)

    with pytest.raises(ValueError, match="at least 2 classes"):
        run(
            data_dir=tmp_path,
            model_dir=tmp_path / "models",
            tracking_uri=sqlite_uri(tmp_path),
            argv=[],
        )


def test_setup_logging_emits_timestamped_named_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging()
    logging.getLogger("flight_delays.test").info("message from test")

    output = capsys.readouterr().out
    assert "INFO" in output
    assert "flight_delays.test" in output
    assert "message from test" in output
