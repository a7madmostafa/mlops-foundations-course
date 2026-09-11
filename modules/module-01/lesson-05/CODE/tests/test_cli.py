"""Command boundary tests — paths, artifacts, logging, and exit behavior."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from flight_delays.cli import main, run
from flight_delays.config import DATA_FILE, MODEL_FILE, TARGET
from flight_delays.log_setup import setup_logging


def write_course_csv(rows: pd.DataFrame, directory: Path) -> None:
    """Write rows under the filename the command expects."""
    rows.to_csv(directory / DATA_FILE, index=False)


def test_main_runs_from_raw_csv_to_saved_model(
    raw_training_flights: pd.DataFrame,
    tmp_path: Path,
) -> None:
    write_course_csv(raw_training_flights, tmp_path)
    model_dir = tmp_path / "models"

    metrics = main(data_dir=tmp_path, model_dir=model_dir)

    assert set(metrics) == {"accuracy", "f1"}
    assert (model_dir / MODEL_FILE).is_file()


def test_run_logs_expected_input_failure_and_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(data_dir=tmp_path, model_dir=tmp_path / "models")

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
        run(data_dir=tmp_path, model_dir=tmp_path / "models")


def test_setup_logging_emits_timestamped_named_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging()
    logging.getLogger("flight_delays.test").info("message from test")

    output = capsys.readouterr().out
    assert "INFO" in output
    assert "flight_delays.test" in output
    assert "message from test" in output
