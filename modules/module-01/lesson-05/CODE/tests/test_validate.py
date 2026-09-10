"""Schema and boundary checks — expected failures become assertions."""

from pathlib import Path

import pandas as pd
import pytest

from flight_delays.config import TARGET
from flight_delays.validate import ensure_columns, ensure_file_exists, ensure_nonempty


def test_ensure_file_exists_passes_for_a_real_file(tmp_path: Path) -> None:
    path = tmp_path / "flights.csv"
    path.write_text("a,b\n1,2\n")
    ensure_file_exists(path)


def test_ensure_file_exists_raises_with_a_remediation_hint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Download the flight-delays CSV"):
        ensure_file_exists(path)


def test_ensure_columns_passes_when_every_required_column_is_present(
    raw_flights: pd.DataFrame,
    raw_columns: list[str],
) -> None:
    ensure_columns(raw_flights, raw_columns)


@pytest.mark.parametrize(
    "missing",
    [
        [TARGET],
        ["Reporting_Airline"],
        ["Reporting_Airline", "Origin", "Dest"],
    ],
)
def test_ensure_columns_raises_naming_exactly_the_missing_columns(
    raw_flights: pd.DataFrame,
    raw_columns: list[str],
    missing: list[str],
) -> None:
    df = raw_flights.drop(columns=missing)
    with pytest.raises(ValueError) as excinfo:
        ensure_columns(df, raw_columns)
    message = str(excinfo.value)
    assert "missing required columns" in message
    for column in missing:
        assert column in message


def test_ensure_columns_does_not_raise_for_a_superset_of_columns(
    raw_flights: pd.DataFrame,
    raw_columns: list[str],
) -> None:
    df = raw_flights.copy()
    df["BonusColumn"] = 1
    ensure_columns(df, raw_columns)


def test_ensure_nonempty_passes_when_rows_exist(
    raw_flights: pd.DataFrame,
) -> None:
    ensure_nonempty(raw_flights, target=TARGET)


def test_ensure_nonempty_raises_for_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="No modeling rows"):
        ensure_nonempty(pd.DataFrame(), target=TARGET)
