"""Cleaning unit tests — row filters and the derived hour column."""

import pandas as pd
import pytest

from flight_delays.clean import clean_flights
from flight_delays.config import TARGET


def test_clean_keeps_only_completed_flights() -> None:
    df = pd.DataFrame(
        {
            "Cancelled": [0, 1, 0],
            "Diverted": [0, 0, 0],
            "CRSDepTime": [900, 600, 2130],
            TARGET: [0, 0, 1],
        }
    )
    flights = clean_flights(df, target=TARGET)
    assert len(flights) == 2
    assert (flights["Cancelled"] == 0).all()


def test_clean_drops_cancelled_and_diverted_flights(
    raw_flights: pd.DataFrame,
) -> None:
    flights = clean_flights(raw_flights, target=TARGET)
    assert (flights["Cancelled"] == 0).all()
    assert (flights["Diverted"] == 0).all()
    assert len(flights) < len(raw_flights)


def test_clean_drops_rows_with_missing_target(
    raw_flights: pd.DataFrame,
) -> None:
    raw_flights.loc[1, TARGET] = pd.NA
    flights = clean_flights(raw_flights, target=TARGET)
    assert flights[TARGET].notna().all()
    assert len(flights) < len(raw_flights)


def test_clean_keeps_completed_rows_when_nothing_is_wrong(
    raw_flights: pd.DataFrame,
) -> None:
    flights = clean_flights(raw_flights, target=TARGET)
    assert len(flights) == len(raw_flights) - 1


@pytest.mark.parametrize(
    "raw_time,expected_hour",
    [
        (0, 0),
        (900, 9),
        (2359, 23),
        (2400, 23),
        (2500, 23),
    ],
)
def test_scheduled_departure_hour_is_clipped_to_a_24h_clock(
    raw_flights: pd.DataFrame,
    raw_time: int,
    expected_hour: int,
) -> None:
    df = raw_flights.copy()
    df.loc[0, "CRSDepTime"] = raw_time
    flights = clean_flights(df, target=TARGET)
    assert flights.loc[0, "scheduled_departure_hour"] == expected_hour


def test_clean_can_return_an_empty_frame_when_everything_is_filtered(
    raw_flights: pd.DataFrame,
) -> None:
    df = raw_flights.copy()
    df["Cancelled"] = 1
    flights = clean_flights(df, target=TARGET)
    assert len(flights) == 0


def test_clean_does_not_mutate_the_input_frame(
    raw_flights: pd.DataFrame,
) -> None:
    before = raw_flights.copy()
    clean_flights(raw_flights, target=TARGET)
    pd.testing.assert_frame_equal(raw_flights, before)
