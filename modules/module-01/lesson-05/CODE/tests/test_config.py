"""Config sanity — the lists the whole pipeline depends on are consistent."""

from flight_delays.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    REQUIRED_RAW_COLUMNS,
)


def test_feature_lists_do_not_overlap() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_features_is_the_concatenation_of_both_lists() -> None:
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_raw_schema_covers_every_feature_source() -> None:
    derived_by_cleaning = ["scheduled_departure_hour"]
    cleaned_columns = set(REQUIRED_RAW_COLUMNS) | set(derived_by_cleaning)
    assert set(FEATURES) <= cleaned_columns


def test_raw_schema_contains_no_derived_column() -> None:
    assert "scheduled_departure_hour" not in REQUIRED_RAW_COLUMNS
