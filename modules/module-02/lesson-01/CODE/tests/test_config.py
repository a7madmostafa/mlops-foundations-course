"""Config sanity — the lists the whole pipeline depends on are consistent."""

from flight_delays.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MODEL_INPUT_COLUMNS,
    NUMERIC_FEATURES,
    REQUIRED_RAW_COLUMNS,
)


def test_feature_lists_do_not_overlap() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_features_is_the_concatenation_of_both_lists() -> None:
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_raw_schema_covers_every_model_input() -> None:
    assert set(MODEL_INPUT_COLUMNS) <= set(REQUIRED_RAW_COLUMNS)


def test_raw_schema_contains_no_derived_column() -> None:
    assert "scheduled_departure_hour" not in REQUIRED_RAW_COLUMNS


def test_engineered_features_replace_raw_time_with_derived_hour() -> None:
    assert "CRSDepTime" in MODEL_INPUT_COLUMNS
    assert "CRSDepTime" not in FEATURES
    assert "scheduled_departure_hour" in FEATURES
