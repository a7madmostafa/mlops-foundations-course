"""Model behavior — fitting, serializing, the prediction contract, metrics.

These tests pin the public behaviors a caller in the project relies on:
a fitted pipeline that trains, a model file that round-trips, probabilities
that stay bounded, and a metrics dict with the two keys evaluation prints.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from flight_delays.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from flight_delays.evaluate import evaluate_model
from flight_delays.features import build_model
from flight_delays.predict import FlightDelayModel
from flight_delays.train import save_model


def test_save_model_writes_a_non_empty_file(
    fitted_model: Pipeline,
    tmp_path: Path,
) -> None:
    path = save_model(fitted_model, tmp_path / "model.joblib")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_saved_model_round_trips_through_load(
    fitted_model: Pipeline,
    tmp_path: Path,
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = flights_xy
    path = save_model(fitted_model, tmp_path / "model.joblib")
    loaded = FlightDelayModel.load(path)
    before = FlightDelayModel(fitted_model).predict_delay(X.iloc[:5])
    after = loaded.predict_delay(X.iloc[:5])
    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("n_rows", [1, 3, 10])
def test_predict_delay_returns_one_probability_per_row(
    fitted_model: Pipeline,
    flights_xy: tuple[pd.DataFrame, pd.Series],
    n_rows: int,
) -> None:
    X, _ = flights_xy
    probabilities = FlightDelayModel(fitted_model).predict_delay(X.iloc[:n_rows])
    assert probabilities.shape == (n_rows,)


def test_predict_delay_probabilities_stay_in_the_unit_interval(
    fitted_model: Pipeline,
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = flights_xy
    probabilities = FlightDelayModel(fitted_model).predict_delay(X.iloc[:10])
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_predict_delay_returns_the_positive_class_probability(
    fitted_model: Pipeline,
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = flights_xy
    rows = X.iloc[:5]
    classes = fitted_model.classes_
    delayed_column = int(np.flatnonzero(classes == 1)[0])
    expected = fitted_model.predict_proba(rows)[:, delayed_column]
    actual = FlightDelayModel(fitted_model).predict_delay(rows)
    np.testing.assert_allclose(actual, expected)


def test_predict_delay_rejects_a_missing_raw_feature(
    fitted_model: Pipeline,
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = flights_xy
    with pytest.raises(ValueError, match="CRSDepTime"):
        FlightDelayModel(fitted_model).predict_delay(X.drop(columns="CRSDepTime"))


def test_predict_delay_requires_target_class_one(
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = flights_xy
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model.fit(X, y.replace({1: 2}))

    with pytest.raises(ValueError, match="target class 1"):
        FlightDelayModel(model).predict_delay(X.iloc[:1])


def test_saved_file_is_a_scikit_learn_pipeline(
    fitted_model: Pipeline,
    tmp_path: Path,
) -> None:
    path = save_model(fitted_model, tmp_path / "model.joblib")
    artifact = joblib.load(path)
    assert hasattr(artifact, "predict_proba")


def test_evaluate_model_returns_accuracy_and_f1(
    fitted_model: Pipeline,
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = flights_xy
    metrics = evaluate_model(fitted_model, X.iloc[:20], y.iloc[:20])
    assert set(metrics) == {"accuracy", "f1"}


def test_evaluate_metrics_stay_between_zero_and_one(
    fitted_model: Pipeline,
    flights_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = flights_xy
    metrics = evaluate_model(fitted_model, X.iloc[:20], y.iloc[:20])
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
