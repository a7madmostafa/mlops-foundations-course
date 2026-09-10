"""train_model — fitting a model object is one crisp contract."""

from flight_delays.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from flight_delays.features import build_model
from flight_delays.train import train_model


def test_train_model_fits_and_returns_the_same_object(
    flights_xy,
) -> None:
    X, y = flights_xy
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    fitted = train_model(model, X, y)
    assert fitted is model


def test_train_model_fills_the_classifier_coefficients(
    flights_xy,
) -> None:
    X, y = flights_xy
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    fitted = train_model(model, X, y)
    classifier = fitted.named_steps["classifier"]
    assert hasattr(classifier, "coef_")
