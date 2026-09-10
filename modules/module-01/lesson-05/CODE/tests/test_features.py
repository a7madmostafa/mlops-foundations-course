"""build_model — the pipeline is assembled from the pieces the project names."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from flight_delays.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from flight_delays.features import build_model


def test_build_model_returns_a_pipeline() -> None:
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    assert isinstance(model, Pipeline)


def test_build_model_has_exactly_two_named_steps() -> None:
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    assert list(model.named_steps) == ["preprocessor", "classifier"]


def test_build_model_ends_with_logistic_regression() -> None:
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)


def test_build_model_preprocessor_splits_numeric_and_categorical() -> None:
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    preprocessor = model.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert transformer_names == ["numeric", "categorical"]


def test_build_model_accepts_custom_max_iter_and_seed() -> None:
    model = build_model(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        max_iter=200,
        random_state=7,
    )
    classifier = model.named_steps["classifier"]
    assert classifier.max_iter == 200
    assert classifier.random_state == 7
