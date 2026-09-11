"""Feature derivation and model-pipeline construction."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_scheduled_departure_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an hour derived from the raw HHMM departure time."""
    features = df.copy()
    features["scheduled_departure_hour"] = (
        features["CRSDepTime"].floordiv(100).astype(int)
    )
    return features


def build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    """Return feature derivation, preprocessing, and classification as one pipeline.

    A raw feature row enters with ``CRSDepTime``. The saved pipeline derives
    ``scheduled_departure_hour``, imputes, scales, encodes, and classifies it.
    Training, evaluation, and later prediction therefore run the same steps.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            (
                "derive_departure_hour",
                FunctionTransformer(add_scheduled_departure_hour, validate=False),
            ),
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
