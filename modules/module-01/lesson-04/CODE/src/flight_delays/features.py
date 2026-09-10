"""Pipeline construction — the one place preprocessing is defined."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    """Return the full pipeline: scaling, one-hot encoding, then logistic regression.

    Everything a feature row needs before classification — impute, scale,
    encode — lives inside this one pipeline. Because the fitted pipeline is
    saved and later loaded for prediction, training and prediction share
    identical preprocessing by construction.
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
