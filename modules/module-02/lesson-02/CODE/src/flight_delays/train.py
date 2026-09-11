"""Train the flight-delay model and persist the artifact."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the pipeline on training data and return the fitted model."""
    model.fit(X_train, y_train)
    return model


def save_model(model: Pipeline, model_path: Path) -> Path:
    """Serialize the fitted pipeline to disk and return the path written."""
    joblib.dump(model, model_path)
    return model_path
