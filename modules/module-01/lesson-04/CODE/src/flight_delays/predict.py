"""The flight-delay model object — owns the prediction contract."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class FlightDelayModel:
    """A fitted flight-delay model with one job: predict delay probability.

    The class wraps a trained scikit-learn ``Pipeline`` — itself a stateful
    object holding the fitted preprocessing and the classifier. Keeping the
    artifact behind one class puts the input contract (a DataFrame of feature
    rows) and the output contract (a delay probability per row) in a single,
    reusable place that a caller never has to think about joblib or pipelines.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def load(cls, model_path: Path) -> "FlightDelayModel":
        """Load a fitted pipeline from disk and wrap it in a model object."""
        return cls(joblib.load(model_path))

    def predict_delay(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return the probability that each row's flight is delayed (ArrDel15)."""
        return self._pipeline.predict_proba(features_df)[:, 1]
