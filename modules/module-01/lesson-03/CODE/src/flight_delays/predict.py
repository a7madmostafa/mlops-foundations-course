"""The flight-delay model object — owns the prediction contract."""

from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class FlightDelayModel:
    """A fitted flight-delay model with one job: predict delay probability.

    The class wraps a trained scikit-learn ``Pipeline`` — itself a stateful
    object holding the fitted preprocessing and the classifier. Keeping the
    artifact behind one class puts the input contract (raw feature rows,
    including ``CRSDepTime``) and the output contract (one delay probability
    per row) in a single,
    reusable place that a caller never has to think about joblib or pipelines.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def load(cls, model_path: Path) -> "FlightDelayModel":
        """Load a fitted pipeline from disk and wrap it in a model object."""
        return cls(joblib.load(model_path))

    def predict_delay(self, raw_features: pd.DataFrame) -> np.ndarray:
        """Return the estimated ArrDel15=1 probability for each raw feature row."""
        return cast(np.ndarray, self._pipeline.predict_proba(raw_features)[:, 1])
