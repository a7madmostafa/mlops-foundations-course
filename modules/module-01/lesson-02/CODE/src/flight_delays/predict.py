"""Load a saved model and predict delay probabilities."""

import joblib


def load_model(model_path):
    """Load a saved fitted pipeline from disk and return it."""
    return joblib.load(model_path)


def predict_delay(model, features_df):
    """Return the estimated delay probability (ArrDel15) for each row."""
    return model.predict_proba(features_df)[:, 1]