"""Evaluate a fitted model on test data."""

from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_test, y_test):
    """Return a dict with accuracy and F1 on the test set."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
