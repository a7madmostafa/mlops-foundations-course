"""Train the flight-delay model."""


def train_model(model, X_train, y_train):
    """Fit the pipeline on training data and return the fitted model."""
    model.fit(X_train, y_train)
    return model
