"""Load the flight-delays CSV into a DataFrame."""

import pandas as pd


def load_csv(path):
    """Read a flight-delays CSV and return a DataFrame."""
    return pd.read_csv(path)
