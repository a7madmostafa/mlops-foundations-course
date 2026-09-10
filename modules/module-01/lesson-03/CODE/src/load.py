"""Load the flight-delays CSV into a DataFrame."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path | str) -> pd.DataFrame:
    """Read a flight-delays CSV and return a DataFrame."""
    return pd.read_csv(path)
