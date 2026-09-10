"""Clear, loud checks for the data the pipeline depends on."""

from pathlib import Path

import pandas as pd


def ensure_file_exists(path: Path) -> None:
    """Raise FileNotFoundError with a helpful hint if the CSV is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. Download the flight-delays CSV "
            "first, then run again. See the download command in "
            "data/README.md."
        )


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required column is missing from the data."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing required columns: {missing}. "
            f"Found {len(df.columns)} columns instead."
        )


def ensure_nonempty(df: pd.DataFrame) -> None:
    """Raise ValueError if there are no rows left to model on."""
    if len(df) == 0:
        raise ValueError(
            "No modeling rows after cleaning. Check that the CSV contains "
            "completed flights with a known ArrDel15 value."
        )
