"""Clear, loud checks for the data the pipeline depends on."""

from pathlib import Path

import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data violates the pipeline's declared contract."""


def ensure_file_exists(path: Path) -> None:
    """Raise FileNotFoundError with a helpful hint if the CSV is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. Download the flight-delays CSV "
            "first, then run again. See the download command in "
            "data/README.md."
        )


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise DataValidationError for missing required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Data is missing required columns: {missing}. "
            f"Found {len(df.columns)} columns instead."
        )


def ensure_nonempty(df: pd.DataFrame, target: str) -> None:
    """Raise DataValidationError if there are no rows left to model on."""
    if len(df) == 0:
        raise DataValidationError(
            "No modeling rows after cleaning. Check that the CSV contains "
            f"completed flights with a known {target} value."
        )


def ensure_scheduled_departure_times(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Raise DataValidationError unless every value is valid HHMM time."""
    times = pd.to_numeric(df[column], errors="coerce")
    invalid = (
        times.isna()
        | times.lt(0)
        | times.gt(2359)
        | times.mod(100).ge(60)
        | times.mod(1).ne(0)
    )
    if invalid.any():
        examples = df.loc[invalid, column].head(5).tolist()
        raise DataValidationError(
            f"Column {column} contains invalid HHMM times: {examples}. "
            "Expected whole numbers from 0000 to 2359 with minutes below 60."
        )
