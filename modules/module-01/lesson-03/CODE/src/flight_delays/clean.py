"""Clean raw flight data into a modeling cohort."""

import pandas as pd


def clean_flights(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return completed, non-diverted, labeled flights without duplicates.

    Steps applied:
      1. Keep only completed, non-diverted flights with a known target.
      2. Remove duplicate rows.
    Returns a new DataFrame — the original is not modified.
    """
    flights = (
        df.loc[df["Cancelled"].eq(0) & df["Diverted"].eq(0) & df[target].notna()]
        .drop_duplicates()
        .copy()
    )

    return flights
