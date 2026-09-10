"""Clean raw flight data into a modeling cohort."""

import pandas as pd


def clean_flights(df, target):
    """Drop cancelled/diverted flights, duplicates, and add features.

    Steps applied:
      1. Keep only completed, non-diverted flights with a known target.
      2. Remove duplicate rows.
      3. Derive ``scheduled_departure_hour`` from ``CRSDepTime``.

    Returns a new DataFrame — the original is not modified.
    """
    flights = (
        df.loc[
            df["Cancelled"].eq(0)
            & df["Diverted"].eq(0)
            & df[target].notna()
        ]
        .drop_duplicates()
        .copy()
    )

    flights["scheduled_departure_hour"] = (
        flights["CRSDepTime"].floordiv(100).clip(0, 23).astype(int)
    )

    return flights
