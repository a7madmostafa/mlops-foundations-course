"""load — reading a CSV off disk into a DataFrame."""

from pathlib import Path

import pandas as pd

from flight_delays.load import load_csv


def test_load_csv_reads_a_small_csv(tmp_path: Path) -> None:
    path = tmp_path / "flights.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    df = load_csv(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_csv_returns_a_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "flights.csv"
    path.write_text("a\n1\n")
    df = load_csv(path)
    assert isinstance(df, pd.DataFrame)
