"""Project configuration — paths, features, and training defaults."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------

# Data lives in the shared course data pool at the repository root.
# The default works when you run from the lesson's CODE/ folder.
REPO_ROOT: Path = Path(__file__).resolve().parents[5]
DATA_DIR: Path = REPO_ROOT / "data"
DATA_FILE: str = "flight_delays_2025_01.csv"
MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE: str = "model_2025_01.joblib"

# --- Feature lists -----------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "DepDelay",
    "Distance",
    "CRSElapsedTime",
    "scheduled_departure_hour",
    "DayOfWeek",
]

CATEGORICAL_FEATURES: list[str] = ["Reporting_Airline", "Origin", "Dest"]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET: str = "ArrDel15"

# --- Training defaults -------------------------------------------------------

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
MAX_ITER: int = 500
