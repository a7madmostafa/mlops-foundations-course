"""Project configuration — paths, features, and training defaults."""

import os
from pathlib import Path

# MLflow ships an "agent hint" that points at an optional tracing skill; we
# are not using tracing, so keep it out of command output.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# --- Paths -------------------------------------------------------------------

# Data lives in the shared course data pool at the repository root.
# The default works when you run from the lesson's CODE/ folder.
REPO_ROOT: Path = Path(__file__).resolve().parents[6]
DATA_DIR: Path = REPO_ROOT / "data"
DATA_FILE: str = "flight_delays_2025_01.csv"
MODEL_DIR: Path = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE: str = "model_2025_01.joblib"

# --- Experiment tracking ---------------------------------------------------

# MLflow keeps run metadata in a local SQLite database. The server command
# below reads the same file, so the UI shows exactly what `run` records:
#     uv run --locked mlflow server --backend-store-uri sqlite:///mlflow.db
# Trained artifacts land under mlruns/ next to the database.
MLFLOW_TRACKING_URI: str = (
    "sqlite:///" + (Path(__file__).resolve().parents[2] / "mlflow.db").as_posix()
)
MLFLOW_EXPERIMENT_NAME: str = "flight-delays"

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

MODEL_INPUT_COLUMNS: list[str] = [
    "DepDelay",
    "Distance",
    "CRSElapsedTime",
    "CRSDepTime",
    "DayOfWeek",
    *CATEGORICAL_FEATURES,
]

TARGET: str = "ArrDel15"

REQUIRED_RAW_COLUMNS: list[str] = [
    "Cancelled",
    "Diverted",
    *MODEL_INPUT_COLUMNS,
    TARGET,
]

# --- Training defaults -------------------------------------------------------

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
MAX_ITER: int = 500
