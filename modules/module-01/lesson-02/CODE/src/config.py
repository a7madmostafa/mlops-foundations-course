"""Project configuration — paths, feature lists, and constants."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------

# Data lives in the shared course data pool at the repository root.
# The default works when you run from the lesson's CODE/ folder.
REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# --- Feature lists -----------------------------------------------------------

NUMERIC_FEATURES = [
    "DepDelay",
    "Distance",
    "CRSElapsedTime",
    "scheduled_departure_hour",
    "DayOfWeek",
]

CATEGORICAL_FEATURES = ["Reporting_Airline", "Origin", "Dest"]

# --- Training defaults -------------------------------------------------------

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 500
