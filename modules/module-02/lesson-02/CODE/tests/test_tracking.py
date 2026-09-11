"""Tracking tests — runs are logged to MLflow, recoverable, and comparable."""

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run

from flight_delays.config import MLFLOW_EXPERIMENT_NAME
from flight_delays.tracking import (
    code_fingerprint,
    compare_runs,
    data_fingerprint,
    dependency_versions,
    record_run,
    snapshot_params,
)


def sqlite_uri(stem: Path) -> str:
    """A tracking-store URI pointing at a SQLite database next to ``stem``."""
    return "sqlite:///" + (stem / "mlflow.db").as_posix()


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """A dedicated local MLflow tracking store for this test."""
    return sqlite_uri(tmp_path)


@pytest.fixture(autouse=True)
def _isolated_tracking(tracking_uri: str) -> None:
    """Point MLflow at this test's store, then clear it afterwards."""
    mlflow.set_tracking_uri(tracking_uri)
    yield
    mlflow.set_tracking_uri("")


def _make_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"fake-model-bytes")
    return artifact


def _search(tracking_uri: str) -> list[Run]:
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME], output_format="list"
    )


def _code_stamp() -> dict[str, object]:
    """A complete code + environment stamp, as the CLI records it."""
    return {
        "python_version": "3.12.4",
        "git_commit": "abc1234",
        "git_dirty": False,
        "dependencies": {},
    }


# ── parameter snapshot ───────────────────────────────────────────────


def test_snapshot_params_returns_the_three_defaults() -> None:
    params = snapshot_params(max_iter=500, random_state=42, test_size=0.2)
    assert params == {"max_iter": 500, "random_state": 42, "test_size": 0.2}


# ── data fingerprint ────────────────────────────────────────────────


def test_data_fingerprint_is_stable_for_the_same_bytes(tmp_path: Path) -> None:
    path = tmp_path / "flights.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = data_fingerprint(path)
    second = data_fingerprint(path)
    assert first == second
    assert first["file"] == "flights.csv"
    assert len(first["sha256"]) == 16


def test_data_fingerprint_changes_when_file_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "flights.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    before = data_fingerprint(path)
    path.write_text("a,b\n1,999\n", encoding="utf-8")
    after = data_fingerprint(path)
    assert before["sha256"] != after["sha256"]


# ── code fingerprint ────────────────────────────────────────────────


def test_code_fingerprint_contains_commit_and_python() -> None:
    fingerprint = code_fingerprint()
    assert isinstance(fingerprint["python_version"], str)
    assert fingerprint["python_version"].startswith("3.")
    assert "git_commit" in fingerprint
    assert "git_dirty" in fingerprint
    assert "dependencies" in fingerprint


def test_dependency_versions_includes_the_core_libraries() -> None:
    versions = dependency_versions()
    for name in ("numpy", "pandas", "scikit-learn", "joblib"):
        assert name in versions
        assert versions[name] != "unknown"


# ── MLflow logging ─────────────────────────────────────────────────


def test_record_run_logs_params_metrics_and_tags(
    tmp_path: Path, tracking_uri: str
) -> None:
    run_id = record_run(
        tracking_uri=tracking_uri,
        params=snapshot_params(max_iter=200, random_state=7, test_size=0.2),
        metrics={"accuracy": 0.9, "f1": 0.8},
        artifact=_make_artifact(tmp_path),
        data={"file": "data.csv", "sha256": "abcdef0123456789"},
        code={
            "python_version": "3.12.4",
            "git_commit": "abc1234",
            "git_dirty": False,
            "dependencies": {"numpy": "2.0.0"},
        },
    )

    runs = _search(tracking_uri)
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == run_id
    assert run.data.params["max_iter"] == "200"
    assert run.data.params["random_state"] == "7"
    assert float(run.data.metrics["f1"]) == 0.8
    assert run.data.tags["data_sha256"] == "abcdef0123456789"
    assert run.data.tags["git_commit"] == "abc1234"
    assert run.data.tags["git_dirty"] == "False"
    assert run.data.tags["dependency.numpy"] == "2.0.0"


def test_record_run_returns_a_new_run_id_per_run(
    tmp_path: Path, tracking_uri: str
) -> None:
    library = snapshot_params(max_iter=500, random_state=42, test_size=0.2)
    metrics = {"accuracy": 0.9, "f1": 0.8}
    first = record_run(
        tracking_uri=tracking_uri,
        params=library,
        metrics=metrics,
        artifact=_make_artifact(tmp_path),
        data={"file": "x.csv", "sha256": "0"},
        code=_code_stamp(),
    )
    second = record_run(
        tracking_uri=tracking_uri,
        params=library,
        metrics=metrics,
        artifact=_make_artifact(tmp_path),
        data={"file": "x.csv", "sha256": "0"},
        code=_code_stamp(),
    )
    assert first != second
    assert len(_search(tracking_uri)) == 2


def test_record_run_copies_the_artifact_under_the_run(
    tmp_path: Path, tracking_uri: str
) -> None:
    from mlflow.tracking.client import MlflowClient

    run_id = record_run(
        tracking_uri=tracking_uri,
        params=snapshot_params(500, 42, 0.2),
        metrics={"accuracy": 0.9, "f1": 0.8},
        artifact=_make_artifact(tmp_path),
        data={"file": "x.csv", "sha256": "0"},
        code=_code_stamp(),
    )

    artifacts = MlflowClient(tracking_uri=tracking_uri).list_artifacts(
        run_id, path="model"
    )
    assert any(a.path == "model/model.joblib" for a in artifacts)


# ── searching ───────────────────────────────────────────────────────


def test_compare_orders_runs_by_f1_descending(
    tmp_path: Path, tracking_uri: str
) -> None:
    for idx, f1 in enumerate([0.70, 0.90, 0.80]):
        record_run(
            tracking_uri=tracking_uri,
            params=snapshot_params(500, idx, 0.2),
            metrics={"accuracy": 0.89, "f1": f1},
            artifact=_make_artifact(tmp_path),
            data={"file": "x.csv", "sha256": "0"},
            code=_code_stamp(),
        )

    runs = compare_runs(tracking_uri=tracking_uri, sort_by="f1")
    ordered_f1 = [float(r["metrics"]["f1"]) for r in runs]  # type: ignore[index]
    assert ordered_f1 == [0.90, 0.80, 0.70]


def test_compare_ignores_runs_of_other_experiments(
    tmp_path: Path, tracking_uri: str
) -> None:
    record_run(
        tracking_uri=tracking_uri,
        params=snapshot_params(500, 42, 0.2),
        metrics={"accuracy": 0.9, "f1": 0.8},
        artifact=_make_artifact(tmp_path),
        data={"file": "x.csv", "sha256": "0"},
        code=_code_stamp(),
    )

    rows = compare_runs(
        tracking_uri=tracking_uri, experiment_name="other", sort_by="f1"
    )
    assert rows == []


def test_compare_returns_empty_on_a_fresh_store(tmp_path: Path) -> None:
    assert (
        compare_runs(
            tracking_uri=sqlite_uri(tmp_path / "empty"),
            experiment_name=MLFLOW_EXPERIMENT_NAME,
        )
        == []
    )
