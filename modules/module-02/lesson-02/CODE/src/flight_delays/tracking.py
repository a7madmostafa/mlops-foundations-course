"""Experiment tracking with MLflow — log, search, and compare training runs.

Lesson 2.1's hand-rolled recorder taught the anatomy of a run record. This
lesson logs the same five pieces — parameters, metrics, the saved artifact,
the data identity, and the code + environment identity — through MLflow
Tracking, which gives us a stable schema, a searchable store, and a browser
UI we would otherwise have to build ourselves.

The fingerprint helpers survive from Lesson 2.1 unchanged: content hashes and
git stamps are still facts *we* compute, and MLflow stores them as run tags.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import mlflow

from flight_delays.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def data_fingerprint(path: Path) -> dict[str, str]:
    """Return the file name and a short SHA-256 of its bytes.

    Content identity: two files with the same bytes have the same digest,
    so the run records a fingerprint, not an opinion about the file name.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return {"file": path.name, "sha256": digest.hexdigest()[:16]}


def git_fingerprint() -> dict[str, str | bool]:
    """Return the short commit and dirty flag, or a clear "<none>" fallback."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.SubprocessError, OSError):
        return {"git_commit": "<none>", "git_dirty": False}


def dependency_versions() -> dict[str, str]:
    """Map each core library to its installed version, when it can be found."""
    versions: dict[str, str] = {}
    for name in ["numpy", "pandas", "scikit-learn", "joblib"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def code_fingerprint() -> dict[str, object]:
    """Describe the code + environment this run was produced with."""
    fingerprint: dict[str, object] = {"python_version": platform.python_version()}
    fingerprint.update(git_fingerprint())
    fingerprint["dependencies"] = dependency_versions()
    return fingerprint


def snapshot_params(
    max_iter: int,
    random_state: int,
    test_size: float,
) -> dict[str, int | float]:
    """Capture the parameters a training run chose."""
    return {
        "max_iter": max_iter,
        "random_state": random_state,
        "test_size": test_size,
    }


def record_run(
    params: dict[str, int | float],
    metrics: dict[str, float],
    artifact: Path,
    data: dict[str, str],
    code: dict[str, object],
    tracking_uri: str | Path = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> str:
    """Log one training run with MLflow and return its run id.

    Parameters and metrics go to MLflow's dedicated stores (stable schema for
    free). The data + code fingerprints are facts we compute, so they are
    attached as run tags, and the trained artifact is copied under the run so
    it stops being overwritten.
    """
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(experiment_name)

    dependencies = cast(dict[str, str], code.get("dependencies", {}))
    tags = {
        "data_file": data["file"],
        "data_sha256": data["sha256"],
        "git_commit": str(code["git_commit"]),
        "git_dirty": str(bool(code["git_dirty"])),
        "python_version": str(code["python_version"]),
        **{f"dependency.{name}": value for name, value in dependencies.items()},
    }

    with mlflow.start_run() as active_run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(artifact), artifact_path="model")
        return cast(str, active_run.info.run_id)


def compare_runs(
    tracking_uri: str | Path = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    sort_by: str = "f1",
) -> list[dict[str, object]]:
    """Return recorded runs ordered by one metric, best first.

    MLflow's search API is the reader: it queries the tracking store, sorts by
    the chosen metric, and hands back full run objects (id, params, metrics,
    tags). No hand-rolled file format to keep in sync.
    """
    mlflow.set_tracking_uri(str(tracking_uri))
    if mlflow.get_experiment_by_name(experiment_name) is None:
        return []

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{sort_by} DESC"],
        output_format="list",
    )
    rows: list[dict[str, object]] = []
    for run in runs:
        data = run.data
        rows.append(
            {
                "run_id": run.info.run_id,
                "params": dict(data.params),
                "metrics": {name: float(value) for name, value in data.metrics.items()},
                "tags": dict(data.tags),
            }
        )
    return rows
