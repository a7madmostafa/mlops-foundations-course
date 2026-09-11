"""Hand-rolled experiment tracking — one JSON line per training run.

This module is intentionally small. It exists so the flight-delay project can
answer "which run was the best?" before the course introduces a dedicated
tracking tool (Lesson 2.2). A recorded run keeps everything needed to
recover and compare it later: parameters, metrics, the saved artifact, the
data it was trained on, and the code + environment that trained it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from flight_delays.config import RUNS_FILE


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


class RunRecorder:
    """Append one JSON line per recorded run and read the runs back.

    Each run gets a timestamp-based id, its trained artifact is copied into
    the runs directory under that id, and the whole record is appended to a
    single ``runs.jsonl`` file — machine-readable, one object per line.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.runs_file = runs_dir / RUNS_FILE

    def record(
        self,
        params: dict[str, int | float],
        metrics: dict[str, float],
        artifact: Path,
        data: dict[str, str],
        code: dict[str, object],
        started_at: datetime,
    ) -> dict[str, object]:
        """Store a completed run and return its full record."""
        run_id = started_at.strftime("%Y%m%d_%H%M%S_%f")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_artifact = self.runs_dir / f"runs_{run_id}.joblib"
        shutil.copyfile(artifact, run_artifact)

        run: dict[str, object] = {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "params": params,
            "metrics": metrics,
            "artifact": str(run_artifact),
            "data": data,
            "code": code,
        }
        line = json.dumps(run, sort_keys=True, separators=(",", ":"))
        with self.runs_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return run

    def runs(self) -> list[dict[str, object]]:
        """Read every recorded run, earliest first."""
        if not self.runs_file.is_file():
            return []
        decoded: list[dict[str, object]] = []
        with self.runs_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    decoded.append(cast(dict[str, object], json.loads(line)))
        return decoded


def compare_runs(runs_dir: Path, sort_by: str = "f1") -> list[dict[str, object]]:
    """Return recorded runs sorted by one metric, best first."""
    runs = RunRecorder(runs_dir).runs()

    def metric_value(run: dict[str, object]) -> float:
        metrics = cast(dict[str, float], run["metrics"])
        return float(metrics[sort_by])

    return sorted(runs, key=metric_value, reverse=True)
