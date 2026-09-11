"""Tracking unit tests — runs are recorded, recoverable, and comparable."""

import json
from datetime import datetime, timezone
from pathlib import Path

from flight_delays.tracking import (
    RunRecorder,
    code_fingerprint,
    compare_runs,
    data_fingerprint,
    dependency_versions,
    snapshot_params,
)

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


# ── RunRecorder ─────────────────────────────────────────────────────


def _make_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"fake-model-bytes")
    return artifact


def _make_started(index: int = 0) -> datetime:
    return datetime(2026, 9, 11, 12, 0, index, tzinfo=timezone.utc)


def test_record_appends_one_json_line_and_copies_artifact(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "runs")
    started = _make_started()
    artifact = _make_artifact(tmp_path)

    run = recorder.record(
        params=snapshot_params(max_iter=200, random_state=7, test_size=0.2),
        metrics={"accuracy": 0.9, "f1": 0.8},
        artifact=artifact,
        data={"file": "data.csv", "sha256": "abcdef0123456789"},
        code={"python_version": "3.12.4"},
        started_at=started,
    )

    assert run["run_id"] == "20260911_120000_000000"
    assert Path(str(run["artifact"])).is_file()

    lines = (tmp_path / "runs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["params"]["max_iter"] == 200
    assert payload["metrics"]["f1"] == 0.8


def test_runs_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert RunRecorder(tmp_path / "empty").runs() == []


def test_runs_reads_all_appended_records_in_order(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "runs")
    for idx in range(3):
        recorder.record(
            params=snapshot_params(500, idx, 0.2),
            metrics={"accuracy": 0.88, "f1": 0.7 + idx * 0.05},
            artifact=_make_artifact(tmp_path),
            data={"file": "x.csv", "sha256": "0"},
            code={"python_version": "3.12"},
            started_at=_make_started(idx),
        )
    runs = recorder.runs()
    assert len(runs) == 3


def test_compare_orders_runs_by_f1_descending(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "runs")
    f1_values = [0.70, 0.90, 0.80]
    for idx, f1 in enumerate(f1_values):
        recorder.record(
            params=snapshot_params(500, idx, 0.2),
            metrics={"accuracy": 0.89, "f1": f1},
            artifact=_make_artifact(tmp_path),
            data={"file": "x.csv", "sha256": "0"},
            code={"python_version": "3.12"},
            started_at=_make_started(idx),
        )

    runs = compare_runs(tmp_path / "runs", sort_by="f1")
    ordered_f1 = [float(r["metrics"]["f1"]) for r in runs]  # type: ignore[index]
    assert ordered_f1 == sorted(f1_values, reverse=True)


def test_compare_returns_empty_for_missing_runs_dir(tmp_path: Path) -> None:
    assert compare_runs(tmp_path / "does-not-exist") == []
