"""test 벤치마크 결과 디스크 저장 (FAC별 test_benchmark.json)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import DATASET_DIR, validate_path_segment


def benchmark_path(fac_id: str) -> Path:
    fac = validate_path_segment(fac_id, "FAC_ID")
    return DATASET_DIR / fac / "test_benchmark.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_state(fac_id: str) -> dict[str, Any]:
    return {
        "fac_id": fac_id,
        "algorithms": [],
        "status": "idle",
        "progress": {"current": 0, "total": 0, "label": ""},
        "updated_at": None,
        "datasets": [],
    }


def load_benchmark(fac_id: str) -> dict[str, Any]:
    path = benchmark_path(fac_id)
    if not path.is_file():
        return empty_state(fac_id)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("fac_id", fac_id)
    data.setdefault("datasets", [])
    data.setdefault("algorithms", [])
    data.setdefault("status", "idle")
    data.setdefault("progress", {"current": 0, "total": 0, "label": ""})
    return data


def save_benchmark(state: dict[str, Any]) -> Path:
    fac_id = validate_path_segment(state["fac_id"], "FAC_ID")
    path = benchmark_path(fac_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = _now_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, separators=(",", ":"))
    return path


def clear_benchmark(fac_id: str) -> None:
    path = benchmark_path(fac_id)
    if path.is_file():
        path.unlink()


def init_benchmark(fac_id: str, algorithms: list[str], total: int) -> dict[str, Any]:
    state = {
        "fac_id": fac_id,
        "algorithms": algorithms,
        "status": "running",
        "progress": {"current": 0, "total": total, "label": ""},
        "updated_at": _now_iso(),
        "datasets": [],
    }
    save_benchmark(state)
    return state


def append_dataset(
    fac_id: str,
    dataset_entry: dict[str, Any],
    progress_current: int,
    progress_total: int,
    progress_label: str,
    done: bool,
) -> dict[str, Any]:
    state = load_benchmark(fac_id)
    folders = {d["input_folder"] for d in state["datasets"]}
    if dataset_entry["input_folder"] in folders:
        state["datasets"] = [
            d for d in state["datasets"]
            if d["input_folder"] != dataset_entry["input_folder"]
        ]
    state["datasets"].append(dataset_entry)
    state["datasets"].sort(key=lambda d: d["input_folder"])
    state["progress"] = {
        "current": progress_current,
        "total": progress_total,
        "label": progress_label,
    }
    state["status"] = "complete" if done else "running"
    save_benchmark(state)
    return state
