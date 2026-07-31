"""
data/dataset_fix.py – dataset JSON 일괄 수리 (로드 시 자동 보정 없음)

사용 예:
    python main.py fix-dataset tool-capacity --facid FAC001 --all
    python main.py fix-dataset tool-capacity --folder FAC001/train/20260621070000
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import (
    CONFIG,
    DATASET_DIR,
    list_fac_ids,
    list_split_folders,
    validate_path_segment,
)
from utils.helpers import _pick_record_field, normalize_tool_capacity_rows


def _lot_cd_to_models(
    discrete_arrange: List[dict],
    lot_master: Optional[List[dict]] = None,
) -> Dict[str, Set[str]]:
    lot_id_to_cd: Dict[str, str] = {}
    for row in lot_master or []:
        lot_id = _pick_record_field(row, "LOT_ID")
        lot_cd = _pick_record_field(row, "LOT_CD")
        if lot_id is None or lot_cd is None:
            continue
        lot_id_to_cd[str(lot_id).strip()] = str(lot_cd).strip()

    lot_cd_models: Dict[str, Set[str]] = {}
    for row in discrete_arrange or []:
        model = _pick_record_field(row, "EQP_MODEL_CD")
        if model is None:
            model = _pick_record_field(row, "EQP_MODEL")
        if model is None or not str(model).strip():
            continue
        model_s = str(model).strip().upper()
        lot_id = _pick_record_field(row, "LOT_ID")
        lot_cd = lot_id_to_cd.get(str(lot_id).strip()) if lot_id is not None else None
        if not lot_cd:
            continue
        lot_cd_models.setdefault(lot_cd, set()).add(model_s)
    return lot_cd_models


def _read_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON 배열이 아닙니다: {path}")
    return data


def _write_json(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def repair_tool_capacity_rows(
    tool_capacity: List[dict],
    discrete_arrange: List[dict],
    lot_master: Optional[List[dict]] = None,
) -> Tuple[List[dict], List[str]]:
    """tool_capacity.json 행을 discrete_arrange 기준으로 수리 (파일 쓰기 전용)."""
    if not tool_capacity:
        return [], []

    lot_cd_models = _lot_cd_to_models(discrete_arrange, lot_master)
    out: List[dict] = []
    notes: List[str] = []

    for idx, record in enumerate(tool_capacity, start=1):
        lot_cd = _pick_record_field(record, "LOT_CD")
        model = _pick_record_field(record, "EQP_MODEL_CD")
        if model is None:
            model = _pick_record_field(record, "EQP_MODEL")
        max_tool = _pick_record_field(record, "MAX_TOOL")
        lot_cd_s = str(lot_cd).strip() if lot_cd is not None else ""
        model_s = str(model).strip().upper() if model is not None else ""

        if not lot_cd_s:
            notes.append(f"[{idx}] LOT_CD 없음 — 행 제외")
            continue

        if model_s:
            out.append({
                "LOT_CD": lot_cd_s,
                "EQP_MODEL_CD": model_s,
                "MAX_TOOL": max_tool,
            })
            continue

        models = sorted(lot_cd_models.get(lot_cd_s, set()))
        if not models:
            notes.append(
                f"[{idx}] EQP_MODEL_CD 없음, discrete_arrange에서 "
                f"LOT_CD={lot_cd_s!r} 모델을 찾지 못함 — 행 제외"
            )
            continue

        notes.append(
            f"[{idx}] EQP_MODEL_CD 없음 → discrete_arrange 기준 "
            f"{', '.join(models)} 로 채움 (LOT_CD={lot_cd_s})"
        )
        for model_code in models:
            out.append({
                "LOT_CD": lot_cd_s,
                "EQP_MODEL_CD": model_code,
                "MAX_TOOL": max_tool,
            })

    return normalize_tool_capacity_rows(out), notes


def repair_tool_capacity_folder(folder_key: str) -> dict:
    """dataset/{folder_key}/input 의 tool_capacity.json 을 직접 수정."""
    folder_key = folder_key.strip().strip("/")
    parts = folder_key.split("/")
    if len(parts) < 2:
        raise ValueError(f"폴더 키 형식: FAC_ID/split[/period] (받은 값: {folder_key!r})")

    input_dir = DATASET_DIR / folder_key / "input"
    tool_path = input_dir / CONFIG.path.tool_capacity_file
    discrete_path = input_dir / CONFIG.path.discrete_arrange_file
    lot_master_path = input_dir / CONFIG.path.lot_master_file

    if not tool_path.exists():
        return {"folder": folder_key, "status": "skip", "reason": "tool_capacity.json 없음"}
    if not discrete_path.exists():
        return {"folder": folder_key, "status": "error", "reason": "discrete_arrange.json 없음"}

    tool_capacity = _read_json(tool_path)
    discrete_arrange = _read_json(discrete_path)
    lot_master = _read_json(lot_master_path)

    repaired, notes = repair_tool_capacity_rows(tool_capacity, discrete_arrange, lot_master)
    before_norm = normalize_tool_capacity_rows(tool_capacity)
    changed = repaired != before_norm

    if changed:
        _write_json(tool_path, repaired)

    return {
        "folder": folder_key,
        "status": "fixed" if changed else "ok",
        "rows_before": len(tool_capacity),
        "rows_after": len(repaired),
        "notes": notes,
    }


def resolve_dataset_folders(
    *,
    fac_id: Optional[str] = None,
    split: str = "train",
    all_folders: bool = False,
    folder_keys: Optional[List[str]] = None,
) -> List[str]:
    if folder_keys:
        return [f.strip().strip("/") for f in folder_keys if f and f.strip()]

    if fac_id is None:
        fac_ids = list_fac_ids(split)
        folders: List[str] = []
        for fid in fac_ids:
            if all_folders:
                folders.extend(list_split_folders(fid, split))
            else:
                available = list_split_folders(fid, split)
                if available:
                    folders.append(available[-1])
        return folders

    fac_id = validate_path_segment(fac_id, "FAC_ID")
    if all_folders:
        return list_split_folders(fac_id, split)
    available = list_split_folders(fac_id, split)
    return [available[-1]] if available else []


def run_tool_capacity_repair(
    *,
    fac_id: Optional[str] = None,
    split: str = "train",
    all_folders: bool = False,
    folder_keys: Optional[List[str]] = None,
) -> List[dict]:
    folders = resolve_dataset_folders(
        fac_id=fac_id, split=split, all_folders=all_folders, folder_keys=folder_keys,
    )
    if not folders:
        raise FileNotFoundError("수리할 dataset 폴더가 없습니다.")

    results = []
    for folder in folders:
        results.append(repair_tool_capacity_folder(folder))
    return results
