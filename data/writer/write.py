"""
data/writer/write.py – 추론 결과 dataset output 기록

loader.fetch_from_db 의 반대: RTS JSON + SQL을 dataset/.../output/ 에 기록.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import CONFIG

from data.writer.rts_json import build_rts_output
from data.writer.rts_sql import write_sql


def write_inference_result(
    result: dict,
    env_data: dict,
    output_dir: Optional[Path] = None,
    *,
    write_sql_files: bool = True,
) -> Path:
    """
    추론 결과 → output.json + sql/*.sql (Oracle 적재용).

    Returns:
        output.json 경로
    """
    d = output_dir or CONFIG.path.output_dir
    d.mkdir(parents=True, exist_ok=True)

    payload = build_rts_output(result, env_data)
    out_path = d / CONFIG.path.output_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[writer] JSON → {out_path}")

    if write_sql_files:
        sql_paths = write_sql(payload, d / "sql")
        for p in sql_paths:
            print(f"[writer] SQL → {p}")

    return out_path


def write_sql_from_json(
    json_path: Path,
    sql_dir: Optional[Path] = None,
    *,
    include_history: bool = True,
) -> list[Path]:
    """기존 output.json → SQL만 재생성."""
    json_path = Path(json_path)
    out_dir = sql_dir or json_path.parent / "sql"
    paths = write_sql(json_path, out_dir, include_history=include_history)
    print(f"[writer] {len(paths)}개 SQL → {out_dir}")
    return paths
