"""
data/writer/db_load.py – RTS output SQL / output.json → Oracle 적재

loader.fetch_from_db 의 출력 반대 경로.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from config import BASE_DIR, CONFIG
from data.db_registry import DbRegistry, parse_sql_db_alias
from data.writer.rts_sql import build_writer_sql_scripts, write_sql
from utils.file_logger import get_daily_file_logger

_DDL_FILE = "rts_output_tables.sql"
_INF_SCRIPTS = ("rts_rslt_inf.sql", "rts_eqpconvplan_inf.sql")
_HIS_SCRIPTS = ("rts_rslt_his.sql", "rts_eqpconvplan_his.sql")
# save_kpi 옵션 켰을 때만 생성되는 스크립트 — 있으면 적재, 없으면 조용히 생략
_OPTIONAL_SCRIPTS = ("rts_perfmon_his.sql", "rts_validation.sql")

# ---------------------------------------------------------------------------
# SQL 실행(INSERT/DELETE/DDL) 파일 로거 — loader.fetch(SELECT)의 sql_fetch.log와 짝
# ---------------------------------------------------------------------------
_sql_logger: Optional[logging.Logger] = None


def _get_sql_logger() -> logging.Logger:
    """logs/sql_load.log 에 기록하는 파일 로거 (자정 회전, 백업 1개만 유지)."""
    global _sql_logger
    if _sql_logger is not None:
        return _sql_logger

    _sql_logger = get_daily_file_logger("sql_load", BASE_DIR / "logs", "sql_load.log")
    return _sql_logger


def _log_sql_execute(
    label: str, stmt: str, row_count: int, *, error: Optional[str] = None,
) -> None:
    """INSERT/DELETE/DDL 등 실행문 1건을 로그 파일에 기록(바인드 없이 이미 값이 인라인된 SQL)."""
    log = _get_sql_logger()
    if error is not None:
        log.log(logging.ERROR, "[%s] FAILED: %s\n%s", label, error, stmt)
        return
    level = logging.WARNING if row_count == 0 else logging.INFO
    log.log(level, "[%s] rows=%d\n%s", label, row_count, stmt)


def _resolve_ddl_path() -> Path:
    """환경별 DDL 우선: data/sql → data/sql.example."""
    for sub in ("sql", "sql.example"):
        path = BASE_DIR / "data" / sub / _DDL_FILE
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"DDL 파일 없음: data/sql/{_DDL_FILE} 또는 data/sql.example/{_DDL_FILE}"
    )


def split_sql_statements(sql_text: str) -> List[str]:
    """세미콜론 단위 분리. 빈 줄·주석만 있는 statement는 제외."""
    parts = re.split(r";", sql_text.strip())
    statements: List[str] = []
    for part in parts:
        lines = [
            ln for ln in part.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        if not lines:
            continue
        statements.append("\n".join(lines))
    return statements


def execute_sql_text(
    conn,
    sql_text: str,
    *,
    label: str = "script",
) -> int:
    """SQL 텍스트(복수 statement) 실행 후 commit."""
    statements = split_sql_statements(sql_text)
    if not statements:
        return 0
    cur = conn.cursor()
    try:
        for stmt in statements:
            try:
                cur.execute(stmt)
            except Exception as exc:
                _log_sql_execute(label, stmt, 0, error=str(exc))
                raise
            row_count = cur.rowcount if cur.rowcount is not None else 0
            _log_sql_execute(label, stmt, row_count)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
    print(f"[db-load] {label}: {len(statements)}개 statement 실행")
    return len(statements)


def execute_sql_file(conn, sql_path: Path) -> int:
    text = sql_path.read_text(encoding="utf-8")
    return execute_sql_text(conn, text, label=sql_path.name)


def apply_output_ddl(*, db_alias: Optional[str] = None) -> None:
    """RTS output 테이블 CREATE (최초 1회)."""
    ddl_path = _resolve_ddl_path()
    alias = db_alias or parse_sql_db_alias(ddl_path.read_text(encoding="utf-8"))
    registry = DbRegistry()
    creds = registry.resolve(alias)
    print(f"[db-load] DDL 적용: {ddl_path} (db={creds.alias})")
    with creds.connect() as conn:
        execute_sql_file(conn, ddl_path)


def _ensure_sql_dir(
    output_dir: Path,
    *,
    include_history: bool,
    regenerate: bool,
) -> Path:
    sql_dir = output_dir / "sql"
    json_path = output_dir / CONFIG.path.output_file
    if regenerate or not sql_dir.is_dir() or not any(sql_dir.glob("*.sql")):
        if not json_path.is_file():
            raise FileNotFoundError(
                f"output.json 없음: {json_path}. 추론 후 save_result 또는 --regenerate-sql 필요"
            )
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
        write_sql(payload, sql_dir, include_history=include_history)
        print(f"[db-load] output.json → SQL 재생성: {sql_dir}")
    return sql_dir


def load_output_sql_files(
    output_dir: Path,
    *,
    db_alias: Optional[str] = None,
    include_history: bool = True,
    regenerate_sql: bool = False,
    script_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    dataset .../output/sql/*.sql 을 Oracle에 실행.

    RTS_RSLT_INF는 동일 FAC_ID 기준 전체 DELETE 후 INSERT하여 항상 최신 결과만
    남긴다(writer 생성 SQL, RULE_TIMEKEY 무관). RTS_EQPCONVPLAN_INF는 동일
    FAC_ID+RULE_TIMEKEY 기존 행만 DELETE 후 INSERT한다(같은 회차 재실행 시
    JOB_ID 중복/PK 위반 방지, 다른 회차 결과는 계속 누적).
    """
    output_dir = Path(output_dir)
    sql_dir = _ensure_sql_dir(
        output_dir,
        include_history=include_history,
        regenerate=regenerate_sql,
    )

    if script_names:
        names = list(script_names)
    else:
        names = list(_INF_SCRIPTS)
        if include_history:
            names.extend(_HIS_SCRIPTS)
        names.extend(name for name in _OPTIONAL_SCRIPTS if (sql_dir / name).is_file())

    registry = DbRegistry()
    creds = registry.get_credentials(db_alias)
    executed: List[Path] = []

    with creds.connect() as conn:
        for name in names:
            path = sql_dir / name
            if not path.is_file():
                if name in _HIS_SCRIPTS and not include_history:
                    continue
                raise FileNotFoundError(f"SQL 파일 없음: {path}")
            execute_sql_file(conn, path)
            executed.append(path)

    print(f"[db-load] 적재 완료: {output_dir} ({len(executed)}개 파일)")
    return executed


def load_output_json(
    payload: Union[dict, Path, str],
    *,
    db_alias: Optional[str] = None,
    include_history: bool = True,
) -> None:
    """output.json(dict/경로)에서 SQL 생성 후 즉시 DB 적재."""
    if not isinstance(payload, dict):
        path = Path(payload)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

    scripts = build_writer_sql_scripts(payload, include_history=include_history)
    creds = DbRegistry().get_credentials(db_alias)
    order = list(_INF_SCRIPTS)
    if include_history:
        order.extend(_HIS_SCRIPTS)
    order.extend(name for name in _OPTIONAL_SCRIPTS if name in scripts)

    with creds.connect() as conn:
        for name in order:
            if name not in scripts:
                continue
            execute_sql_text(conn, scripts[name], label=name)

    rtk = payload.get("meta", {}).get("RULE_TIMEKEY", "")
    print(f"[db-load] JSON 직접 적재 완료 RULE_TIMEKEY={rtk}")


def resolve_output_dir(
    *,
    fac_id: str,
    split: str,
    period: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    from config import resolve_dataset_path

    _, out = resolve_dataset_path(fac_id, split, period)
    return out
