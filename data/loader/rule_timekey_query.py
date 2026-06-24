"""
data/loader/rule_timekey_query.py – DB 에서 RULE_TIMEKEY 목록/최신값 조회

메타 SQL (data/sql/, SQL_JSON_MAP 제외):
    rule_timekey_latest.sql  – 최신 1건
    rule_timekey_list.sql    – FROM~TO 구간 목록
    rule_timekey_recent.sql  – 최근 N개

활성화:
    data/sql/rule_timekey_*.sql 파일이 있으면 자동 사용
    RULE_TIMEKEY_FROM_DB=1|0 으로 강제 on/off
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CONFIG,
    iter_rule_timekeys,
    normalize_rule_timekey,
    resolve_train_period_range,
    rule_timekey_now,
    validate_path_segment,
)
from data.db_registry import DbRegistry, parse_sql_db_alias
from data.loader.fetch import _execute_query, _read_sql

RULE_TIMEKEY_LATEST_SQL = "rule_timekey_latest.sql"
RULE_TIMEKEY_LIST_SQL = "rule_timekey_list.sql"
RULE_TIMEKEY_RECENT_SQL = "rule_timekey_recent.sql"

_META_SQL_FILES = (
    RULE_TIMEKEY_LATEST_SQL,
    RULE_TIMEKEY_LIST_SQL,
    RULE_TIMEKEY_RECENT_SQL,
)


def use_db_rule_timekey() -> bool:
    """DB 메타 SQL 사용 여부 (auto / env override)."""
    raw = os.environ.get("RULE_TIMEKEY_FROM_DB", "").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    sql_dir = CONFIG.path.sql_dir
    return any((sql_dir / name).exists() for name in _META_SQL_FILES)


def _meta_sql_path(filename: str) -> Path:
    return CONFIG.path.sql_dir / filename


def _extract_rule_timekey(row: dict) -> Optional[str]:
    for key, value in row.items():
        if str(key).upper() == "RULE_TIMEKEY" and value is not None:
            text = str(value).strip()
            if text:
                return normalize_rule_timekey(text)
    return None


def _rows_to_keys(rows: List[dict]) -> List[str]:
    keys: List[str] = []
    for row in rows:
        key = _extract_rule_timekey(row)
        if key:
            keys.append(key)
    return keys


def execute_meta_sql(
    sql_filename: str,
    binds: Dict[str, Any],
    *,
    db_registry: Optional[DbRegistry] = None,
) -> List[dict]:
    """메타 SQL 1개 실행 → row list."""
    sql_path = _meta_sql_path(sql_filename)
    sql = _read_sql(sql_path)
    own_registry = db_registry is None
    registry = db_registry or DbRegistry()
    alias = parse_sql_db_alias(sql, registry.default_alias)
    try:
        conn = registry.connect(alias)
        return _execute_query(conn, sql, binds)
    except Exception as exc:
        raise RuntimeError(
            f"RULE_TIMEKEY 메타 SQL 실패: {sql_filename}\n"
            f"  path: {sql_path}\n"
            f"  db alias: {alias}\n"
            f"  binds: {binds}\n"
            f"  원인 ({type(exc).__name__}): {exc}",
        ) from exc
    finally:
        if own_registry:
            registry.close_all()


def fetch_latest_rule_timekey(
    fac_id: str,
    *,
    db_registry: Optional[DbRegistry] = None,
) -> Optional[str]:
    """DB 최신 RULE_TIMEKEY (sql 없거나 비활성 시 None)."""
    if not use_db_rule_timekey():
        return None
    path = _meta_sql_path(RULE_TIMEKEY_LATEST_SQL)
    if not path.exists():
        return None
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    rows = execute_meta_sql(
        RULE_TIMEKEY_LATEST_SQL,
        {"FAC_ID": fac_id},
        db_registry=db_registry,
    )
    keys = _rows_to_keys(rows)
    return keys[0] if keys else None


def fetch_rule_timekey_list(
    fac_id: str,
    from_key: str,
    to_key: str,
    *,
    db_registry: Optional[DbRegistry] = None,
) -> Optional[List[str]]:
    """DB 구간 RULE_TIMEKEY 목록 (sql 없으면 None → 호출측 fallback)."""
    if not use_db_rule_timekey():
        return None
    path = _meta_sql_path(RULE_TIMEKEY_LIST_SQL)
    if not path.exists():
        return None
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    start = normalize_rule_timekey(from_key)
    end = normalize_rule_timekey(to_key)
    rows = execute_meta_sql(
        RULE_TIMEKEY_LIST_SQL,
        {
            "FAC_ID": fac_id,
            "FROM_RULE_TIMEKEY": start,
            "TO_RULE_TIMEKEY": end,
        },
        db_registry=db_registry,
    )
    keys = _rows_to_keys(rows)
    return keys if keys else []


def fetch_recent_rule_timekeys(
    fac_id: str,
    prevdays: int,
    *,
    db_registry: Optional[DbRegistry] = None,
) -> Optional[List[str]]:
    """DB 최근 N개 RULE_TIMEKEY (오름차순). sql 없으면 None."""
    if not use_db_rule_timekey():
        return None
    path = _meta_sql_path(RULE_TIMEKEY_RECENT_SQL)
    if not path.exists():
        return None
    if prevdays < 1:
        raise ValueError("--prevdays 는 1 이상이어야 합니다.")
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    rows = execute_meta_sql(
        RULE_TIMEKEY_RECENT_SQL,
        {"FAC_ID": fac_id, "PREV_DAYS": prevdays},
        db_registry=db_registry,
    )
    keys = sorted(_rows_to_keys(rows))
    return keys if keys is not None else []


def _db_rule_timekey_error(fac_id: str, detail: str) -> ValueError:
    return ValueError(
        f"DB RULE_TIMEKEY 조회 실패 ({detail}).\n"
        f"  fac_id={fac_id}\n"
        f"  data/sql/rule_timekey_*.sql 을 배치하고 DB 연결을 확인하세요.\n"
        f"  (RULE_TIMEKEY_FROM_DB=0 이면 collector 폴더명은 DB 키를 쓸 수 없습니다.)",
    )


def resolve_collect_periods(
    fac_id: str,
    *,
    prevdays: int = 1,
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
    db_registry: Optional[DbRegistry] = None,
    require_db: bool = False,
) -> tuple[List[str], str]:
    """
    collector 수집 대상 RULE_TIMEKEY 목록.

    require_db=True 이면 DB 메타 SQL 결과만 사용 (로컬 일별 키 생성 없음).

    Returns:
        (periods, source)  source: ``db`` | ``local`` | ``cli``
    """
    fac_id = validate_path_segment(fac_id, "FAC_ID")

    if from_key and to_key:
        db_keys = fetch_rule_timekey_list(
            fac_id, from_key, to_key, db_registry=db_registry,
        )
        if db_keys is not None:
            if not db_keys:
                if require_db:
                    raise _db_rule_timekey_error(
                        fac_id,
                        f"구간 {from_key}~{to_key} 에 해당하는 RULE_TIMEKEY 없음",
                    )
                return [], "db"
            return db_keys, "db"
        if require_db:
            raise _db_rule_timekey_error(fac_id, "rule_timekey_list.sql 미설정 또는 비활성")
        return list(iter_rule_timekeys(from_key, to_key)), "local"

    if from_key or to_key:
        raise ValueError("--from 와 --to 를 함께 지정하세요.")

    db_keys = fetch_recent_rule_timekeys(
        fac_id, prevdays, db_registry=db_registry,
    )
    if db_keys is not None:
        if not db_keys:
            if require_db:
                raise _db_rule_timekey_error(
                    fac_id,
                    f"최근 {prevdays}개 RULE_TIMEKEY 없음",
                )
            return [], "db"
        return db_keys, "db"

    if require_db:
        raise _db_rule_timekey_error(fac_id, "rule_timekey_recent.sql 미설정 또는 비활성")

    start, end = resolve_train_period_range(prevdays=prevdays)
    return list(iter_rule_timekeys(start, end)), "local"


def resolve_snapshot_rule_timekey(
    fac_id: str,
    period: Optional[str] = None,
    *,
    db_registry: Optional[DbRegistry] = None,
    require_db: bool = False,
) -> tuple[str, str]:
    """스냅샷 1건 RULE_TIMEKEY (period 미지정 시 DB 최신, require_db 시 로컬 fallback 없음)."""
    if period:
        return normalize_rule_timekey(period), "cli"

    db_key = fetch_latest_rule_timekey(fac_id, db_registry=db_registry)
    if db_key:
        return db_key, "db"

    if require_db:
        raise _db_rule_timekey_error(fac_id, "rule_timekey_latest.sql 미설정 또는 비활성")

    return rule_timekey_now(), "local"
