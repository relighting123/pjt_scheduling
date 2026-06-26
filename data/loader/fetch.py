"""
data/loader/fetch.py – JSON 로드 및 Oracle SQL → JSON 변환 (입력 fetch)

SQL 템플릿: data/sql/{name}.sql  →  data/dataset/.../input/{name}.json
각 SQL 상단 ``-- @db: <alias>`` 로 DB 지정 (.env DB_<ALIAS>_*).

공통 바인드: :FAC_ID, :RULE_TIMEKEY(기간 조회 시)
:LOT_CD (--lotcd, discrete_arrange 제외)
"""
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CONFIG,
    SQL_JSON_MAP,
    SQL_REQUIRED_KEYS,
    iter_rule_timekeys,
    normalize_rule_timekey,
    resolve_dataset_path,
    validate_path_segment,
    format_missing_input_file_error,
)
from data.db_registry import DbRegistry, parse_sql_db_alias
from data.loader.sql_binds import merge_fetch_binds, resolve_lot_cd
from utils.helpers import (
    validate_records,
    validate_tool_capacity_records,
    normalize_tool_capacity_rows,
    REQUIRED_DISCRETE_ARRANGE_FIELDS,
    REQUIRED_ABSTRACT_ARRANGE_FIELDS,
    REQUIRED_PLAN_FIELDS,
    REQUIRED_FLOW_FIELDS,
    REQUIRED_SPLIT_FIELDS,
    REQUIRED_LOT_MASTER_FIELDS,
    REQUIRED_BATCH_INFO_FIELDS,
)
from data.generator import build_abstract_arrange

# ---------------------------------------------------------------------------
# SQL 실행 파일 로거
# ---------------------------------------------------------------------------
_sql_logger: logging.Logger | None = None


def _get_sql_logger() -> logging.Logger:
    """logs/sql_fetch_YYYYMMDD.log 에 기록하는 파일 로거 (호출 시 1회 초기화)."""
    global _sql_logger
    if _sql_logger is not None:
        return _sql_logger

    from config import BASE_DIR  # 순환 import 방지용 지연 import

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    log_path = log_dir / f"sql_fetch_{today}.log"

    logger = logging.getLogger("sql_fetch")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    _sql_logger = logger
    return logger


def _inline_binds(sql_text: str, binds: dict) -> str:
    """Oracle :PARAM 바인드 변수를 리터럴 값으로 치환해 바로 실행 가능한 SQL 반환."""
    import re

    result = sql_text
    # 긴 키 먼저 치환해 부분 치환 오류 방지 (예: :RULE_TIMEKEY vs :RULE)
    for key in sorted(binds, key=len, reverse=True):
        val = binds[key]
        if val is None:
            literal = "NULL"
        elif isinstance(val, str):
            escaped = val.replace("'", "''")
            literal = f"'{escaped}'"
        else:
            literal = str(val)
        result = re.sub(rf":{re.escape(key)}\b", literal, result)
    return result


def _log_sql_execute(
    sql_file: str,
    alias: str,
    period: str | None,
    binds: dict,
    sql_text: str,
    row_count: int,
) -> None:
    """SQL 실행 1건을 로그 파일에 기록 (바인드 값 인라인된 실행 가능 SQL 포함)."""
    log = _get_sql_logger()

    per_str = period or "(없음)"
    runnable_sql = _inline_binds(sql_text, binds)

    level = logging.WARNING if row_count == 0 else logging.INFO
    log.log(
        level,
        "[%s] @%s  period=%s  rows=%d\n%s",
        sql_file,
        alias,
        per_str,
        row_count,
        runnable_sql,
    )


def _read_json_file(path: Path) -> List[dict]:
    # utf-8-sig: BOM 있는 파일(PowerShell 저장 등)과 없는 파일 모두 처리
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_data(input_dir: Path = None) -> Dict[str, List[dict]]:
    """dataset input 폴더 JSON 로드"""
    d = input_dir or CONFIG.path.input_dir

    def _read(filename: str) -> List[dict]:
        path = d / filename
        if not path.exists():
            raise FileNotFoundError(format_missing_input_file_error(d, filename))
        return _read_json_file(path)

    def _read_optional(filename: str) -> List[dict]:
        path = d / filename
        if not path.exists():
            return []
        return _read_json_file(path)

    def _read_discrete() -> List[dict]:
        path = d / CONFIG.path.discrete_arrange_file
        if not path.exists():
            raise FileNotFoundError(
                format_missing_input_file_error(d, CONFIG.path.discrete_arrange_file)
            )
        return _read_json_file(path)

    discrete_arrange = _read_discrete()
    flow = _read(CONFIG.path.flow_file)

    abstract_arrange = _read_optional(CONFIG.path.abstract_arrange_file)
    if not abstract_arrange and all(
        r.get("EQP_MODEL_CD") is not None
        and (not isinstance(r.get("EQP_MODEL_CD"), str) or r.get("EQP_MODEL_CD").strip())
        for r in discrete_arrange
    ):
        abstract_arrange = build_abstract_arrange(discrete_arrange, flow)

    tool_capacity = _read_optional(CONFIG.path.tool_capacity_file)
    if tool_capacity:
        tool_capacity = normalize_tool_capacity_rows(tool_capacity)

    return {
        "discrete_arrange":  discrete_arrange,
        "abstract_arrange":  abstract_arrange,
        "plan":              _read(CONFIG.path.plan_file),
        "flow":              flow,
        "split":             _read_optional(CONFIG.path.split_file),
        "lot_master":        _read_optional(CONFIG.path.lot_master_file),
        "tool_capacity":     tool_capacity,
        "eqp_initial_state": _read_optional(CONFIG.path.eqp_initial_state_file),
        "batch_info":        _read_optional(CONFIG.path.batch_info_file),
    }


def validate_data(raw: Dict[str, List[dict]]) -> List[str]:
    errors = []
    errors += validate_records(
        raw["discrete_arrange"], REQUIRED_DISCRETE_ARRANGE_FIELDS, "discrete_arrange",
    )
    errors += validate_records(
        raw["abstract_arrange"], REQUIRED_ABSTRACT_ARRANGE_FIELDS, "abstract_arrange",
    )
    errors += validate_records(raw["plan"], REQUIRED_PLAN_FIELDS, "plan")
    errors += validate_records(raw["flow"], REQUIRED_FLOW_FIELDS, "flow")
    if raw.get("split"):
        errors += validate_records(raw["split"], REQUIRED_SPLIT_FIELDS, "split")
    if raw.get("lot_master"):
        errors += validate_records(raw["lot_master"], REQUIRED_LOT_MASTER_FIELDS, "lot_master")
    if raw.get("tool_capacity"):
        errors += validate_tool_capacity_records(raw["tool_capacity"])
    if raw.get("batch_info"):
        errors += validate_records(raw["batch_info"], REQUIRED_BATCH_INFO_FIELDS, "batch_info")
    return errors


def _read_sql(sql_path: Path) -> str:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL 파일 없음: {sql_path}")
    text = sql_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"SQL 파일이 비어 있습니다: {sql_path}")
    return text


def _row_to_dict(cursor, row) -> dict:
    cols = [d[0] for d in cursor.description]
    out = {}
    for col, val in zip(cols, row):
        if hasattr(val, "isoformat"):
            val = val.strftime("%Y-%m-%d %H:%M:%S")
        out[col] = val
    return out


def _execute_query(conn, sql: str, binds: dict) -> List[dict]:
    cur = conn.cursor()
    try:
        cur.execute(sql, binds)
        return [_row_to_dict(cur, row) for row in cur.fetchall()]
    finally:
        cur.close()


def _format_fetch_error(
    *,
    sql_file: str,
    sql_path: Path,
    alias: str,
    period: Optional[str],
    binds: Dict[str, Any],
    cause: BaseException,
) -> str:
    per = f"period={period}" if period else "period=(없음)"
    bind_keys = ", ".join(sorted(binds))
    return (
        f"SQL 실패: {sql_file}\n"
        f"  path: {sql_path}\n"
        f"  db alias: {alias}\n"
        f"  {per}\n"
        f"  binds: {bind_keys}\n"
        f"  원인 ({type(cause).__name__}): {cause}"
    )


def _oracle_connect():
    """하위 호환 – default alias 단일 연결."""
    from data.db_registry import DbRegistry
    return DbRegistry().connect()


def fetch_from_db(
    fac_id: str,
    output_dir: Optional[Path] = None,
    split: str = "train",
    snapshot: Optional[str] = None,
    period: Optional[str] = None,
    extra_binds: Optional[Dict[str, Any]] = None,
    lot_cd: Optional[str] = None,
    db_registry: Optional[DbRegistry] = None,
    *,
    verbose: bool = False,
    dry_run: bool = False,
) -> Path:
    """data/sql/*.sql 실행 → JSON 저장 (쿼리별 @db alias 사용)."""
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    per = period or snapshot
    if output_dir is None:
        output_dir, _ = resolve_dataset_path(fac_id, split, per)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    sql_dir = CONFIG.path.sql_dir

    own_registry = db_registry is None
    registry = db_registry or DbRegistry()
    if verbose:
        print(
            f"[loader] 준비 fac={fac_id} split={split} {f'period={per}' if per else ''} "
            f"sql_dir={sql_dir} dry_run={dry_run}",
        )
        if registry.default_warn:
            print(f"[loader] note: {registry.default_warn}")
    try:
        for key, (sql_file, json_file) in SQL_JSON_MAP.items():
            sql_path = sql_dir / sql_file
            if not sql_path.exists() and key not in SQL_REQUIRED_KEYS:
                if verbose or dry_run:
                    print(f"[loader] 선택 SQL 없음 – skip {sql_file}")
                continue
            try:
                sql = _read_sql(sql_path)
                alias = parse_sql_db_alias(sql, registry.default_alias)
                out_path = output_dir / json_file
                binds = merge_fetch_binds(
                    fac_id,
                    per,
                    lot_cd=lot_cd,
                    include_lot_cd=(sql_file != "discrete_arrange.sql"),
                    extra_binds=extra_binds,
                )
                if verbose or dry_run:
                    print(
                        f"[loader] 계획 @{alias} {sql_file} → {out_path} "
                        f"binds={binds}",
                    )
                if dry_run:
                    registry.get_credentials(alias)
                    continue
                conn = registry.connect(alias)
                rows = _execute_query(conn, sql, binds)
                _log_sql_execute(sql_file, alias, per, binds, sql, len(rows))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
                row_warn = " ⚠ 0건" if not rows else ""
                print(
                    f"[loader] @{alias} {sql_file} → {out_path} ({len(rows)} rows{row_warn})",
                )
            except Exception as exc:
                alias_guess = registry.default_alias
                if sql_path.exists():
                    try:
                        alias_guess = parse_sql_db_alias(
                            sql_path.read_text(encoding="utf-8"),
                            registry.default_alias,
                        )
                    except Exception:
                        pass
                raise RuntimeError(
                    _format_fetch_error(
                        sql_file=sql_file,
                        sql_path=sql_path,
                        alias=alias_guess,
                        period=per,
                        binds=binds,
                        cause=exc,
                    ),
                ) from exc
    finally:
        if own_registry:
            registry.close_all()

    return output_dir


def fetch_period_range(
    fac_id: str,
    from_timekey: Optional[str] = None,
    to_timekey: Optional[str] = None,
    split: str = "train",
    extra_binds: Optional[Dict[str, Any]] = None,
    lot_cd: Optional[str] = None,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    periods: Optional[List[str]] = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> List[Path]:
    """RULE_TIMEKEY 구간별 Oracle SQL → JSON 저장"""
    if periods is not None:
        keys = [normalize_rule_timekey(p) for p in periods]
        range_label = (
            f"{keys[0]}~{keys[-1]} (db {len(keys)}건)"
            if keys else "(empty)"
        )
    else:
        start_key = from_timekey or from_date
        end_key = to_timekey or to_date
        if not start_key or not end_key:
            raise ValueError(
                "from_timekey와 to_timekey(또는 from_date/to_date, periods)를 지정하세요."
            )
        keys = list(iter_rule_timekeys(start_key, end_key))
        range_label = f"{start_key}~{end_key}"

    if dry_run and keys:
        keys = keys[:1]
        if verbose:
            print(f"[loader] dry-run: {range_label} 중 1일만 검증")

    paths: List[Path] = []
    with DbRegistry() as registry:
        for period in keys:
            day_binds = {"RULE_TIMEKEY": period, **(extra_binds or {})}
            path = fetch_from_db(
                fac_id=fac_id,
                split=split,
                period=period,
                extra_binds=day_binds,
                lot_cd=lot_cd,
                db_registry=registry,
                verbose=verbose,
                dry_run=dry_run,
            )
            paths.append(path)
    if dry_run:
        print(f"[loader] dry-run 완료 – {len(paths)}개 폴더 계획 확인")
    else:
        print(f"[loader] {split} RULE_TIMEKEY {range_label} → {len(paths)}개 폴더 생성")
    return paths
