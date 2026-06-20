"""
data/loader.py – JSON 로드 및 Oracle SQL → JSON 변환

SQL 템플릿: external/sql/{name}.sql  →  dataset/.../input/{name}.json
각 SQL은 :FAC_ID 등 바인드 변수 WHERE 절을 포함합니다.

환경 변수 (Oracle):
  ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CONFIG,
    SQL_JSON_MAP,
    iter_rule_timekeys,
    normalize_rule_timekey,
    resolve_dataset_path,
    validate_path_segment,
)
from utils.helpers import (
    validate_records,
    REQUIRED_SCHEDULE_FIELDS,
    REQUIRED_AVAILABILITY_FIELDS,
    REQUIRED_PLAN_FIELDS,
    REQUIRED_FLOW_FIELDS,
    REQUIRED_SPLIT_FIELDS,
)


def load_data(input_dir: Path = None) -> Dict[str, List[dict]]:
    """
    dataset input 폴더의 JSON 4종 로드
    """
    d = input_dir or CONFIG.path.input_dir

    def _read(filename: str) -> List[dict]:
        path = d / filename
        if not path.exists():
            raise FileNotFoundError(
                f"입력 파일 없음: {path}\n"
                f"python main.py sample 또는 python main.py fetch 로 데이터를 생성하세요."
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _read_optional(filename: str) -> List[dict]:
        path = d / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return {
        "schedule":     _read(CONFIG.path.schedule_file),
        "availability": _read(CONFIG.path.availability_file),
        "plan":         _read(CONFIG.path.plan_file),
        "flow":         _read(CONFIG.path.flow_file),
        "split":        _read_optional(CONFIG.path.split_file),
    }


def validate_data(raw: Dict[str, List[dict]]) -> List[str]:
    errors = []
    errors += validate_records(raw["schedule"],     REQUIRED_SCHEDULE_FIELDS,     "schedule")
    errors += validate_records(raw["availability"], REQUIRED_AVAILABILITY_FIELDS, "availability")
    errors += validate_records(raw["plan"],         REQUIRED_PLAN_FIELDS,         "plan")
    errors += validate_records(raw["flow"],         REQUIRED_FLOW_FIELDS,         "flow")
    if raw.get("split"):
        errors += validate_records(raw["split"], REQUIRED_SPLIT_FIELDS, "split")
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


def _oracle_connect():
    try:
        import oracledb
    except ImportError as e:
        raise ImportError(
            "oracledb 패키지가 필요합니다. pip install oracledb"
        ) from e

    cfg = CONFIG.oracle
    if not cfg.user or not cfg.password or not cfg.dsn:
        raise ValueError(
            "Oracle 접속 정보가 없습니다. "
            "환경 변수 ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN 을 설정하세요."
        )
    return oracledb.connect(user=cfg.user, password=cfg.password, dsn=cfg.dsn)


def fetch_from_db(
    fac_id: str,
    output_dir: Optional[Path] = None,
    split: str = "train",
    snapshot: Optional[str] = None,
    period: Optional[str] = None,
    extra_binds: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    external/sql/*.sql 실행 → JSON 저장

    Parameters
    ----------
    fac_id : 공장 ID (WHERE :FAC_ID)
    output_dir : 저장 input 경로 (None → dataset 규칙)
    split : train | test | infer
    snapshot / period : train|test RULE_TIMEKEY 폴더명 (YYYYMMDDHHmmss)
    extra_binds : 추가 SQL 바인드
    """
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    per = period or snapshot
    if output_dir is None:
        output_dir, _ = resolve_dataset_path(fac_id, split, per)
    output_dir.mkdir(parents=True, exist_ok=True)

    sql_dir = CONFIG.path.sql_dir
    binds: Dict[str, Any] = {"FAC_ID": fac_id}
    if per:
        binds["RULE_TIMEKEY"] = normalize_rule_timekey(per)
    if extra_binds:
        binds.update(extra_binds)
    if CONFIG.oracle.extra_binds:
        binds.update(CONFIG.oracle.extra_binds)

    conn = _oracle_connect()
    try:
        for key, (sql_file, json_file) in SQL_JSON_MAP.items():
            sql_path = sql_dir / sql_file
            sql = _read_sql(sql_path)
            rows = _execute_query(conn, sql, binds)
            out_path = output_dir / json_file
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
            print(f"[loader] {sql_file} → {out_path} ({len(rows)} rows)")
    finally:
        conn.close()

    return output_dir


def fetch_period_range(
    fac_id: str,
    from_timekey: Optional[str] = None,
    to_timekey: Optional[str] = None,
    split: str = "train",
    extra_binds: Optional[Dict[str, Any]] = None,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[Path]:
    """
    RULE_TIMEKEY(YYYYMMDDHHmmss) 구간별 Oracle SQL → JSON 저장
    각 기간마다 :RULE_TIMEKEY 바인드를 폴더명과 동일하게 설정합니다.
    """
    start_key = from_timekey or from_date
    end_key = to_timekey or to_date
    if not start_key or not end_key:
        raise ValueError("from_timekey와 to_timekey(또는 from_date/to_date)를 지정하세요.")

    paths: List[Path] = []
    for period in iter_rule_timekeys(start_key, end_key):
        day_binds = {"RULE_TIMEKEY": period, **(extra_binds or {})}
        path = fetch_from_db(
            fac_id=fac_id,
            split=split,
            period=period,
            extra_binds=day_binds,
        )
        paths.append(path)
    print(f"[loader] {split} RULE_TIMEKEY {start_key}~{end_key} → {len(paths)}개 폴더 생성")
    return paths
