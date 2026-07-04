"""
utils/helpers.py – 공통 유틸리티
범주형 인코딩, 검증 등 프로젝트 전반에서 재사용되는 함수 모음.
"""
from datetime import datetime
from typing import Dict, List, Optional


def minutes_to_str(minutes: int, base: datetime) -> str:
    """
    목적: 시뮬레이션 분(minute offset)을 가독성 있는 문자열로 변환
    Input:  minutes=120, base=datetime(2024,1,1,8,0)
    Output: "2024-01-01 10:00:00"
    """
    from datetime import timedelta
    return (base + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


# ── 범주형 인코딩 ──────────────────────────────────────────────────────────────

def build_index_map(items: List[str]) -> Dict[str, int]:
    """
    목적: 범주형 값 리스트를 0-based 인덱스 딕셔너리로 변환
    Input:  ["OPER001", "OPER002", "OPER003"]
    Output: {"OPER001": 0, "OPER002": 1, "OPER003": 2}
    """
    return {v: i for i, v in enumerate(sorted(set(items)))}


def encode_normalized(value: Optional[str], index_map: Dict[str, int], total: int) -> float:
    """
    목적: 범주형 값을 [0, 1] 정규화 인덱스로 변환 (RL 관측 벡터용)
    Input:  value="OPER002", index_map={"OPER001":0,"OPER002":1}, total=10
    Output: 0.1  (= 1/10)
    """
    if value is None or value not in index_map:
        return 0.0
    return index_map[value] / max(total - 1, 1)


# ── 검증 ───────────────────────────────────────────────────────────────────────

REQUIRED_DISCRETE_ARRANGE_FIELDS = {
    "EQP_ID", "LOT_ID", "PLAN_PROD_ATTR_VAL", "OPER_ID", "ST", "EQP_MODEL_CD", "WF_QTY",
}
REQUIRED_ABSTRACT_ARRANGE_FIELDS = {"PLAN_PROD_ATTR_VAL", "OPER_ID", "EQP_MODEL_CD", "ST"}
REQUIRED_PLAN_FIELDS         = {"PLAN_PROD_ATTR_VAL", "OPER_ID",
                                 "D0_PLAN_QTY", "D1_PLAN_QTY", "PLAN_PRIORITY"}
# PLAN_PRIORITY: null 허용 (null=최하위 우선순위). 값이 있으면 작을수록 우선.
# OVER_PRODUCTION_YN: 필수 아님(생략 시 'Y' = 초과생산 제약 없음)이라 REQUIRED에 넣지 않음.
NULLABLE_PLAN_FIELDS         = {"PLAN_PRIORITY"}
REQUIRED_FLOW_FIELDS         = {"PLAN_PROD_ATTR_VAL", "OPER_SEQ", "OPER_ID"}
REQUIRED_SPLIT_FIELDS        = {"PLAN_PROD_ATTR_VAL", "EQP_MODEL_CD", "SPLIT_QTY"}  # OPER_ID optional (defaults to *)
REQUIRED_LOT_MASTER_FIELDS   = {"LOT_ID", "LOT_CD", "TEMP"}
REQUIRED_TOOL_CAPACITY_FIELDS = {"LOT_CD", "EQP_MODEL_CD", "MAX_TOOL"}
REQUIRED_BATCH_INFO_FIELDS   = {"LOT_CD", "TEMP", "PLAN_PROD_ATTR_VAL", "OPER_ID"}


def _pick_record_field(record: dict, *names: str):
    """레코드에서 필드 조회 (대소문자 무시)."""
    for name in names:
        if name in record:
            return record[name]
    upper_keys = {k.upper(): k for k in record}
    for name in names:
        key = upper_keys.get(name.upper())
        if key is not None:
            return record[key]
    return None


def normalize_tool_capacity_rows(records: List[dict]) -> List[dict]:
    """
    tool_capacity.json 정규화.
    split/discrete와 동일하게 EQP_MODEL_CD 사용 (구형 EQP_MODEL 키는 호환).
    """
    out: List[dict] = []
    for record in records:
        lot_cd = _pick_record_field(record, "LOT_CD")
        model = _pick_record_field(record, "EQP_MODEL_CD")
        if model is None:
            model = _pick_record_field(record, "EQP_MODEL")
        max_tool = _pick_record_field(record, "MAX_TOOL")
        out.append({
            "LOT_CD": str(lot_cd).strip() if lot_cd is not None else "",
            "EQP_MODEL_CD": str(model).strip().upper() if model is not None else "",
            "MAX_TOOL": max_tool,
        })
    return out


def validate_tool_capacity_records(records: List[dict]) -> List[str]:
    """tool_capacity 전용 검증 – EQP_MODEL_CD 사용."""
    errors: List[str] = []
    if not records:
        errors.append("tool_capacity: 데이터가 비어 있습니다.")
        return errors
    normalized = normalize_tool_capacity_rows(records)
    errors += validate_records(normalized, REQUIRED_TOOL_CAPACITY_FIELDS, "tool_capacity")
    return errors


def effective_proc_time(st_per_wafer: int, wf_qty: int) -> int:
    """장당 ST(분/장) × split 이후 wf_qty → LOT 실제 가공 소요시간(분)."""
    return max(int(st_per_wafer), 0) * max(int(wf_qty), 0)


def coerce_int(value, *, field: str = "값") -> int:
    """JSON/Oracle 숫자 필드 → int (25, 25.0, \"25\", \"25.0\" 지원)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field}이(가) 비어 있습니다.")
    if isinstance(value, bool):
        raise ValueError(f"{field} 정수 변환 불가: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        try:
            return int(float(s))
        except ValueError as exc:
            raise ValueError(f"{field} 정수 변환 불가: {value!r}") from exc
    raise ValueError(f"{field} 정수 변환 불가: {value!r}")


def coerce_int_or_none(value, *, field: str = "값") -> Optional[int]:
    """coerce_int와 동일하되 null(None/빈 문자열)은 그대로 None 유지.

    PLAN_PRIORITY처럼 '값 없음'과 '값 있음'을 구분해야 하는 필드용
    (없음은 최하위 우선순위로 취급 — 호출부에서 정렬 시 처리).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return coerce_int(value, field=field)


def split_wf_qty(total: int, split_qty: int) -> List[int]:
    """
    wafer 수량을 SPLIT_QTY(장) 단위로 분할.
    예: 25장, split_qty=3 → [3,3,3,3,3,3,3,2,2]
    """
    if total <= 0:
        return []
    if split_qty <= 0:
        return [total]
    if total <= split_qty:
        return [total]

    n_full = total // split_qty
    remainder = total % split_qty

    if remainder == 0:
        return [split_qty] * n_full

    if remainder == 1 and n_full >= 1:
        return [split_qty] * (n_full - 1) + [split_qty - 1, split_qty - 1]

    return [split_qty] * n_full + [remainder]


def validate_records(
    records: List[dict], required: set, label: str, nullable: set = frozenset(),
) -> List[str]:
    """
    목적: 레코드 리스트가 필수 컬럼을 모두 갖추고 있는지 검사
    Input:  records=[{...}], required={"EQP_ID","LOT_ID",...}, label="discrete_arrange"
    Output: [] (오류 없음) 또는 ["discrete_arrange: 필드 누락 – {'ST'}"]
    nullable: required에는 포함되지만 값 자체는 null(빈 값)을 허용할 필드 집합
              (예: PLAN_PRIORITY — 키는 있어야 하되 값은 null일 수 있음).
    """
    errors = []
    if not records:
        errors.append(f"{label}: 데이터가 비어 있습니다.")
        return errors
    for idx, record in enumerate(records, start=1):
        missing = required - set(record.keys())
        if missing:
            errors.append(f"{label}[{idx}]: 필드 누락 – {missing}")
            continue

        empty = {
            field
            for field in required - nullable
            if record[field] is None
            or (isinstance(record[field], str) and not record[field].strip())
        }
        if empty:
            errors.append(f"{label}[{idx}]: 필드 값 비어 있음 – {empty}")
    return errors
