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
    "EQP_ID", "LOT_ID", "PLAN_PROD_KEY", "OPER_ID", "ST", "EQP_MODEL_CD", "WF_QTY",
}
REQUIRED_ABSTRACT_ARRANGE_FIELDS = {"PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "ST"}
REQUIRED_PLAN_FIELDS         = {"PLAN_PROD_KEY", "OPER_ID",
                                 "D0_PLAN_QTY", "D1_PLAN_QTY", "PLAN_PRIORITY"}
REQUIRED_FLOW_FIELDS         = {"PLAN_PROD_KEY", "OPER_SEQ", "OPER_ID"}
REQUIRED_SPLIT_FIELDS        = {"PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "SPLIT_QTY"}
REQUIRED_LOT_MASTER_FIELDS   = {"LOT_ID", "LOT_CD", "TEMP"}
REQUIRED_TOOL_CAPACITY_FIELDS = {"LOT_CD", "EQP_MODEL", "MAX_TOOL"}
REQUIRED_BATCH_INFO_FIELDS   = {"LOT_CD", "TEMP", "PLAN_PROD_KEY", "OPER_ID"}


def effective_proc_time(st_per_wafer: int, wf_qty: int) -> int:
    """장당 ST(분/장) × split 이후 wf_qty → LOT 실제 가공 소요시간(분)."""
    return max(int(st_per_wafer), 0) * max(int(wf_qty), 0)


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


def validate_records(records: List[dict], required: set, label: str) -> List[str]:
    """
    목적: 레코드 리스트가 필수 컬럼을 모두 갖추고 있는지 검사
    Input:  records=[{...}], required={"EQP_ID","LOT_ID",...}, label="discrete_arrange"
    Output: [] (오류 없음) 또는 ["discrete_arrange: 필드 누락 – {'ST'}"]
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
            for field in required
            if record[field] is None
            or (isinstance(record[field], str) and not record[field].strip())
        }
        if empty:
            errors.append(f"{label}[{idx}]: 필드 값 비어 있음 – {empty}")
    return errors
