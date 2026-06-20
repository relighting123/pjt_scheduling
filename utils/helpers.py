"""
utils/helpers.py – 공통 유틸리티
날짜 파싱, 범주형 인코딩, 색상 맵 등 프로젝트 전반에서 재사용되는 함수 모음.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ── 날짜/시간 ──────────────────────────────────────────────────────────────────

def parse_datetime(s: str) -> datetime:
    """
    목적: 문자열 날짜를 datetime 객체로 변환
    Input:  "2024-01-01 08:00:00"
    Output: datetime(2024, 1, 1, 8, 0, 0)
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"날짜 형식을 인식할 수 없습니다: {s}")


def datetime_to_minutes(dt: datetime, base: datetime) -> int:
    """
    목적: 기준 시각(base)으로부터 dt까지의 경과 분을 반환
    Input:  dt=datetime(2024,1,1,10,0), base=datetime(2024,1,1,8,0)
    Output: 120
    """
    return int((dt - base).total_seconds() / 60)


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


# ── 색상 ───────────────────────────────────────────────────────────────────────

def build_color_map(keys: List[str], palette: List[str]) -> Dict[str, str]:
    """
    목적: 키 리스트에 색상 팔레트를 순환 할당
    Input:  keys=["PPK001","PPK002"], palette=["#4C72B0","#DD8452",...]
    Output: {"PPK001": "#4C72B0", "PPK002": "#DD8452"}
    """
    unique_keys = sorted(set(keys))
    return {k: palette[i % len(palette)] for i, k in enumerate(unique_keys)}


# ── 검증 ───────────────────────────────────────────────────────────────────────

REQUIRED_SCHEDULE_FIELDS    = {"EQP_ID", "LOT_ID", "CARRIER_ID", "PLAN_PROD_KEY",
                                "ST", "SEQ", "STARTTM", "ENDTM"}
REQUIRED_AVAILABILITY_FIELDS = {"EQP_ID", "LOT_ID", "PLAN_PROD_KEY", "ST", "WF_QTY"}
REQUIRED_PLAN_FIELDS         = {"PLAN_PROD_KEY", "OPER_ID",
                                 "D0_PLAN_QTY", "D1_PLAN_QTY", "PLAN_PRIORITY"}
REQUIRED_FLOW_FIELDS         = {"PLAN_PROD_KEY", "SEQ_ID", "OPER_ID"}
REQUIRED_INCOMING_WIP_FIELDS = {"PLAN_PROD_KEY", "EQP_MODEL", "ARRIVE_TM",
                                 "PROC_TIME", "LOT_QTY", "WF_QTY", "OPER_ID"}
REQUIRED_SPLIT_FIELDS        = {"PLAN_PROD_KEY", "EQP_MODEL", "SPLIT_QTY"}


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
    Input:  records=[{...}], required={"EQP_ID","LOT_ID",...}, label="schedule"
    Output: [] (오류 없음) 또는 ["schedule: 필드 누락 – {'STARTTM'}"]
    """
    errors = []
    if not records:
        errors.append(f"{label}: 데이터가 비어 있습니다.")
        return errors
    first_keys = set(records[0].keys())
    missing = required - first_keys
    if missing:
        errors.append(f"{label}: 필드 누락 – {missing}")
    return errors
