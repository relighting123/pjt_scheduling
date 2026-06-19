"""
data/loader.py – JSON 데이터 로딩 & 샘플 데이터 생성
DB 조회 결과가 `external/input/` 폴더에 JSON으로 저장되어 있다고 가정합니다.
샘플 데이터 생성 함수로 외부 DB 없이 개발·테스트할 수 있습니다.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from config import CONFIG, EXTERNAL_DIR
from utils.helpers import (
    validate_records,
    REQUIRED_SCHEDULE_FIELDS,
    REQUIRED_AVAILABILITY_FIELDS,
    REQUIRED_PLAN_FIELDS,
    REQUIRED_FLOW_FIELDS,
)


# ── 로딩 ───────────────────────────────────────────────────────────────────────

def load_data(input_dir: Path = None) -> Dict[str, List[dict]]:
    """
    목적: external/input/ 폴더의 4개 JSON 파일을 읽어 딕셔너리로 반환
    Input:  input_dir=None → CONFIG.path.input_dir (external/<폴더명>/)
    Output: {
        "schedule":     [{EQP_ID, LOT_ID, CARRIER_ID, PLAN_PROD_KEY, ST, SEQ, STARTTM, ENDTM}, ...],
        "availability": [{EQP_ID, LOT_ID, PLAN_PROD_KEY, ST, WF_QTY}, ...],
        "plan":         [{PLAN_PROD_KEY, OPER_ID, D0_PLAN_QTY, D1_PLAN_QTY, PLAN_PRIORITY}, ...],
        "flow":         [{PLAN_PROD_KEY, SEQ_ID, OPER_ID}, ...]
    }
    """
    d = input_dir or CONFIG.path.input_dir

    def _read(filename: str) -> List[dict]:
        path = d / filename
        if not path.exists():
            raise FileNotFoundError(f"입력 파일 없음: {path}\n"
                                    f"generate_sample_data()를 먼저 실행하세요.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    raw = {
        "schedule":     _read(CONFIG.path.schedule_file),
        "availability": _read(CONFIG.path.availability_file),
        "plan":         _read(CONFIG.path.plan_file),
        "flow":         _read(CONFIG.path.flow_file),
    }
    return raw


def validate_data(raw: Dict[str, List[dict]]) -> List[str]:
    """
    목적: 로드된 데이터의 필수 컬럼 존재 여부 검증
    Input:  raw = load_data() 반환값
    Output: [] (오류 없음) 또는 오류 메시지 리스트
    """
    errors = []
    errors += validate_records(raw["schedule"],     REQUIRED_SCHEDULE_FIELDS,     "schedule")
    errors += validate_records(raw["availability"], REQUIRED_AVAILABILITY_FIELDS, "availability")
    errors += validate_records(raw["plan"],         REQUIRED_PLAN_FIELDS,         "plan")
    errors += validate_records(raw["flow"],         REQUIRED_FLOW_FIELDS,         "flow")
    return errors


# ── 샘플 시나리오 빌더 ────────────────────────────────────────────────────────

_SAMPLE_BASE = datetime(2024, 1, 15, 8, 0, 0)


def _fmt_minutes(minutes: int) -> str:
    return (_SAMPLE_BASE + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _schedule_row(
    eqp_id: str, lot_id: str, carrier_id: str, ppk: str, seq: int,
    start_min: int, end_min: int, eqp_model: str = "A",
) -> dict:
    proc = max(end_min - start_min, 1)
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "PLAN_PROD_KEY": ppk, "EQP_MODEL": eqp_model, "ST": proc, "SEQ": seq,
        "STARTTM": _fmt_minutes(start_min), "ENDTM": _fmt_minutes(end_min),
    }


def _avail_row(
    eqp_id: str, lot_id: str, ppk: str, proc_time: int,
    wf_qty: int = 25, eqp_model: str = "A",
) -> dict:
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "PLAN_PROD_KEY": ppk,
        "ST": proc_time, "EQP_MODEL": eqp_model, "WF_QTY": wf_qty,
    }


def _write_sample_bundle(
    output_dir: Path,
    schedule: List[dict],
    availability: List[dict],
    plan: List[dict],
    flow: List[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in [
        (CONFIG.path.schedule_file, schedule),
        (CONFIG.path.availability_file, availability),
        (CONFIG.path.plan_file, plan),
        (CONFIG.path.flow_file, flow),
    ]:
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _build_default_sample() -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    schedule = [
        _schedule_row("EQP001", "LOT001", "CAR001", "PPK001", 1, 0, 120),
        _schedule_row("EQP001", "LOT002", "CAR002", "PPK002", 1, 120, 210),
        _schedule_row("EQP001", "LOT003", "CAR003", "PPK001", 1, 210, 330),
        _schedule_row("EQP002", "LOT004", "CAR004", "PPK003", 1, 0, 105),
        _schedule_row("EQP002", "LOT005", "CAR005", "PPK001", 1, 105, 225),
        _schedule_row("EQP002", "LOT006", "CAR006", "PPK002", 1, 225, 345),
        _schedule_row("EQP003", "LOT007", "CAR007", "PPK002", 2, 0, 180),
        _schedule_row("EQP003", "LOT008", "CAR008", "PPK003", 2, 180, 300),
        _schedule_row("EQP003", "LOT009", "CAR009", "PPK001", 2, 300, 480),
    ]
    availability = [
        _avail_row("EQP001", "LOT001", "PPK001", 120),
        _avail_row("EQP001", "LOT002", "PPK002", 90),
        _avail_row("EQP001", "LOT003", "PPK001", 120),
        _avail_row("EQP001", "LOT004", "PPK003", 105),
        _avail_row("EQP002", "LOT004", "PPK003", 105),
        _avail_row("EQP002", "LOT005", "PPK001", 120),
        _avail_row("EQP002", "LOT006", "PPK002", 120),
        _avail_row("EQP003", "LOT007", "PPK002", 180),
        _avail_row("EQP003", "LOT008", "PPK003", 120),
        _avail_row("EQP003", "LOT009", "PPK001", 180),
    ]
    plan = [
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 75, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK002", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 75, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK003", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 75, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK002", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK003", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
    ]
    flow = [
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": "PPK003", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK003", "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]
    return schedule, availability, plan, flow


def _build_single_heavy_wip_sample() -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    단일 제품(PPK001) · 공정 2개 · 재공 다량 시나리오
      - OPER001: 8 LOT, 초기 START 간격 120분, ST(소요) 120분, EQP001·002
      - OPER002: 8 LOT, 초기 START 간격 60분, ST(소요) 90분 고정, EQP001·002·003 전체
      - 공정별 200매(25매×8) 계획
    """
    ppk = "PPK001"
    wf_qty = 25
    n_lots = 8
    proc1 = 120
    proc2 = 90
    st_step1 = 120
    st_step2 = 60
    oper1_eqps = ("EQP001", "EQP002")
    all_eqps = ("EQP001", "EQP002", "EQP003")

    schedule: List[dict] = []
    availability: List[dict] = []

    for i in range(n_lots):
        lot_id = f"LOT{i + 1:03d}"
        st_min = i * st_step1
        eqp = "EQP001" if i < n_lots // 2 else "EQP002"
        schedule.append(_schedule_row(
            eqp, lot_id, f"CAR{i + 1:03d}", ppk, 1, st_min, st_min + proc1,
        ))
        for eqp_id in oper1_eqps:
            availability.append(_avail_row(eqp_id, lot_id, ppk, proc1))

    for i in range(n_lots):
        lot_id = f"LOT{101 + i}"
        st_min = i * st_step2
        schedule.append(_schedule_row(
            "EQP003", lot_id, f"CAR{101 + i}", ppk, 2, st_min, st_min + proc2,
        ))
        for eqp_id in all_eqps:
            availability.append(_avail_row(eqp_id, lot_id, ppk, proc2))

    total = n_lots * wf_qty
    plan = [
        {"PLAN_PROD_KEY": ppk, "OPER_ID": "OPER001",
         "D0_PLAN_QTY": total, "D1_PLAN_QTY": total + 50, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": ppk, "OPER_ID": "OPER002",
         "D0_PLAN_QTY": total, "D1_PLAN_QTY": total + 50, "PLAN_PRIORITY": 1},
    ]
    flow = [
        {"PLAN_PROD_KEY": ppk, "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": ppk, "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]
    return schedule, availability, plan, flow


SampleBuilder = Callable[[], Tuple[List[dict], List[dict], List[dict], List[dict]]]

SAMPLE_SCENARIOS: Dict[str, dict] = {
    "default": {
        "name": "기본 (3제품)",
        "description": "PPK 3종, LOT 9개, 혼합 priority",
        "default_folder": "input",
        "build": _build_default_sample,
    },
    "single_heavy_wip": {
        "name": "단일제품 ST½ 재공다량",
        "description": "PPK001 단일 제품, OPER002 ST 90분·전 설비, OPER002 START 간격 OPER001의 1/2",
        "default_folder": "single_heavy_wip",
        "build": _build_single_heavy_wip_sample,
    },
}


def list_sample_scenarios() -> List[dict]:
    return [
        {
            "id": sid,
            "name": meta["name"],
            "description": meta["description"],
            "default_folder": meta["default_folder"],
        }
        for sid, meta in SAMPLE_SCENARIOS.items()
    ]


# ── 샘플 데이터 생성 ────────────────────────────────────────────────────────────

def generate_sample_data(output_dir: Path = None, scenario: str = "default") -> Path:
    """
    목적: 시나리오별 샘플 JSON 4개를 external/<폴더>/ 에 생성
    Input:  output_dir=None → 시나리오 기본 폴더
            scenario="default" | "single_heavy_wip"
    """
    if scenario not in SAMPLE_SCENARIOS:
        raise ValueError(
            f"알 수 없는 시나리오: {scenario}. "
            f"사용 가능: {', '.join(SAMPLE_SCENARIOS)}"
        )
    meta = SAMPLE_SCENARIOS[scenario]
    d = output_dir or (EXTERNAL_DIR / meta["default_folder"])
    schedule, availability, plan, flow = meta["build"]()
    _write_sample_bundle(d, schedule, availability, plan, flow)
    print(f"[loader] 샘플 생성 ({scenario}) → {d}")
    return d
