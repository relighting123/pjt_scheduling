"""
data/loader.py – JSON 데이터 로딩 & 샘플 데이터 생성
DB 조회 결과가 external/input/ 폴더에 JSON으로 저장되어 있다고 가정합니다.
샘플 데이터 생성 함수로 외부 DB 없이 개발·테스트할 수 있습니다.
"""
import json
from pathlib import Path
from typing import Dict, List

from config import CONFIG
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
    Input:  input_dir=Path("../external/input")
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


# ── 샘플 데이터 생성 ────────────────────────────────────────────────────────────

def generate_sample_data(output_dir: Path = None) -> None:
    """
    목적: 개발·테스트용 샘플 JSON 파일 4개를 external/input/에 생성
    Input:  output_dir=None (CONFIG 기본값 사용)
    Output: 없음 (파일 생성)

    샘플 구성:
      - EQP 3대: EQP001, EQP002, EQP003
      - LOT 9개: LOT001~LOT009  (각 LOT는 OPER별 재공)
      - 제품 3종: PPK001, PPK002, PPK003
      - 공정 2종: OPER001(포토), OPER002(식각)
      - 시작 시각: 2024-01-15 08:00:00
    """
    d = output_dir or CONFIG.path.input_dir
    d.mkdir(parents=True, exist_ok=True)

    # ── schedule.json ──────────────────────────────────────────────────────────
    schedule = [
        {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
         "PLAN_PROD_KEY": "PPK001", "ST": "A", "SEQ": 1,
         "STARTTM": "2024-01-15 08:00:00", "ENDTM": "2024-01-15 10:00:00"},
        {"EQP_ID": "EQP001", "LOT_ID": "LOT002", "CARRIER_ID": "CAR002",
         "PLAN_PROD_KEY": "PPK002", "ST": "A", "SEQ": 1,
         "STARTTM": "2024-01-15 10:00:00", "ENDTM": "2024-01-15 11:30:00"},
        {"EQP_ID": "EQP001", "LOT_ID": "LOT003", "CARRIER_ID": "CAR003",
         "PLAN_PROD_KEY": "PPK001", "ST": "A", "SEQ": 1,
         "STARTTM": "2024-01-15 11:30:00", "ENDTM": "2024-01-15 13:30:00"},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT004", "CARRIER_ID": "CAR004",
         "PLAN_PROD_KEY": "PPK003", "ST": "A", "SEQ": 1,
         "STARTTM": "2024-01-15 08:00:00", "ENDTM": "2024-01-15 09:45:00"},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT005", "CARRIER_ID": "CAR005",
         "PLAN_PROD_KEY": "PPK001", "ST": "A", "SEQ": 1,
         "STARTTM": "2024-01-15 09:45:00", "ENDTM": "2024-01-15 11:45:00"},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT006", "CARRIER_ID": "CAR006",
         "PLAN_PROD_KEY": "PPK002", "ST": "A", "SEQ": 1,
         "STARTTM": "2024-01-15 11:45:00", "ENDTM": "2024-01-15 13:15:00"},
        {"EQP_ID": "EQP003", "LOT_ID": "LOT007", "CARRIER_ID": "CAR007",
         "PLAN_PROD_KEY": "PPK002", "ST": "A", "SEQ": 2,
         "STARTTM": "2024-01-15 08:00:00", "ENDTM": "2024-01-15 11:00:00"},
        {"EQP_ID": "EQP003", "LOT_ID": "LOT008", "CARRIER_ID": "CAR008",
         "PLAN_PROD_KEY": "PPK003", "ST": "A", "SEQ": 2,
         "STARTTM": "2024-01-15 11:00:00", "ENDTM": "2024-01-15 13:00:00"},
        {"EQP_ID": "EQP003", "LOT_ID": "LOT009", "CARRIER_ID": "CAR009",
         "PLAN_PROD_KEY": "PPK001", "ST": "A", "SEQ": 2,
         "STARTTM": "2024-01-15 13:00:00", "ENDTM": "2024-01-15 16:00:00"},
    ]

    # ── availability.json ──────────────────────────────────────────────────────
    availability = [
        # EQP001 – OPER001 가능 (PPK001, PPK002, PPK003)
        {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "PLAN_PROD_KEY": "PPK001", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP001", "LOT_ID": "LOT002", "PLAN_PROD_KEY": "PPK002", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP001", "LOT_ID": "LOT003", "PLAN_PROD_KEY": "PPK001", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP001", "LOT_ID": "LOT004", "PLAN_PROD_KEY": "PPK003", "ST": "A", "WF_QTY": 25},
        # EQP002 – OPER001 가능
        {"EQP_ID": "EQP002", "LOT_ID": "LOT004", "PLAN_PROD_KEY": "PPK003", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT005", "PLAN_PROD_KEY": "PPK001", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP002", "LOT_ID": "LOT006", "PLAN_PROD_KEY": "PPK002", "ST": "A", "WF_QTY": 25},
        # EQP003 – OPER002 가능
        {"EQP_ID": "EQP003", "LOT_ID": "LOT007", "PLAN_PROD_KEY": "PPK002", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP003", "LOT_ID": "LOT008", "PLAN_PROD_KEY": "PPK003", "ST": "A", "WF_QTY": 25},
        {"EQP_ID": "EQP003", "LOT_ID": "LOT009", "PLAN_PROD_KEY": "PPK001", "ST": "A", "WF_QTY": 25},
    ]

    # ── plan.json ─────────────────────────────────────────────────────────────
    plan = [
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 75, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK002", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 75,  "PLAN_PRIORITY": 2},
        {"PLAN_PROD_KEY": "PPK003", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50,  "PLAN_PRIORITY": 3},
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 75,  "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK002", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50,  "PLAN_PRIORITY": 2},
        {"PLAN_PROD_KEY": "PPK003", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50,  "PLAN_PRIORITY": 3},
    ]

    # ── flow.json ─────────────────────────────────────────────────────────────
    flow = [
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": "PPK003", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK003", "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]

    for filename, data in [
        (CONFIG.path.schedule_file,     schedule),
        (CONFIG.path.availability_file, availability),
        (CONFIG.path.plan_file,         plan),
        (CONFIG.path.flow_file,         flow),
    ]:
        with open(d / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[loader] 샘플 데이터 생성 완료 → {d}")
