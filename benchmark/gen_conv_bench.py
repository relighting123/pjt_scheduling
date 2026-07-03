"""
CONV_BENCH 데이터셋 생성기
=========================================================
시나리오: 전환 비용이 직접 생산 수량 손실로 나타나는 단순 3×3 시나리오

핵심 수치
  ST  = 60분/carrier  (장당 처리시간)
  CONV= 60분          (LOT_CD 변경 시 setup 소요)
  SIM = 480분 (8h)
  → EQP당 최대 처리 가능 carrier = 480/60 = 8개 (딱 맞음)
  → 전환 1회당 60분 소모 = 캐리어 1개 손실 (직관적)

구성
  EQP: 3대 (A모델, 동일 성능)
  PPK: 3종 × 각 다른 LOT_CD (LC_A / LC_B / LC_C)
  WIP: 8 carrier/PPK = 24 total
  Plan: 8/PPK

최적해 (이론)
  EQP001 → PPK001 전담 (8 carrier, 0 conv, LC_A 고정)
  EQP002 → PPK002 전담 (8 carrier, 0 conv, LC_B 고정)
  EQP003 → PPK003 전담 (8 carrier, 0 conv, LC_C 고정)
  생산: 24개, 전환: 0회

전환 비용 수량화
  0 conv → 24개 (최적)
  2 conv → 22개 (EQP 1대에서 2회)
  6 conv → 18개 (각 EQP 2회씩)
  n conv → max(24-n, 0)개

discrete_arrange 분배 전략
  - 각 EQP의 "홈" lot를 의도적으로 3 PPK 혼합 배정
  - 이로 인해 단순 알고리즘(홈 lot 순서대로)은 PPK 전환 발생
  - 스마트 알고리즘만 비홈(abstract) lot를 전략적으로 선택해 전담 달성 가능
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data/dataset/CONV_BENCH/train/20260629000000/input"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
PPKS  = ["PPK001", "PPK002", "PPK003"]
EQPS  = ["EQP001", "EQP002", "EQP003"]
OPER  = "OPER001"
MODEL = "A"
ST    = 60     # 분/carrier
N_CARRIER_PER_PPK = 8  # EQP당 capacity(8h/60min)와 정확히 일치 → 전환 1회=1개 손실

# 각 PPK별 LOT_CD (다르면 PPK 전환 시 setup(전환) 발생)
LOT_CD_BY_PPK = {"PPK001": "LC_A", "PPK002": "LC_B", "PPK003": "LC_C"}
TEMP = "T600"

# ── LOT 목록 ─────────────────────────────────────────────────────────────────
# LOT001-008: PPK001(LC_A), LOT009-016: PPK002(LC_B), LOT017-024: PPK003(LC_C)
lots = []
for pi, ppk in enumerate(PPKS):
    for ci in range(N_CARRIER_PER_PPK):
        lot_num = pi * N_CARRIER_PER_PPK + ci + 1
        lots.append({
            "lot_id":  f"LOT{lot_num:03d}",
            "ppk":     ppk,
            "lot_cd":  LOT_CD_BY_PPK[ppk],
            "car_id":  f"CAR{lot_num:03d}",
        })

# ── discrete_arrange 홈 배정 (의도적 혼합) ────────────────────────────────────
# 각 PPK의 8개 lot을 EQP에 3-3-2, 2-3-3, 3-2-3 패턴으로 나눔
# → 각 EQP의 홈 lot 구성: EQP001=(PPK1:3,PPK2:2,PPK3:3), EQP002=(3,3,2), EQP003=(2,3,3)
HOME_ASSIGN = {
    "PPK001": ["EQP001", "EQP001", "EQP001",
               "EQP002", "EQP002", "EQP002",
               "EQP003", "EQP003"],
    "PPK002": ["EQP001", "EQP001",
               "EQP002", "EQP002", "EQP002",
               "EQP003", "EQP003", "EQP003"],
    "PPK003": ["EQP001", "EQP001", "EQP001",
               "EQP002", "EQP002",
               "EQP003", "EQP003", "EQP003"],
}

ppk_lot_map = {ppk: [] for ppk in PPKS}
for lot in lots:
    ppk_lot_map[lot["ppk"]].append(lot)

discrete_rows = []
for ppk in PPKS:
    ppk_lots   = ppk_lot_map[ppk]
    home_eqps  = HOME_ASSIGN[ppk]
    for lot, home_eqp in zip(ppk_lots, home_eqps):
        discrete_rows.append({
            "EQP_ID":        home_eqp,
            "LOT_ID":        lot["lot_id"],
            "PLAN_PROD_ATTR_VAL": ppk,
            "OPER_ID":       OPER,
            "ST":            ST,
            "EQP_MODEL_CD":  MODEL,
            "WF_QTY":        1,
            "SEQ":           1,
            "CARRIER_ID":    lot["car_id"],
        })

# ── abstract_arrange ─────────────────────────────────────────────────────────
abstract_rows = [
    {"EQP_MODEL_CD": MODEL, "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": OPER, "ST": ST}
    for ppk in PPKS
]

# ── lot_master (LOT_CD 개별 지정) ─────────────────────────────────────────────
lot_master_rows = [
    {"LOT_ID": lot["lot_id"], "LOT_CD": lot["lot_cd"], "TEMP": TEMP}
    for lot in lots
]

# ── plan ─────────────────────────────────────────────────────────────────────
plan_rows = [
    {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": OPER,
     "D0_PLAN_QTY": N_CARRIER_PER_PPK, "D1_PLAN_QTY": N_CARRIER_PER_PPK,
     "PLAN_PRIORITY": 1}
    for ppk in PPKS
]

# ── flow / split / batch_info / tool_capacity ─────────────────────────────────
flow_rows = [
    {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": OPER}
    for ppk in PPKS
]
split_rows = [
    {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": OPER, "EQP_MODEL_CD": MODEL, "SPLIT_QTY": 1}
    for ppk in PPKS
]
batch_rows = [
    {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": OPER, "LOT_CD": LOT_CD_BY_PPK[ppk], "TEMP": TEMP}
    for ppk in PPKS
]
tool_rows = [{"LOT_CD": lc, "TEMP": TEMP, "MAX_TOOL": 99}
             for lc in LOT_CD_BY_PPK.values()]

# ── 파일 저장 ─────────────────────────────────────────────────────────────────
files = {
    "abstract_arrange.json": abstract_rows,
    "discrete_arrange.json": discrete_rows,
    "lot_master.json":       lot_master_rows,
    "plan.json":             plan_rows,
    "flow.json":             flow_rows,
    "split.json":            split_rows,
    "batch_info.json":       batch_rows,
    "tool_capacity.json":    tool_rows,
}
for fname, data in files.items():
    path = OUT_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  생성: {path.name}  ({len(data)} rows)")

print(f"\nCONV_BENCH 데이터셋 생성 완료 → {OUT_DIR}")
print(f"  EQP={EQPS}, PPK={PPKS}")
print(f"  WIP={N_CARRIER_PER_PPK}/PPK × {len(PPKS)}PPK = {N_CARRIER_PER_PPK*len(PPKS)} lots")
print(f"  ST={ST}분, Conv=60분, Sim=480분")
print(f"  이론 최적: 24개 생산, 0 전환 (전환 1회=1개 손실)")

if __name__ == "__main__":
    pass  # 이 파일을 직접 실행하면 데이터 생성
