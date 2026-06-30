"""
다양한 테스트 데이터셋 묶음(BENCH_SUITE) 생성기
=========================================================
CONV_BENCH(3×3·ST60)를 일반화해, 설비/제품 수·캐리어·ST·전환시간·혼합도를
달리한 10개 데이터셋을 만든다. 모든 셋은 동일 시뮬레이터·KPI로 평가된다.

공통 설계
  - 단일 공정(OPER001), 단일 설비모델(A) → 설비 n대가 제품 n종을 전담하면 최적
  - 각 제품의 캐리어를 여러 설비에 분산 배치(home assign) → 단순 규칙은 전환 불가피
  - 설비당 처리 가능 캐리어 = SIM/ST. 캐리어 수를 이에 맞추면 전환 1회=손실 1개

각 데이터셋 파라미터: (이름, 설비수=제품수 n, 제품당 캐리어 c, ST, 전환분, 혼합강도)
  혼합강도 mix: 0.0=전담배치(쉬움) … 1.0=최대 분산(어려움)
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
TIMEKEY = "20260629000000"
OPER = "OPER001"
MODEL = "A"
TEMP = "T600"

# (name, n_eqp(=n_ppk), carriers_per_ppk, ST, conv_min, mix)
SPECS = [
    ("BENCH01_3x3_st60",   3,  8, 60, 60, 0.9),
    ("BENCH02_4x4_st60",   4,  6, 60, 60, 0.8),
    ("BENCH03_5x5_st45",   5,  8, 45, 45, 0.8),
    ("BENCH04_2x2_st40",   2, 12, 40, 40, 1.0),
    ("BENCH05_3x3_st48",   3, 10, 48, 48, 0.9),
    ("BENCH06_6x6_st72",   6,  5, 72, 72, 0.7),
    ("BENCH07_4x4_cv50",   4,  8, 50, 50, 0.85),
    ("BENCH08_3x3_st30",   3, 16, 30, 30, 0.9),
    ("BENCH09_5x5_cv90",   5,  6, 60, 90, 0.8),
    ("BENCH10_4x4_st36",   4, 10, 36, 36, 0.85),
]


def gen_one(name, n, c, st, conv, mix):
    ppks = [f"PPK{i+1:03d}" for i in range(n)]
    eqps = [f"EQP{i+1:03d}" for i in range(n)]
    lot_cd_by_ppk = {ppk: f"LC_{chr(65+i)}" for i, ppk in enumerate(ppks)}
    sim = c * st  # 설비당 정확히 c 캐리어 처리 가능

    # LOT 목록
    lots = []
    num = 0
    for pi, ppk in enumerate(ppks):
        for ci in range(c):
            num += 1
            lots.append({
                "lot_id": f"LOT{num:03d}", "ppk": ppk,
                "lot_cd": lot_cd_by_ppk[ppk], "car_id": f"CAR{num:03d}",
            })

    # home 배정: mix가 높을수록 각 제품의 캐리어를 여러 설비에 고르게 분산
    rng = random.Random(hash(name) & 0xFFFF)
    home = {}
    for pi, ppk in enumerate(ppks):
        assign = []
        for ci in range(c):
            if rng.random() < mix:
                # 분산: 라운드로빈으로 다른 설비에
                assign.append(eqps[(pi + ci) % n])
            else:
                # 전담: 자기 홈 설비
                assign.append(eqps[pi])
        home[ppk] = assign

    ppk_lots = {ppk: [l for l in lots if l["ppk"] == ppk] for ppk in ppks}
    discrete = []
    for ppk in ppks:
        for lot, he in zip(ppk_lots[ppk], home[ppk]):
            discrete.append({
                "EQP_ID": he, "LOT_ID": lot["lot_id"], "PLAN_PROD_KEY": ppk,
                "OPER_ID": OPER, "ST": st, "EQP_MODEL_CD": MODEL,
                "WF_QTY": 1, "SEQ": 1, "CARRIER_ID": lot["car_id"],
            })

    abstract = [{"EQP_MODEL_CD": MODEL, "PLAN_PROD_KEY": ppk, "OPER_ID": OPER, "ST": st}
                for ppk in ppks]
    lot_master = [{"LOT_ID": l["lot_id"], "LOT_CD": l["lot_cd"], "TEMP": TEMP} for l in lots]
    plan = [{"PLAN_PROD_KEY": ppk, "OPER_ID": OPER, "D0_PLAN_QTY": c, "D1_PLAN_QTY": c,
             "PLAN_PRIORITY": 1} for ppk in ppks]
    flow = [{"PLAN_PROD_KEY": ppk, "OPER_SEQ": 1, "OPER_ID": OPER} for ppk in ppks]
    split = [{"PLAN_PROD_KEY": ppk, "OPER_ID": OPER, "EQP_MODEL_CD": MODEL, "SPLIT_QTY": 1}
             for ppk in ppks]
    batch = [{"PLAN_PROD_KEY": ppk, "OPER_ID": OPER, "LOT_CD": lot_cd_by_ppk[ppk], "TEMP": TEMP}
             for ppk in ppks]
    tool = [{"LOT_CD": lc, "TEMP": TEMP, "MAX_TOOL": 99} for lc in lot_cd_by_ppk.values()]

    files = {
        "abstract_arrange.json": abstract, "discrete_arrange.json": discrete,
        "lot_master.json": lot_master, "plan.json": plan, "flow.json": flow,
        "split.json": split, "batch_info.json": batch, "tool_capacity.json": tool,
    }
    out = SUITE_ROOT / name / "train" / TIMEKEY / "input"
    out.mkdir(parents=True, exist_ok=True)
    for fn, data in files.items():
        with open(out / fn, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        "name": name, "n_eqp": n, "n_ppk": n, "carriers": c, "st": st,
        "conv": conv, "sim": sim, "total": n * c, "mix": mix, "dir": str(out),
    }


def main():
    meta = []
    for spec in SPECS:
        m = gen_one(*spec)
        meta.append(m)
        print(f"  {m['name']:<20} {m['n_eqp']}x{m['n_ppk']} "
              f"c={m['carriers']} ST={m['st']} conv={m['conv']} "
              f"sim={m['sim']} total={m['total']} mix={m['mix']}")
    meta_path = SUITE_ROOT / "bench_suite_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nBENCH_SUITE {len(meta)}개 데이터셋 생성 완료 → {SUITE_ROOT}")
    return meta


if __name__ == "__main__":
    main()
