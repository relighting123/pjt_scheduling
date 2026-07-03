"""
대표 벤치마크 묶음(BENCH_SUITE) 생성기 — 변별력 있는 시나리오
=========================================================
단순 진행률 균형 휴리스틱(Min-Progress)이 '항상 최적'이 되지 않도록,
비대칭·제약 시나리오를 4개 카테고리로 구성한다.

  ① 대칭 기준 (SYM)    : 설비=제품, 전담하면 최적 → 대조군
  ② 제품 과잉 (OVER)   : 제품 > 설비 → 전환 불가피, '어떤 제품을 전환할지'가 KPI를 가름
  ③ 부하 불균등 (LOAD) : 제품별 계획량·처리시간 상이 → 진행률 균형이 곧 최적이 아님
  ④ 전환 과중 (CONV)   : 전환시간 ≫ 처리시간 → 전환 1회 손실이 커 신중한 블록 점유가 유리

공통: 단일 공정(OPER001), 단일 모델(A). 캐리어를 여러 설비에 분산 배치(home assign).
시뮬 시간 = 최소 전환만 했을 때 전량 생산 가능한 길이 → 초과 전환은 생산 손실로 직결.
"""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
TIMEKEY = "20260629000000"
OPER = "OPER001"
MODEL = "A"
TEMP = "T600"

# 각 벤치마크 스펙
#  id, category, n_eqp, n_ppk, carriers(int|list), st(int|list), conv, mix, max_tool, desc, tests
SPECS = [
    dict(id="SYM_3x3", cat="대칭 기준", n_eqp=3, n_ppk=3, carriers=8, st=60, conv=60, mix=0.9,
         tool=99, desc="3설비·3제품·각 8캐리어. 설비당 1제품 전담 시 전환 0으로 최적.",
         tests="기본 전담 최적해 도달 (대조군)"),
    dict(id="SYM_5x5", cat="대칭 기준", n_eqp=5, n_ppk=5, carriers=6, st=48, conv=48, mix=0.85,
         tool=99, desc="5설비·5제품·각 6캐리어. 규모를 키운 대칭 케이스.",
         tests="규모 확장 시 전담 유지"),
    dict(id="OVER_5p3", cat="제품 과잉", n_eqp=3, n_ppk=5, carriers=4, st=60, conv=60, mix=0.9,
         tool=99, desc="3설비가 5제품을 처리 → 최소 2회 전환 불가피.",
         tests="불가피한 전환을 최소 횟수로 묶기"),
    dict(id="OVER_7p4", cat="제품 과잉", n_eqp=4, n_ppk=7, carriers=4, st=45, conv=45, mix=0.9,
         tool=99, desc="4설비가 7제품을 처리 → 최소 3회 전환. 분배 난이도 높음.",
         tests="다수 제품 분배 + 전환 최소화"),
    dict(id="LOAD_skew", cat="부하 불균등", n_eqp=4, n_ppk=4, carriers=[14, 8, 4, 4], st=60, conv=60,
         mix=0.9, tool=99, desc="제품별 계획량 14·8·4·4로 편중. 균등 분배가 곧 최적이 아님.",
         tests="물량 편중에서의 균형 배분"),
    dict(id="LOAD_stmix", cat="부하 불균등", n_eqp=4, n_ppk=4, carriers=8, st=[30, 45, 60, 90],
         conv=60, mix=0.9, tool=99, desc="제품별 처리시간 30·45·60·90으로 이질. 처리량 차이 발생.",
         tests="처리시간 이질성 하의 효율 배분"),
    dict(id="CONV_x2", cat="전환 과중", n_eqp=4, n_ppk=4, carriers=6, st=45, conv=90, mix=0.9,
         tool=99, desc="전환시간(90)=처리시간(45)의 2배. 전환 1회 손실이 2캐리어.",
         tests="큰 전환비용 하의 전담 가치"),
    dict(id="CONV_x3", cat="전환 과중", n_eqp=3, n_ppk=5, carriers=4, st=40, conv=120, mix=0.9,
         tool=99, desc="제품 과잉 + 전환시간(120)=처리시간(40)의 3배. 최난도.",
         tests="전환 과중 + 제품 과잉 복합"),
]


def _as_list(v, n):
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


def gen_one(spec):
    bid = spec["id"]; ne = spec["n_eqp"]; npk = spec["n_ppk"]
    carriers = _as_list(spec["carriers"], npk)
    sts = _as_list(spec["st"], npk)
    conv = spec["conv"]; mix = spec["mix"]; tool = spec["tool"]
    ppks = [f"PPK{i+1:03d}" for i in range(npk)]
    eqps = [f"EQP{i+1:03d}" for i in range(ne)]
    lot_cd_by_ppk = {ppk: f"LC_{chr(65+i)}" for i, ppk in enumerate(ppks)}

    # 시뮬 시간: 최소 전환만 했을 때 전량 생산 가능한 길이
    total_work = sum(c * s for c, s in zip(carriers, sts))
    min_conv = max(npk - ne, 0)
    sim = int(math.ceil(total_work / ne) + min_conv * conv)
    sim = int(math.ceil(sim / 10.0) * 10)  # 10분 단위 정렬

    # LOT 목록
    lots = []
    num = 0
    for pi, ppk in enumerate(ppks):
        for ci in range(carriers[pi]):
            num += 1
            lots.append(dict(lot_id=f"LOT{num:03d}", ppk=ppk, pi=pi,
                             lot_cd=lot_cd_by_ppk[ppk], car_id=f"CAR{num:03d}", st=sts[pi]))

    # home 배정: mix 높을수록 분산
    rng = random.Random(abs(hash(bid)) & 0xFFFF)
    discrete = []
    by_ppk = {ppk: [l for l in lots if l["ppk"] == ppk] for ppk in ppks}
    for pi, ppk in enumerate(ppks):
        for ci, lot in enumerate(by_ppk[ppk]):
            if rng.random() < mix:
                he = eqps[(pi + ci) % ne]
            else:
                he = eqps[pi % ne]
            discrete.append(dict(EQP_ID=he, LOT_ID=lot["lot_id"], PLAN_PROD_ATTR_VAL=ppk,
                                 OPER_ID=OPER, ST=lot["st"], EQP_MODEL_CD=MODEL,
                                 WF_QTY=1, SEQ=1, CARRIER_ID=lot["car_id"]))

    abstract = [dict(EQP_MODEL_CD=MODEL, PLAN_PROD_ATTR_VAL=ppk, OPER_ID=OPER, ST=sts[pi])
                for pi, ppk in enumerate(ppks)]
    lot_master = [dict(LOT_ID=l["lot_id"], LOT_CD=l["lot_cd"], TEMP=TEMP) for l in lots]
    plan = [dict(PLAN_PROD_ATTR_VAL=ppk, OPER_ID=OPER, D0_PLAN_QTY=carriers[pi],
                 D1_PLAN_QTY=carriers[pi], PLAN_PRIORITY=1) for pi, ppk in enumerate(ppks)]
    flow = [dict(PLAN_PROD_ATTR_VAL=ppk, OPER_SEQ=1, OPER_ID=OPER) for ppk in ppks]
    split = [dict(PLAN_PROD_ATTR_VAL=ppk, OPER_ID=OPER, EQP_MODEL_CD=MODEL, SPLIT_QTY=1) for ppk in ppks]
    batch = [dict(PLAN_PROD_ATTR_VAL=ppk, OPER_ID=OPER, LOT_CD=lot_cd_by_ppk[ppk], TEMP=TEMP) for ppk in ppks]
    tool_rows = [dict(LOT_CD=lc, TEMP=TEMP, MAX_TOOL=tool) for lc in lot_cd_by_ppk.values()]

    files = {
        "abstract_arrange.json": abstract, "discrete_arrange.json": discrete,
        "lot_master.json": lot_master, "plan.json": plan, "flow.json": flow,
        "split.json": split, "batch_info.json": batch, "tool_capacity.json": tool_rows,
    }
    out = SUITE_ROOT / bid / "train" / TIMEKEY / "input"
    out.mkdir(parents=True, exist_ok=True)
    for fn, data in files.items():
        with open(out / fn, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return dict(id=bid, cat=spec["cat"], n_eqp=ne, n_ppk=npk,
                carriers=carriers, st=sts, conv=conv, sim=sim,
                total=sum(carriers), min_conv=min_conv, mix=mix, tool=tool,
                desc=spec["desc"], tests=spec["tests"], dir=str(out))


def main():
    meta = []
    for spec in SPECS:
        m = gen_one(spec)
        meta.append(m)
        print(f"  {m['id']:<12} [{m['cat']:<6}] {m['n_eqp']}설비×{m['n_ppk']}제품 "
              f"총{m['total']}캐리어 sim={m['sim']} 최소전환={m['min_conv']}  · {m['tests']}")
    with open(SUITE_ROOT / "bench_suite_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nBENCH_SUITE {len(meta)}종 생성 완료 → {SUITE_ROOT}")
    return meta


if __name__ == "__main__":
    main()
