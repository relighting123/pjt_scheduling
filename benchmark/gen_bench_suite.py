"""
대표 벤치마크 묶음(BENCH_SUITE) 생성기 — 변별력 있는 시나리오
=========================================================
단순 진행률 균형 휴리스틱(Min-Progress)이 '항상 최적'이 되지 않도록,
비대칭·제약 시나리오를 4개 카테고리로 구성한다.

  ① 대칭 기준 (SYM)    : 설비=제품, 전담하면 최적 → 대조군
  ② 제품 과잉 (OVER)   : 제품 > 설비 → 전환 불가피, '어떤 제품을 전환할지'가 KPI를 가름
  ③ 부하 불균등 (LOAD) : 제품별 계획량·처리시간 상이 → 진행률 균형이 곧 최적이 아님
  ④ 전환 과중 (CONV)   : 전환시간 ≫ 처리시간 → 전환 1회 손실이 커 신중한 블록 점유가 유리

공통: 단일 공정(OPER001), 단일 모델(A). 캐리어를 여러 설비에 분산 배치(home assign, mix).
시뮬 시간 = 최소 전환만 했을 때 전량 생산 가능한 길이 → 초과 전환은 생산 손실로 직결.

SYM 카테고리만 mix=0.0(분산 없음, 캐리어별 discrete_arrange 홈이 곧 담당 설비)으로
고정한다: simulation/simulator.py의 _lot_conv_discrete_eligible()이 '전환 불필요한
배정도 discrete(EQP×carrier 실측 조합)로 등록된 홈 설비가 아니면 배정 불가'로 막기
때문에, mix>0로 캐리어 홈을 다른 설비에 흩어두면 한 설비가 그 제품 전량을 맡을 방법이
없어져 "전담하면 전환 0" 이라는 대조군 전제 자체가 성립하지 않는다(실제로 mix=0.85/0.9였을
때는 전담을 그대로 구현한 DedicationAgent 오라클조차 SYM_5x5에서 전환 17회가 나왔다 —
정책 품질과 무관한 데이터 생성 버그). OVER/LOAD/CONV 카테고리는 애초에 전환이 불가피하거나
전담이 곧 최적이 아니므로 이 제약이 KPI 변별력에 문제가 되지 않아 mix를 그대로 둔다.

⚠️ meta의 `min_conv`는 `max(n_ppk - n_eqp, 0)` — **제품 수가 설비 수를 넘을 때만**
전환이 필요하다는 가정이다. mix>0인 데이터셋에서는 이 값이 실제 하한이 아니다.
위 discrete 제약 때문에, 셋업이 맞는 설비라도 그 carrier의 홈 설비가 아니면
무전환 배정이 막히고, 결국 "전환 1회 → 임의 carrier 1장 + 그 설비에 홈 배정된
같은 제품 carrier들"이 한 셋업의 상한이 된다. mix=0.9면 제품당 홈 carrier가
설비마다 1~2장뿐이라 전환이 구조적으로 많이 발생한다(LOAD_skew는 min_conv=0으로
표시되지만 어떤 알고리즘도 12회 아래로 내려가지 못한다). mix>0 데이터셋의
`min_conv`는 참고값으로만 보고, 알고리즘 비교는 상대 점수로 판단할 것.
"""
import json
import math
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
# 모든 합성 벤치마크/학습 시나리오(BENCH_SUITE/HOLDOUT_SUITE/TRAIN_POOL/
# TOOL_CHANGE_BENCH)를 실FAC_ID와 겹치지 않는 이 하나의 가짜 FAC_ID 아래
# 시나리오ID=period 폴더로 모은다(과거엔 시나리오마다 최상위 폴더 하나씩 써서
# 실FAC_ID처럼 보였다).
BASE_FAC_ID = "BASE"
OPER = "OPER001"
MODEL = "A"
TEMP = "T600"

# 각 벤치마크 스펙
#  id, category, n_eqp, n_ppk, carriers(int|list), st(int|list), conv, mix, max_tool, desc, tests
SPECS = [
    dict(id="SYM_3x3", cat="대칭 기준", n_eqp=3, n_ppk=3, carriers=8, st=60, conv=60, mix=0.0,
         tool=99, desc="3설비·3제품·각 8캐리어. 설비당 1제품 전담 시 전환 0으로 최적.",
         tests="기본 전담 최적해 도달 (대조군)"),
    dict(id="SYM_5x5", cat="대칭 기준", n_eqp=5, n_ppk=5, carriers=6, st=48, conv=48, mix=0.0,
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
    # bid로부터 결정적 시드를 뽑아야 재생성해도 같은 데이터가 나온다.
    # 파이썬 내장 hash()는 문자열에 대해 프로세스마다 무작위로 솔트가
    # 들어가므로(PYTHONHASHSEED 보안 랜덤화) bid가 같아도 실행할 때마다
    # 다른 값이 나와, 이 함수를 재실행하면 carrier의 home 배정(따라서
    # 전환 횟수·정답 스케줄)이 매번 바뀌는 버그가 있었다. zlib.crc32는
    # 프로세스와 무관하게 항상 같은 값을 낸다.
    rng = random.Random(zlib.crc32(bid.encode()) & 0xFFFF)
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
    out = SUITE_ROOT / BASE_FAC_ID / "train" / bid / "input"
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
