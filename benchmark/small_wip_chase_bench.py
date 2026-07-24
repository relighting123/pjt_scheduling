"""
benchmark/small_wip_chase_bench.py — "재공 소량 버킷을 계획 진도율 때문에
무리하게 쫓아가는가?" 벤치마크

시나리오
  EQP 2대, PPK 2종, 단일 OPER(OPER001).
    PPK_BIG   : WIP 20carrier, D0_PLAN_QTY=20  (재공=계획, 정상 규모)
    PPK_SMALL : WIP  2carrier, D0_PLAN_QTY=100 (재공 대비 계획이 크게 부풀려짐
                → "진도율"만 보면 항상 크게 뒤처진 것처럼 보인다)
  EQP001은 시작부터 PPK_BIG의 LOT_CD/TEMP로 이미 셋업돼 있어 PPK_BIG을
  계속 처리하면 전환 0회, PPK_SMALL로 옮기면 전환 1회다.
  EQP002는 초기 셋업이 없어(첫 배정은 무엇을 하든 무조건 무전환) PPK_SMALL의
  2carrier를 공짜로 처리할 수 있다 — 즉 PPK_SMALL을 처리하는 데 EQP001을
  건드릴 이유가 전혀 없다.

관찰 지표
  - EQP001이 PPK_BIG에서 PPK_SMALL로 전환되는 사건이 있는가
    (conversion_plans에서 from_lot_cd=BIG → to_lot_cd=SMALL 검출)
  - 있다면 그로 인한 생산/시간 손실은 얼마인가

실행
  python -m benchmark.small_wip_chase_bench
"""
from collections import defaultdict

from data.generator import (
    _abstract_row,
    _discrete_row,
    build_batch_info_from_discrete,
    build_lot_master_from_discrete,
    build_split_rules,
)
from data.loader.preprocess import preprocess
from env.scheduling_env import SchedulingEnv

OPER = "OPER001"
MODEL = "A"
ST = 60
CONV = 120
SIM = 720

PPK_BIG = "PPK001"     # LOT_CD=LC001
PPK_SMALL = "PPK002"   # LOT_CD=LC002
WIP_BIG = 20
WIP_SMALL = 2
PLAN_BIG = 20
PLAN_SMALL = 100        # 재공(2) 대비 크게 부풀린 계획 — "진도율" 착시 유발

EQP_DEDICATED = "EQP001"   # 시작부터 PPK_BIG 셋업 — 계속 쓰면 전환 0회
EQP_FREE = "EQP002"        # 초기 셋업 없음 — 첫 배정은 뭐든 전환 0회


def build_env() -> dict:
    discrete = []
    for i in range(WIP_BIG):
        home = EQP_DEDICATED if i % 2 == 0 else EQP_FREE
        discrete.append(_discrete_row(
            home, f"LOTBIG{i:03d}", PPK_BIG, OPER, ST, 1,
            eqp_model=MODEL, carrier_id=f"CARBIG{i:03d}", seq=i + 1,
        ))
    for i in range(WIP_SMALL):
        discrete.append(_discrete_row(
            EQP_FREE, f"LOTSML{i:03d}", PPK_SMALL, OPER, ST, 1,
            eqp_model=MODEL, carrier_id=f"CARSML{i:03d}", seq=i + 1,
        ))

    flow = [
        {"PLAN_PROD_ATTR_VAL": PPK_BIG, "OPER_SEQ": 1, "OPER_ID": OPER},
        {"PLAN_PROD_ATTR_VAL": PPK_SMALL, "OPER_SEQ": 1, "OPER_ID": OPER},
    ]
    plan = [
        {"PLAN_PROD_ATTR_VAL": PPK_BIG, "OPER_ID": OPER,
         "D0_PLAN_QTY": PLAN_BIG, "D1_PLAN_QTY": PLAN_BIG, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_ATTR_VAL": PPK_SMALL, "OPER_ID": OPER,
         "D0_PLAN_QTY": PLAN_SMALL, "D1_PLAN_QTY": PLAN_SMALL, "PLAN_PRIORITY": 1},
    ]
    abstract = [
        _abstract_row(PPK_BIG, OPER, MODEL, ST),
        _abstract_row(PPK_SMALL, OPER, MODEL, ST),
    ]
    lot_master = build_lot_master_from_discrete(discrete)
    batch_info = build_batch_info_from_discrete(discrete)
    big_lot_cd = next(r["LOT_CD"] for r in lot_master if r["LOT_ID"].startswith("LOTBIG"))
    big_temp = next(r["TEMP"] for r in lot_master if r["LOT_ID"].startswith("LOTBIG"))
    small_lot_cd = next(r["LOT_CD"] for r in lot_master if r["LOT_ID"].startswith("LOTSML"))
    assert big_lot_cd != small_lot_cd, "시나리오 전제(서로 다른 LOT_CD) 깨짐"

    eqp_initial_state = [{
        "EQP_ID": EQP_DEDICATED, "LOT_CD": big_lot_cd, "TEMP": big_temp,
        "PLAN_PROD_ATTR_VAL": PPK_BIG, "OPER_ID": OPER,
    }]
    tool_capacity = [
        {"LOT_CD": lc, "EQP_MODEL_CD": MODEL, "MAX_TOOL": 99}
        for lc in (big_lot_cd, small_lot_cd)
    ]

    raw = {
        "discrete_arrange": discrete, "abstract_arrange": abstract,
        "plan": plan, "flow": flow, "split": build_split_rules(flow),
        "lot_master": lot_master, "batch_info": batch_info,
        "tool_capacity": tool_capacity, "eqp_initial_state": eqp_initial_state,
    }
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = SIM
    ed["conversion_minutes"] = CONV
    ed["_big_lot_cd"] = big_lot_cd
    ed["_small_lot_cd"] = small_lot_cd
    return ed


def _heuristic_agent(algorithm: str, ed: dict):
    if algorithm == "minprogress":
        from agent.minprogress_agent import MinProgressAgent
        return MinProgressAgent(ed)
    if algorithm == "earliest_st":
        from agent.earliest_st_agent import EarliestSTAgent
        return EarliestSTAgent()
    if algorithm == "dedication":
        from agent.dedication_agent import DedicationAgent
        return DedicationAgent(ed)
    raise ValueError(algorithm)


def run_heuristic(ed: dict, algorithm: str) -> dict:
    """inference.runner.run_inference_with_agent()와 동일한 루프를 직접 돌린다
    (torch/sb3 의존인 agent.rl_agent를 끌어들이는 inference.runner import를
    피하기 위해 — 이 벤치마크는 휴리스틱 3종만 비교하면 충분하다)."""
    run_data = dict(ed)
    if algorithm == "earliest_st":
        run_data["eqp_selection"] = "min_st"
    agent = _heuristic_agent(algorithm, run_data)

    env = SchedulingEnv(run_data, record_history=False, max_episode_steps=20_000, truncate_on_time=False)
    env.reset()
    done = False
    while not done:
        env._ensure_decision_eqp()
        action = agent.predict(env.sim)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return {
        "schedule": env.get_schedule(),
        "conversion_plans": list(env.sim.conversion_plans),
        "stats": dict(env.sim.stats),
    }


def evaluate(ed: dict, algorithm: str) -> None:
    result = run_heuristic(ed, algorithm)
    sched = [r for r in result["schedule"] if r.get("END_TM", 0) <= SIM]
    convs = result.get("conversion_plans", [])
    stats = result["stats"]

    by_eqp_ppk = defaultdict(lambda: defaultdict(int))
    for r in sched:
        by_eqp_ppk[r["EQP_ID"]][r.get("PLAN_PROD_ATTR_VAL", "?")] += 1

    # EQP_DEDICATED이 BIG에서 다른 LOT_CD로 전환된 사건 = "재공 소량 버킷을 쫓아간" 신호
    bad_switches = [
        c for c in convs
        if c["eqp_id"] == EQP_DEDICATED and c.get("from_lot_cd") == ed["_big_lot_cd"]
        and c.get("to_lot_cd") != ed["_big_lot_cd"]
    ]

    print(f"\n--- {algorithm} ---")
    print(f"  총생산={len(sched)}/{WIP_BIG + WIP_SMALL}  전환={stats.get('conversions', 0)}회")
    for eqp_id in sorted(by_eqp_ppk):
        dist = dict(sorted(by_eqp_ppk[eqp_id].items()))
        print(f"    {eqp_id}: {dist}")
    if bad_switches:
        for c in bad_switches:
            print(
                f"  [경고] {EQP_DEDICATED}이 t={c['conv_start_min']}에 PPK_BIG "
                f"셋업을 버리고 PPK_SMALL 계열로 전환함 (재공 소량 버킷을 "
                f"쫓아간 것으로 보임) — 전환 {CONV}분 소요"
            )
    else:
        print(f"  [OK] {EQP_DEDICATED}은 PPK_BIG 셋업을 계속 유지 (소량 버킷을 쫓아가지 않음)")


def main() -> None:
    ed = build_env()
    print("=" * 64)
    print("SMALL_WIP_CHASE_BENCH: 계획 진도율 때문에 소량 재공을 무리하게")
    print("쫓아가는지 검증")
    print(f"  PPK_BIG:   WIP={WIP_BIG} PLAN={PLAN_BIG} (정상)")
    print(f"  PPK_SMALL: WIP={WIP_SMALL} PLAN={PLAN_SMALL} (재공 대비 계획 부풀림)")
    print(f"  ST={ST}분 CONV={CONV}분 SIM={SIM}분")
    print(f"  {EQP_DEDICATED}=PPK_BIG 기 셋업(계속 쓰면 전환 0) / "
          f"{EQP_FREE}=무셋업(뭘 해도 첫 배정은 전환 0)")
    print("=" * 64)

    for algo in ("minprogress", "earliest_st", "dedication"):
        evaluate(ed, algo)

    try:
        from agent.rl_agent import SchedulingAgent
        from inference.runner import run_inference

        agent = SchedulingAgent.load(env_data=ed)
        result = run_inference(ed, algorithm="scheduling_rl", agent=agent)
        sched = [r for r in result["schedule"] if r.get("END_TM", 0) <= SIM]
        print(f"\n--- scheduling_rl ---")
        print(f"  총생산={len(sched)}/{WIP_BIG + WIP_SMALL}  "
              f"전환={result['stats'].get('conversions', 0)}회")
    except Exception as e:
        print(f"\n[안내] scheduling_rl 생략 (모델/의존성 없음: {e})")


if __name__ == "__main__":
    main()
