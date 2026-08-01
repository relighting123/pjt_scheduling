"""휴리스틱 추론 속도/정합성 회귀 테스트.

배경(실측):
1. 제품(PPK) 수가 config.env.max_prod_count(10)를 넘는 데이터에서 flat 인덱스
   (oi*P+pi)가 충돌했다 — flat 10 이상이 decode(flat%P)에서 다른 버킷으로
   오역돼 초과분 제품의 재공이 영영 배정되지 않았고, 시뮬이 끝나지 않아
   max_steps(1만)까지 공회전하며 dedication 실행이 수 초~수십 초 걸렸다.
   → flat 축을 config 값과 데이터 실제 종류 수 중 큰 쪽으로 잡아 충돌 제거.
2. 휴리스틱은 step()의 obs를 읽지 않는데 매 스텝 obs_dim 벡터를 만들고
   있었다(실측 전체의 ~15%) → compute_obs=False로 건너뛴다.
3. available_lots가 스텝마다 abstract×WIP 전체를 다시 훑던 것을
   (state_version, current_time) 키로 캐시한다.
"""
import numpy as np
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_env import SchedulingEnv
from agent.dedication_agent import DedicationAgent
from inference.runner import run_inference, run_inference_with_agent

MAX_P = CONFIG.env.max_prod_count


def _raw(n_eqp: int, n_ppk: int, carriers: int, st: int) -> dict:
    """n_eqp × n_ppk 대칭 데이터 (홈 설비 전담, mix=0)."""
    ppks = [f"PPK{i + 1:03d}" for i in range(n_ppk)]
    eqps = [f"EQP{i + 1:03d}" for i in range(n_eqp)]
    discrete, lot_master, plan, flow, abstract = [], [], [], [], []
    split, batch, tool = [], [], []
    num = 0
    for pi, ppk in enumerate(ppks):
        lc = f"LC_{pi:02d}"
        home = eqps[pi % n_eqp]
        for _ in range(carriers):
            num += 1
            lid, car = f"LOT{num:03d}", f"CAR{num:03d}"
            discrete.append({
                "EQP_ID": home, "LOT_ID": lid, "PLAN_PROD_ATTR_VAL": ppk,
                "OPER_ID": "OPER001", "ST": st, "EQP_MODEL_CD": "A",
                "WF_QTY": 1, "SEQ": 1, "CARRIER_ID": car, "LOT_STAT_CD": "WAIT",
            })
            lot_master.append({"LOT_ID": lid, "LOT_CD": lc, "TEMP": "T600"})
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                     "D0_PLAN_QTY": carriers, "D1_PLAN_QTY": carriers,
                     "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                         "EQP_MODEL_CD": "A", "ST": st})
        split.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                      "EQP_MODEL_CD": "A", "SPLIT_QTY": 1})
        batch.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                      "LOT_CD": lc, "TEMP": "T600"})
        tool.append({"LOT_CD": lc, "TEMP": "T600", "MAX_TOOL": 99})
    return {
        "discrete_arrange": discrete, "lot_master": lot_master, "plan": plan,
        "flow": flow, "abstract_arrange": abstract, "split": split,
        "batch_info": batch, "tool_capacity": tool,
    }


def _ed(raw: dict, sim_end: int = 4000, conv: int = 45) -> dict:
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = sim_end
    ed["conversion_minutes"] = conv
    return ed


# 제품 수가 config 축을 넘는 케이스 (MAX_P + 2 종)
N_EQP, N_PPK, CARRIERS = 4, MAX_P + 2, 6
OVERFLOW_ED = _ed(_raw(N_EQP, N_PPK, CARRIERS, st=45))


def test_flat_index_roundtrip_is_injective_on_axis_overflow():
    """축 초과 데이터에서도 flat 인덱스가 버킷마다 고유해야 한다 (충돌 제거)."""
    env = SchedulingEnv(OVERFLOW_ED, record_history=False, compute_obs=False)
    env.reset()
    sim = env.sim
    flats = {}
    for ppk in OVERFLOW_ED["prod_keys"]:
        flat = sim.ppk_oper_flat_index("OPER001", ppk)
        assert flat not in flats, f"flat {flat} 충돌: {flats.get(flat)} vs {ppk}"
        flats[flat] = ppk
        back_ppk, back_oper = sim.ppk_oper_from_flat(flat)
        assert (back_ppk, back_oper) == (ppk, "OPER001")


def test_flat_index_within_config_axes_is_unchanged():
    """축 안 데이터는 기존과 동일하게 config 고정 축(P=max_prod_count)을 쓴다."""
    ed = _ed(_raw(2, 3, 2, st=45))
    env = SchedulingEnv(ed, record_history=False, compute_obs=False)
    env.reset()
    sim = env.sim
    assert sim._flat_P == MAX_P
    flat = sim.ppk_oper_flat_index("OPER001", "PPK003")
    assert flat == 2                      # oi=0, pi=2 → 0*P + 2
    assert sim.ppk_oper_from_flat(flat) == ("PPK003", "OPER001")


@pytest.mark.parametrize("algorithm", ["dedication", "minprogress", "earliest_st"])
def test_heuristics_complete_on_axis_overflow(algorithm):
    """축 초과 데이터에서 휴리스틱이 max_steps 공회전 없이 정상 종료해야 한다."""
    res = run_inference(OVERFLOW_ED, algorithm=algorithm, record_history=False)
    stats = res["stats"]
    total = N_PPK * CARRIERS
    assert stats["terminated"] and not stats["truncated"]
    assert len(res["schedule"]) == total
    # 축 초과분 제품(PPK011+)의 재공도 전부 배정돼야 한다 (예전엔 0건이었다)
    produced = {r["PLAN_PROD_ATTR_VAL"] for r in res["schedule"]}
    assert produced == set(OVERFLOW_ED["prod_keys"])
    # 공회전 방지: steps는 배정 수의 수 배 이내여야 한다 (예전엔 max_steps=10,000)
    assert stats["steps"] < total * 10


def test_dedication_on_overflow_is_fast_and_optimal():
    """dedication은 전담 가능한 대칭 데이터에서 0전환 완료 + 스텝 수 최소."""
    ed = _ed(_raw(4, MAX_P + 2, 6, st=45))
    res = run_inference(ed, algorithm="dedication", record_history=False)
    stats = res["stats"]
    total = (MAX_P + 2) * 6
    assert len(res["schedule"]) == total
    assert stats["conversions"] == N_PPK - N_EQP  # 최소 전환(제품-설비) 달성
    assert stats["steps"] < total * 3


def _schedule_signature(res: dict) -> list:
    return [
        (r["EQP_ID"], r["LOT_ID"], r["START_TM"], r["END_TM"], r["PLAN_PROD_ATTR_VAL"])
        for r in res["schedule"]
    ]


def test_compute_obs_false_does_not_change_results():
    """obs 생성을 건너뛰어도(compute_obs=False) 스케줄·통계가 동일해야 한다."""
    ed = _ed(_raw(3, 4, 4, st=45), sim_end=2000)

    # obs 없는 공식 경로 (run_inference_with_agent → compute_obs=False)
    fast = run_inference(ed, algorithm="dedication", record_history=False)

    # obs 있는 수동 루프 (같은 에이전트·같은 env, compute_obs=True)
    env = SchedulingEnv(
        dict(ed, enable_wip_inflow=False,
             termination_mode="current_wip_assigned"),
        record_history=False, truncate_on_time=False, compute_obs=True,
    )
    env.reset()
    agent = DedicationAgent(env._env_data)
    for _ in range(10_000):
        env._ensure_decision_eqp()
        _, _, term, trunc, _ = env.step(agent.predict(env.sim))
        if term or trunc:
            break

    assert _schedule_signature(fast) == [
        (r["EQP_ID"], r["LOT_ID"], r["START_TM"], r["END_TM"], r["PLAN_PROD_ATTR_VAL"])
        for r in env.sim.schedule
    ]
    assert fast["stats"]["conversions"] == env.sim.stats.get("conversions", 0)


def test_available_lots_cache_hits_and_invalidates():
    """같은 (버전, 시각)에서는 캐시 hit, 배정으로 버전이 바뀌면 재계산돼야 한다."""
    ed = _ed(_raw(2, 2, 3, st=30), sim_end=1000, conv=30)
    env = SchedulingEnv(
        dict(ed, enable_wip_inflow=False, termination_mode="current_wip_assigned"),
        record_history=False, truncate_on_time=False, compute_obs=False,
    )
    env.reset()
    sim = env.sim
    env._ensure_decision_eqp()
    eqp = sim.current_idle_eqp()
    first = sim.available_lots(eqp)
    assert sim.available_lots(eqp) is first          # 캐시 hit (같은 객체)
    n_before = len(first)

    agent = DedicationAgent(env._env_data)
    _, _, _, _, _ = env.step(agent.predict(env.sim))  # 배정 → 버전 변경
    env._ensure_decision_eqp()
    second = sim.available_lots(eqp)
    assert second is not first                        # 무효화 → 재계산
    assert len(second) < n_before                     # 소비된 WIP 반영
