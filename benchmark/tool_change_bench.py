"""
TOOL_CHANGE_BENCH 평가 — 10개 데이터셋 × 알고리즘 vs 정답지(최적해) 비교
=========================================================================
실행
  python benchmark/tool_change_bench.py            # earliest_st, minprogress 비교
  python benchmark/tool_change_bench.py --verify    # + Category C 오라클 정책으로 정답지 실제 검증

산출
  data/dataset/tool_change_bench_results.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from inference.runner import run_inference
from env.scheduling_env import SchedulingEnv

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
META = json.load(open(SUITE_ROOT / "tool_change_bench_meta.json", encoding="utf-8"))
TIMEKEY = "20260703000000"


def load_ed(m: dict) -> dict:
    inp = SUITE_ROOT / m["id"] / "train" / TIMEKEY / "input"
    ed = preprocess(load_data(inp))
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = m["sim_end_minutes"]
    ed["conversion_minutes"] = m["conversion_minutes"]
    return ed


def _inflow_enabled(m: dict) -> bool:
    """Category C(재공편중·안전재공)만 하류 공정이 상류 완료로부터 유입되는
    구조라 wip inflow가 필요하다. A/B는 공정마다 재공을 이미 충분히 미리
    실어뒀으므로(각 공정이 서로 독립) inflow를 켜면 상류 완료분이 하류에
    '추가로' 더 공급되어 재공 총량이 부풀려진다 — 반드시 꺼야 한다."""
    return m["category"].startswith("C.")


def _productive_conversions(conversion_plans: list, sim_end: int) -> int:
    """conv_end_min(전환 완료 시각)이 horizon을 넘기면 생산에 전혀 기여 못 하는
    '경계부 낭비 전환'이므로 제외한다 — production 0 기여 전환을 셈에서 뺀
    지표라야 알고리즘 간 공정 비교가 된다(모든 정책이 동일 규칙 적용받음)."""
    return sum(1 for c in conversion_plans if c.get("conv_end_min", 0) <= sim_end)


def kpi(result: dict, m: dict) -> dict:
    sched = [r for r in result["schedule"] if r.get("END_TM", 0) <= m["sim_end_minutes"]]
    prod = len(sched)  # WF_QTY=1/carrier이므로 행 수 = carrier 수
    conv = _productive_conversions(result.get("conversion_plans", []), m["sim_end_minutes"])
    opt = m["optimal"]
    return {
        "prod": prod, "conv": conv,
        "prod_opt": opt["production"], "conv_opt": opt["conversions"],
        "prod_gap": opt["production"] - prod,
        "conv_gap": conv - opt["conversions"],
        "prod_pct": round(100 * prod / max(opt["production"], 1), 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# Category B 오라클 정책 — 장비별 '고정 제품 블록'을 순서대로 끝까지 처리.
# (구성적 최적해가 실제로 달성 가능한지 검증)
# ═══════════════════════════════════════════════════════════════════════

class BlockDedicationOracle:
    """eqp_blocks: {eqp_id: [ppk, ppk, ...]} — 각 장비가 순서대로 제품을
    '완전히' 처리한 뒤 다음 제품으로 넘어간다(전환 = len(blocks)-1 회/장비)."""

    def __init__(self, eqp_blocks: dict):
        self.eqp_blocks = eqp_blocks
        self._ptr = {eqp: 0 for eqp in eqp_blocks}

    def predict(self, sim):
        eqp_id = sim.current_idle_eqp()
        if eqp_id is None or eqp_id not in self.eqp_blocks:
            return np.array([0], dtype=np.int64)
        feasible = sim.get_feasible_ppk_oper(eqp_id)
        if not feasible:
            return np.array([0], dtype=np.int64)
        blocks = self.eqp_blocks[eqp_id]
        feasible_ppks = {sim.ppk_oper_from_flat(f)[0]: f for f in feasible}
        while self._ptr[eqp_id] < len(blocks):
            target = blocks[self._ptr[eqp_id]]
            if target in feasible_ppks:
                return np.array([feasible_ppks[target]], dtype=np.int64)
            self._ptr[eqp_id] += 1
        return np.array([0], dtype=np.int64)


def run_block_oracle(m: dict, ed: dict, eqp_blocks: dict, max_conversions_per_eqp: int = None) -> dict:
    """max_conversions_per_eqp: 지정 시(TCB04/06처럼 장비당 전환이 전혀 필요 없는
    구성) 프레임워크의 '유휴 시 무조건 뭔가 배정' 기본 동작이 불필요한 보조 전환을
    유발하지 못하도록 강제 차단 — 정답지의 0전환 주장이 실제로 달성 가능함을 증명."""
    agent = BlockDedicationOracle(eqp_blocks)
    ed = dict(ed, enable_wip_inflow=_inflow_enabled(m))
    if max_conversions_per_eqp is not None:
        ed["max_conversions_per_eqp"] = max_conversions_per_eqp
    env = SchedulingEnv(ed, record_history=False)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 20_000:
        # env.step()이 내부적으로 idle EQP를 '지연 해석'하기 때문에, predict() 호출
        # 시점에 current_idle_eqp()가 아직 None일 수 있다 — 먼저 해석을 강제해야
        # 액션이 의도한 EQP에 정확히 적용된다(안 하면 이전 라운드의 stale 액션이
        # 엉뚱한 EQP에 적용되는 버그가 생긴다).
        env._ensure_decision_eqp()
        action = agent.predict(env.sim)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
    sched = [r for r in env.get_schedule() if r.get("END_TM", 0) <= m["sim_end_minutes"]]
    conv = _productive_conversions(env.sim.conversion_plans, m["sim_end_minutes"])
    return {
        "prod": len(sched), "conv": conv,
        "makespan": max((r["END_TM"] for r in env.get_schedule()), default=0),
    }


BLOCK_ORACLE_SPECS = {
    "TCB04_MULTI_SYM_2STAGE": {
        f"EQP{i + 1:03d}": [f"PPK{(i % 3) + 1:03d}"] for i in range(6)
    },
    "TCB05_MULTI_OVER_2STAGE": {
        **{f"EQP{i + 1:03d}": b for i, b in enumerate([["PPK001", "PPK002"], ["PPK003", "PPK004"], ["PPK005"]])},
        **{f"EQP{i + 4:03d}": b for i, b in enumerate([["PPK001", "PPK002"], ["PPK003", "PPK004"], ["PPK005"]])},
    },
    "TCB06_MULTI_LOADSKEW_2STAGE": {
        f"EQP{i + 1:03d}": [f"PPK{(i % 4) + 1:03d}"] for i in range(8)
    },
    "TCB07_MULTI_CONVHEAVY_2STAGE": {
        **{f"EQP{i + 1:03d}": b for i, b in enumerate([["PPK001", "PPK002"], ["PPK003", "PPK004"], ["PPK005"]])},
        **{f"EQP{i + 4:03d}": b for i, b in enumerate([["PPK001", "PPK002"], ["PPK003", "PPK004"], ["PPK005"]])},
    },
}

# 장비당 허용 전환 상한(오라클 검증 전용) — 정답지가 "장비당 필요한 전환 수"를
# 정확히 명시하는 데이터셋에서, 프레임워크의 '유휴 시 무조건 뭔가 배정' 기본
# 동작이 그 이상으로 불필요한 보조 전환을 만들지 못하도록 강제한다. TCB05/07은
# 이미 제품별 전용 EQP 그룹(모델)으로 막혀 있어 필요 없다.
BLOCK_ORACLE_MAX_CONV_PER_EQP = {
    "TCB04_MULTI_SYM_2STAGE": 0,
    "TCB06_MULTI_LOADSKEW_2STAGE": 0,
}


# ═══════════════════════════════════════════════════════════════════════
# Category C 오라클 정책 — 정답지(안전재공 확보 후 1회 전환)를 실제로 재현해
# 시뮬레이터가 손계산과 동일한 makespan/전환/생산을 내는지 검증한다.
# ═══════════════════════════════════════════════════════════════════════

class SafetySwitchOracle:
    """겸용 장비 n대: switch_time 이전엔 전원 상류(up) 공정, 그 이후엔
    switch_count대만 하류(down)로 영구 전환 후 유지, 나머지는 상류 유지."""

    def __init__(self, up_oper: str, down_oper: str, switch_time: float, switch_eqp_ids: set):
        self.up_oper = up_oper
        self.down_oper = down_oper
        self.switch_time = switch_time
        self.switch_eqp_ids = switch_eqp_ids

    def predict(self, sim):
        eqp_id = sim.current_idle_eqp()
        if eqp_id is None:
            return np.array([0], dtype=np.int64)
        feasible = sim.get_feasible_ppk_oper(eqp_id)
        if not feasible:
            return np.array([0], dtype=np.int64)
        is_switch_eqp = eqp_id in self.switch_eqp_ids
        want_down = is_switch_eqp and sim.current_time >= self.switch_time
        target_oper = self.down_oper if want_down else self.up_oper
        for flat in feasible:
            ppk, oper_id = sim.ppk_oper_from_flat(flat)
            if oper_id == target_oper:
                return np.array([flat], dtype=np.int64)
        if not is_switch_eqp:
            # 상류 전담 장비: down 재공이 실수로 보이더라도 절대 건드리지 않는다
            # (목표(up) 재공이 소진되면 그냥 idle 유지 — 추가 전환 유발 금지).
            return np.array([0], dtype=np.int64)
        # 전환 대상 장비인데 아직 down 재공이 안 쌓였다면(설계상 발생하지 않아야
        # 하지만 방어적으로) up을 임시로 계속 — 단, 이후 다시 down 전환 시 재전환
        # 비용이 생길 수 있음을 인지하고 최후 수단으로만 사용.
        for flat in feasible:
            ppk, oper_id = sim.ppk_oper_from_flat(flat)
            if oper_id == self.up_oper:
                return np.array([flat], dtype=np.int64)
        return np.array([0], dtype=np.int64)


ORACLE_SPECS = {
    "TCB08_SAFETY_1EQP": dict(switch_time=400, switch_eqp_ids={"EQP001"}),
    "TCB09_SAFETY_2EQP_BUFFER": dict(switch_time=120, switch_eqp_ids={"EQP002"}),
    "TCB10_SAFETY_3EQP_SKEWED": dict(switch_time=80, switch_eqp_ids={"EQP003"}),
}


def run_oracle(m: dict, ed: dict) -> dict:
    spec = ORACLE_SPECS[m["id"]]
    up_oper = ed["flow"][ed["prod_keys"][0]][0]["oper_id"]
    down_oper = ed["flow"][ed["prod_keys"][0]][1]["oper_id"]
    agent = SafetySwitchOracle(up_oper, down_oper, spec["switch_time"], spec["switch_eqp_ids"])

    env = SchedulingEnv(dict(ed, enable_wip_inflow=_inflow_enabled(m)), record_history=False)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 20_000:
        # env.step()이 내부적으로 idle EQP를 '지연 해석'하기 때문에, predict() 호출
        # 시점에 current_idle_eqp()가 아직 None일 수 있다 — 먼저 해석을 강제해야
        # 액션이 의도한 EQP에 정확히 적용된다(안 하면 이전 라운드의 stale 액션이
        # 엉뚱한 EQP에 적용되는 버그가 생긴다).
        env._ensure_decision_eqp()
        action = agent.predict(env.sim)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
    sched = [r for r in env.get_schedule() if r.get("END_TM", 0) <= m["sim_end_minutes"]]
    conv = _productive_conversions(env.sim.conversion_plans, m["sim_end_minutes"])
    return {
        "prod": len(sched), "conv": conv,
        "makespan": max((r["END_TM"] for r in env.get_schedule()), default=0),
    }


# ═══════════════════════════════════════════════════════════════════════

def main():
    verify = "--verify" in sys.argv
    eds = [load_ed(m) for m in META]

    results = []
    print(f"{'dataset':<32}{'algo':<14}{'prod':>12}{'conv':>10}{'gap':>8}")
    for m, ed in zip(META, eds):
        row = {"id": m["id"], "category": m["category"], "sim_end_minutes": m["sim_end_minutes"],
               "optimal": m["optimal"], "algos": {}}
        for algo, label in (("earliest_st", "Earliest-ST"), ("minprogress", "Min-Progress")):
            res = run_inference(ed, algorithm=algo, record_history=False,
                                 enable_wip_inflow=_inflow_enabled(m))
            k = kpi(res, m)
            row["algos"][algo] = k
            gap = f"-{k['prod_gap']}" if k["prod_gap"] else "0"
            print(f"{m['id']:<32}{label:<14}{k['prod']:>4}/{k['prod_opt']:<7}"
                  f"{k['conv']:>4}/{k['conv_opt']:<5}{gap:>8}")
        if verify and m["id"] in ORACLE_SPECS:
            oc = run_oracle(m, ed)
            row["oracle_verification"] = oc
            match = (oc["prod"] == m["optimal"]["production"] and oc["conv"] == m["optimal"]["conversions"])
            print(f"    ↳ oracle(정답지 재현): prod={oc['prod']} conv={oc['conv']} "
                  f"makespan={oc['makespan']}  {'✓ 일치' if match else '✗ 불일치'}")
        if verify and m["id"] in BLOCK_ORACLE_SPECS:
            oc = run_block_oracle(m, ed, BLOCK_ORACLE_SPECS[m["id"]],
                                   max_conversions_per_eqp=BLOCK_ORACLE_MAX_CONV_PER_EQP.get(m["id"]))
            row["oracle_verification"] = oc
            match = (oc["prod"] == m["optimal"]["production"] and oc["conv"] == m["optimal"]["conversions"])
            print(f"    ↳ oracle(정답지 재현): prod={oc['prod']} conv={oc['conv']} "
                  f"makespan={oc['makespan']}  {'✓ 일치' if match else '✗ 불일치'}")
        results.append(row)
        print("")

    def agg(algo):
        rows = [r["algos"][algo] for r in results]
        tot_prod = sum(r["prod"] for r in rows)
        tot_opt = sum(r["prod_opt"] for r in rows)
        tot_conv = sum(r["conv"] for r in rows)
        tot_conv_opt = sum(r["conv_opt"] for r in rows)
        return {
            "prod": tot_prod, "prod_opt": tot_opt,
            "prod_pct": round(100 * tot_prod / max(tot_opt, 1), 1),
            "conv": tot_conv, "conv_opt": tot_conv_opt,
            "n_optimal": sum(1 for r in rows if r["prod"] == r["prod_opt"] and r["conv"] == r["conv_opt"]),
            "n_sets": len(rows),
        }

    summary = {a: agg(a) for a in ["earliest_st", "minprogress"]}
    out = {"datasets": results, "summary": summary}
    with open(SUITE_ROOT / "tool_change_bench_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"{'집계':<14}{'생산률(정답지 대비)':>22}{'총전환':>10}{'정답지 도달':>12}")
    for a, lbl in [("earliest_st", "Earliest-ST"), ("minprogress", "Min-Progress")]:
        s = summary[a]
        print(f"{lbl:<14}{s['prod_pct']:>20.1f}%{s['conv']:>10}(정답 {s['conv_opt']}){s['n_optimal']:>6}/{s['n_sets']}")
    print(f"\n저장 → {SUITE_ROOT / 'tool_change_bench_results.json'}")


if __name__ == "__main__":
    main()
