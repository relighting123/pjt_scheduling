"""
롤링 재추론(Rolling Re-inference) 시뮬레이션
===========================================
`compare_vs_dedication.py`/`consistency_bench.py`는 둘 다 **정적** 시나리오
하나를 한 번에 끝까지(sim_end까지) 돌리는 단일 추론이다. 실제 운영은 그게
아니라 5분(등) 주기로 **완전히 새 env_data**(MES 재조회 결과)를 만들어 매번
새로 추론한다 — 앞선 회차가 아직 안 끝난 재공/설비 상태를 이어받되, 정책은
그 순간부터 다시 결정한다.

이 차이가 실제로 전환 횟수에 영향을 주는지 오프라인에서 재현해 측정한다:
매 `interval_minutes`(기본 5분)마다 살아있는 시뮬레이터의 잔여 WIP·설비
상태를 `SchedulingSimulator.export_live_snapshot()`으로 뽑아 **새 env_data**를
만들고, 그걸로 **새 시뮬레이터/새 env**를 구성해 다시 추론한다(정책 워밍업
없이 매번 관측을 새로 계산 → 결정도 새로). 진행 중이던 가공/전환은
`_busy_snapshot`으로 그대로 이어받아 중간에 끊기지 않는다.

비교 대상(`static`): 같은 시나리오를 한 번의 연속 시뮬레이션으로(정책이
한 번도 재시작되지 않고) 끝까지 돌린 결과 — "만약 정말 한 번만 계획을 세우고
다시는 안 건드렸다면"의 기준선.

실행
  python benchmark/rolling_reinference.py --suite bench --interval-minutes 5
  python benchmark/rolling_reinference.py --model PATH --limit 3 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from env.scheduling_env import SchedulingEnv  # noqa: E402
from env.scheduling_rl_env import SchedulingRLEnv  # noqa: E402
from benchmark.compare_vs_dedication import load_meta, load_ed  # noqa: E402

ROOT = Path(__file__).parent.parent

# ── env_data → 추론 실행 조건 정렬 (inference/runner.py의 current_wip_only 로직과 동일) ──


def _align_run_data(ed: dict, enable_wip_inflow: Optional[bool]) -> dict:
    data = dict(ed)
    if enable_wip_inflow is None:
        enable_wip_inflow = bool(data.get("enable_wip_inflow", False))
    data["enable_wip_inflow"] = enable_wip_inflow
    if not enable_wip_inflow:
        data["termination_mode"] = "current_wip_assigned"
    return data


# ── 단일 라운드 구동 ─────────────────────────────────────────────────────────


def _make_rl_predict(agent):
    def _predict(env: SchedulingRLEnv, obs: np.ndarray):
        mask = env.action_masks()
        return env, agent.predict(obs, deterministic=True, action_masks=mask), mask
    return _predict


def _drive_round(
    ed: dict,
    algorithm: str,
    agent,
    boundary_minutes: Optional[int],
    max_steps: int,
) -> dict:
    """env_data 하나로 새 env/시뮬레이터를 만들어 `boundary_minutes`(None이면
    끝까지)까지 진행. 반환: {"schedule", "conversions", "current_time",
    "done", "sim"}.  sim은 다음 라운드 스냅샷 추출용으로 그대로 반환한다."""
    from agent.minprogress_agent import MinProgressAgent
    from agent.earliest_st_agent import EarliestSTAgent
    from agent.dedication_agent import DedicationAgent

    if algorithm == "scheduling_rl":
        env = SchedulingRLEnv(ed, record_history=False, record_event_log=False,
                               truncate_on_time=False)
        obs, _ = env.reset()
        sim = env.sim
        steps = 0
        while True:
            if boundary_minutes is not None and sim.current_time >= boundary_minutes:
                break
            mask = env.action_masks()
            action = agent.predict(obs, deterministic=True, action_masks=mask, env_data=ed)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if terminated or truncated:
                return {"schedule": env.get_schedule(), "sim": sim, "done": True,
                        "current_time": sim.current_time}
            if steps >= max_steps:
                return {"schedule": env.get_schedule(), "sim": sim, "done": True,
                        "current_time": sim.current_time}
        return {"schedule": env.get_schedule(), "sim": sim, "done": False,
                "current_time": sim.current_time}

    if algorithm == "minprogress":
        heuristic_agent = MinProgressAgent(ed)
    elif algorithm == "earliest_st":
        heuristic_agent = EarliestSTAgent()
    elif algorithm == "dedication":
        heuristic_agent = DedicationAgent(ed)
    else:
        raise ValueError(f"알 수 없는 algorithm: {algorithm}")

    env = SchedulingEnv(ed, record_history=False, max_episode_steps=max_steps,
                        truncate_on_time=False)
    env.reset()
    sim = env.sim
    steps = 0
    while True:
        if boundary_minutes is not None and sim.current_time >= boundary_minutes:
            break
        env._ensure_decision_eqp()
        action = heuristic_agent.predict(sim)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated or truncated:
            return {"schedule": env.get_schedule(), "sim": sim, "done": True,
                    "current_time": sim.current_time}
        if steps >= max_steps:
            return {"schedule": env.get_schedule(), "sim": sim, "done": True,
                    "current_time": sim.current_time}
    return {"schedule": env.get_schedule(), "sim": sim, "done": False,
            "current_time": sim.current_time}


# ── 다음 라운드 env_data 구성 ────────────────────────────────────────────────


def _next_round_ed(ed: dict, sim, elapsed: int) -> dict:
    """방금 라운드가 진행한 만큼(elapsed분) 시간을 당긴 다음 라운드 env_data.

    sim.export_live_snapshot()의 동적 필드로 덮어쓰고, horizon/시작시각은
    consistency_bench.perturb_horizon()과 같은 방식으로 당긴다.
    """
    nxt = dict(ed)
    nxt.update(sim.export_live_snapshot())
    nxt["sim_end_minutes"] = int(ed["sim_end_minutes"]) - elapsed
    if "soft_cutoff_minutes" in ed:
        nxt["soft_cutoff_minutes"] = int(ed["soft_cutoff_minutes"]) - elapsed
    base_time = ed.get("sim_base_time")
    if base_time is not None and hasattr(base_time, "__add__"):
        from datetime import timedelta
        nxt["sim_base_time"] = base_time + timedelta(minutes=elapsed)
    return nxt


# ── 메인 러너 ────────────────────────────────────────────────────────────────


def run_dynamic_rolling(
    ed: dict,
    *,
    algorithm: str = "scheduling_rl",
    agent=None,
    interval_minutes: int = 5,
    enable_wip_inflow: Optional[bool] = None,
    max_rounds: int = 500,
    max_steps_per_round: int = 20_000,
) -> dict:
    """`interval_minutes`마다 완전히 새 env_data로 재추론하는 롤링 시뮬레이션.

    Returns: {"rounds": [...], "total_conversions": int, "total_produced": int,
              "elapsed_minutes": int, "n_rounds": int}
    """
    cur_ed = _align_run_data(ed, enable_wip_inflow)
    rounds = []
    total_elapsed = 0
    cumulative_conv_before = 0

    for round_idx in range(max_rounds):
        result = _drive_round(
            cur_ed, algorithm, agent, interval_minutes, max_steps_per_round,
        )
        sim = result["sim"]
        conv_now = sim.stats.get("conversions", 0)
        # 라운드마다 새 env/시뮬레이터라 schedule은 그 라운드 안에서 새로
        # 생성된 행만 담는다(이전 라운드에서 넘어온 진행 중 작업은
        # _busy_snapshot으로 이어받되 schedule에 다시 추가되지 않음) —
        # 그래서 라운드 간 누적 차감 없이 그대로 이번 라운드의 생산량이다.
        rounds.append({
            "round": round_idx,
            "start_offset": total_elapsed,
            "duration": result["current_time"],
            "conversions_delta": conv_now - cumulative_conv_before,
            "conversions_cumulative": conv_now,
            "produced_delta": len(result["schedule"]),
        })
        cumulative_conv_before = conv_now
        total_elapsed += result["current_time"]

        if result["done"]:
            break
        cur_ed = _next_round_ed(cur_ed, sim, result["current_time"])
    else:
        rounds[-1]["hit_max_rounds"] = True

    last_sim = result["sim"]
    return {
        "rounds": rounds,
        "n_rounds": len(rounds),
        "elapsed_minutes": total_elapsed,
        "total_conversions": last_sim.stats.get("conversions", 0),
        "total_produced": sum(r["produced_delta"] for r in rounds),
        "final_schedule_rows": len(result["schedule"]),
    }


def run_static_baseline(
    ed: dict,
    *,
    algorithm: str = "scheduling_rl",
    agent=None,
    enable_wip_inflow: Optional[bool] = None,
    max_steps: int = 20_000,
) -> dict:
    """롤링과 같은 env_data를 처음부터 끝까지 한 번만 추론(재시작 없음)."""
    cur_ed = _align_run_data(ed, enable_wip_inflow)
    result = _drive_round(cur_ed, algorithm, agent, None, max_steps)
    sim = result["sim"]
    return {
        "total_conversions": sim.stats.get("conversions", 0),
        "total_produced": len(result["schedule"]),
        "elapsed_minutes": result["current_time"],
    }


def run_comparison(
    ed: dict,
    *,
    algorithm: str = "scheduling_rl",
    agent=None,
    interval_minutes: int = 5,
    enable_wip_inflow: Optional[bool] = None,
) -> dict:
    static = run_static_baseline(
        ed, algorithm=algorithm, agent=agent, enable_wip_inflow=enable_wip_inflow,
    )
    dynamic = run_dynamic_rolling(
        ed, algorithm=algorithm, agent=agent, interval_minutes=interval_minutes,
        enable_wip_inflow=enable_wip_inflow,
    )
    return {
        "static": static,
        "dynamic": dynamic,
        "conversions_delta": dynamic["total_conversions"] - static["total_conversions"],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="bench", choices=["bench", "holdout", "train_pool"])
    ap.add_argument("--algorithm", default="scheduling_rl")
    ap.add_argument("--model", default=None, help="RL 모델 zip 경로")
    ap.add_argument("--interval-minutes", type=int, default=5,
                    help="재추론 주기(분). 기본 5 — 실제 운영 주기와 동일")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개 데이터셋만")
    ap.add_argument("--enable-wip-inflow", dest="enable_wip_inflow",
                    action="store_true", default=None)
    ap.add_argument("--no-wip-inflow", dest="enable_wip_inflow", action="store_false")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    meta = load_meta(args.suite)
    if args.limit:
        meta = meta[: args.limit]

    agent = None
    if args.algorithm == "scheduling_rl":
        from agent.rl_agent import SchedulingAgent
        eds0 = load_ed(meta[0])
        agent = SchedulingAgent.load(args.model, env_data=eds0)

    header = f"{'dataset':<24}{'static conv':>13}{'dynamic conv':>14}{'delta':>8}{'rounds':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for m in meta:
        ed = load_ed(m)
        cmp = run_comparison(
            ed, algorithm=args.algorithm, agent=agent,
            interval_minutes=args.interval_minutes,
            enable_wip_inflow=args.enable_wip_inflow,
        )
        rows.append({"id": m["id"], **cmp})
        print(f"{m['id']:<24}{cmp['static']['total_conversions']:>13}"
              f"{cmp['dynamic']['total_conversions']:>14}"
              f"{cmp['conversions_delta']:>+8}{cmp['dynamic']['n_rounds']:>8}")

    total_static = sum(r["static"]["total_conversions"] for r in rows)
    total_dynamic = sum(r["dynamic"]["total_conversions"] for r in rows)
    print("-" * len(header))
    print(f"{'합계':<24}{total_static:>13}{total_dynamic:>14}{total_dynamic - total_static:>+8}")

    out = {
        "suite": args.suite, "algorithm": args.algorithm,
        "interval_minutes": args.interval_minutes, "datasets": rows,
        "summary": {
            "total_static_conversions": total_static,
            "total_dynamic_conversions": total_dynamic,
            "delta": total_dynamic - total_static,
        },
    }
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n저장 → {path}")


if __name__ == "__main__":
    main()
