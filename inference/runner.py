"""
inference/runner.py – 추론 실행 및 결과 저장
학습된 에이전트 또는 휴리스틱으로 스케줄링을 실행하고 결과를 저장합니다.
"""
import json
from pathlib import Path
from typing import Optional

from config import CONFIG
from agent.rl_agent import SchedulingAgent
from agent.minprogress_agent import MinProgressAgent
from agent.earliest_st_agent import EarliestSTAgent
from agent.registry import validate_algorithm, VALID_ALGORITHMS
from env.scheduling_env import SchedulingEnv
from data.writer import write_inference_result
from utils.helpers import minutes_to_str


def _inference_max_steps(env_data: dict) -> int:
    """추론 안전장치용 step 상한. 시간 horizon이 아니라 작업 규모 기준."""
    initial_lots = max(
        len(env_data.get("abstract_lot_meta", {})),
        len(env_data.get("lots", [])),
        1,
    )
    flow_depth = max(
        (len(steps) for steps in env_data.get("flow", {}).values()),
        default=max(len(env_data.get("oper_ids", [])), 1),
    )
    eqp_count = max(len(env_data.get("eqp_ids", [])), 1)
    expected_assignments = initial_lots * max(flow_depth, 1)
    return max(10_000, expected_assignments * 4 + eqp_count * 100 + 500)


def run_inference(
    env_data: dict,
    algorithm: str = "scheduling_rl",
    agent: Optional[SchedulingAgent] = None,
    model_path: Optional[str] = None,
    deterministic: bool = True,
    record_history: bool = True,
    record_decision_log: bool = False,
    enable_wip_inflow: bool = False,
    current_wip_only: Optional[bool] = None,
    max_conversions: Optional[int] = None,
    max_conversions_per_eqp: Optional[int] = None,
    conversion_minutes: Optional[int] = None,
) -> dict:
    """
    목적: 선택한 알고리즘으로 Scheduling 추론 실행
    Input:
        env_data      (dict): preprocessor.preprocess() 반환값
        algorithm     (str):  "scheduling_rl" | "minprogress" | "earliest_st"
        agent         (SchedulingAgent|None): RL용 (None이면 model_path로 로드)
        model_path    (str|None): 모델 파일 경로
        deterministic (bool): RL 예측 시 greedy 여부
    Output:
        {
          "schedule", "history", "stats", "plan", "algorithm"
        }
    """
    algorithm = validate_algorithm(algorithm)

    if record_decision_log:
        import warnings
        warnings.warn(
            "record_decision_log=True는 추론 속도를 크게 저하시킵니다. "
            "디버깅 목적이 아니라면 False(기본값)를 사용하세요.",
            stacklevel=2,
        )

    if current_wip_only is None:
        current_wip_only = not enable_wip_inflow
    else:
        enable_wip_inflow = not current_wip_only

    run_data = dict(env_data)
    run_data["enable_wip_inflow"] = enable_wip_inflow
    if max_conversions is not None:
        run_data["max_conversions"] = max_conversions
    if max_conversions_per_eqp is not None:
        run_data["max_conversions_per_eqp"] = max_conversions_per_eqp
    if conversion_minutes is not None:
        run_data["conversion_minutes"] = conversion_minutes
    if current_wip_only:
        run_data["termination_mode"] = "current_wip_assigned"
    if algorithm == "earliest_st":
        run_data["eqp_selection"] = "min_st"

    if algorithm == "scheduling_rl":
        if agent is None:
            agent = SchedulingAgent.load(model_path, env_data=env_data)

    heuristic_agent = None
    if algorithm == "minprogress":
        heuristic_agent = MinProgressAgent(env_data)
    elif algorithm == "earliest_st":
        heuristic_agent = EarliestSTAgent()

    max_steps = _inference_max_steps(env_data)

    # SchedulingRLEnv는 MultiDiscrete action space — 휴리스틱과 별도 루프에서 직접 실행
    if algorithm == "scheduling_rl":
        return _run_scheduling_rl_inference(
            run_data, env_data, agent, max_steps, deterministic,
            record_history, record_decision_log, algorithm,
        )

    env = SchedulingEnv(
        run_data,
        record_history=record_history,
        record_decision_log=record_decision_log,
        max_episode_steps=max_steps,
        truncate_on_time=False,
    )
    sched_env: SchedulingEnv = env
    env.reset()
    done = False
    steps = 0
    terminated = False
    truncated = False

    while not done:
        action = heuristic_agent.predict(sched_env.sim)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        if steps >= max_steps:
            break

    schedule = sched_env.get_schedule()
    history = sched_env.get_history()
    stats = sched_env.sim.stats
    base_time = env_data["sim_base_time"]

    for rec in schedule:
        rec["START_TM_STR"] = minutes_to_str(rec["START_TM"], base_time)
        rec["END_TM_STR"] = minutes_to_str(rec["END_TM"], base_time)

    return {
        "schedule":         schedule,
        "history":          history,
        "event_log":        list(sched_env.sim.event_log),
        "decision_log":     sched_env.get_decision_log() if record_decision_log else [],
        "conversion_plans": list(sched_env.sim.conversion_plans),
        "stats":            {
            "idle_total":    stats["idle_total"],
            "oper_switches": stats["oper_switches"],
            "prod_switches": stats["prod_switches"],
            "conversions":   stats.get("conversions", 0),
            "completed_qty": {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): v
                              for k, v in stats["completed_qty"].items()},
            "remaining_wip":  sched_env.sim.get_wip_waiting(),
            "remaining_current_wip": sched_env.sim.get_remaining_current_wip(),
            "steps":          steps,
            "terminated":     terminated,
            "truncated":      truncated,
            "current_time":   sched_env.sim.current_time,
            "sim_end_minutes": sched_env.sim.sim_end,
            "termination_mode": sched_env.sim._termination_mode,
            "enable_wip_inflow": sched_env.sim._enable_wip_inflow,
        },
        "plan":      env_data["plan"],
        "sim_base_time": base_time.isoformat(sep=" ") if hasattr(base_time, "isoformat") else str(base_time),
        "algorithm": algorithm,
    }


def _run_scheduling_rl_inference(
    run_data: dict,
    env_data: dict,
    agent,
    max_steps: int,
    deterministic: bool,
    record_history: bool,
    record_decision_log: bool,
    algorithm: str,
) -> dict:
    """SchedulingRLEnv(MultiDiscrete) 전용 추론 루프."""
    from env.scheduling_rl_env import SchedulingRLEnv

    env = SchedulingRLEnv(
        run_data,
        record_history=record_history,
        record_event_log=record_history,
        truncate_on_time=False,
        record_decision_log=record_decision_log,
    )
    obs, _ = env.reset()
    done = False
    steps = 0
    terminated = False
    truncated = False

    while not done:
        mask = env.action_masks()
        action = agent.predict(
            obs,
            deterministic=deterministic,
            action_masks=mask,
            env_data=env_data,
        )
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        if steps >= max_steps:
            break

    sched_env = env
    schedule = sched_env.get_schedule()
    history = sched_env.get_history()
    stats = sched_env.sim.stats
    base_time = env_data["sim_base_time"]

    for rec in schedule:
        rec["START_TM_STR"] = minutes_to_str(rec["START_TM"], base_time)
        rec["END_TM_STR"] = minutes_to_str(rec["END_TM"], base_time)

    return {
        "schedule":         schedule,
        "history":          history,
        "event_log":        list(sched_env.sim.event_log),
        "decision_log":     sched_env.get_decision_log() if record_decision_log else [],
        "conversion_plans": list(sched_env.sim.conversion_plans),
        "stats": {
            "idle_total":    stats["idle_total"],
            "oper_switches": stats["oper_switches"],
            "prod_switches": stats["prod_switches"],
            "conversions":   stats.get("conversions", 0),
            "completed_qty": {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): v
                              for k, v in stats["completed_qty"].items()},
            "remaining_wip":  sched_env.sim.get_wip_waiting(),
            "remaining_current_wip": sched_env.sim.get_remaining_current_wip(),
            "steps":          steps,
            "terminated":     terminated,
            "truncated":      truncated,
            "current_time":   sched_env.sim.current_time,
            "sim_end_minutes": sched_env.sim.sim_end,
            "termination_mode": sched_env.sim._termination_mode,
            "enable_wip_inflow": sched_env.sim._enable_wip_inflow,
        },
        "plan":      env_data["plan"],
        "sim_base_time": base_time.isoformat(sep=" ") if hasattr(base_time, "isoformat") else str(base_time),
        "algorithm": algorithm,
    }


def save_result(
    result: dict,
    output_dir: Path = None,
    result_name: str = "result",
    env_data: Optional[dict] = None,
    *,
    write_sql: bool = True,
    write_kpi: bool = False,
) -> Path:
    """
    추론 결과 저장:
      - output.json  : RTS 적재 JSON (data.writer)
      - result_full.json : UI·디버그용 전체 결과

    write_kpi=True: KPI(RTS_PERFMON_HIS) 도 output.json/sql에 포함 (옵션).
    """
    d = output_dir or CONFIG.path.infer_output_dir
    d.mkdir(parents=True, exist_ok=True)

    writer_path = None
    if env_data is not None:
        writer_path = write_inference_result(
            result, env_data, output_dir=d, write_sql_files=write_sql, write_kpi=write_kpi,
        )

    full_path = d / f"{result_name}_full.json"
    from api.serializers import serialize_history

    serializable = {
        "schedule":         result["schedule"],
        "history":          serialize_history(result.get("history", [])),
        "event_log":        result.get("event_log", []),
        "decision_log":     result.get("decision_log", []),
        "conversion_plans": result.get("conversion_plans", []),
        "stats":            result["stats"],
        "plan":             result["plan"],
        "algorithm":        result.get("algorithm", "scheduling_rl"),
        "validation":       result.get("validation"),
    }
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[runner] 결과 저장 → {full_path}")
    return writer_path if writer_path is not None else full_path


def run_inference_compare(
    env_data: dict,
    algorithms: list[str],
    model_path: Optional[str] = None,
    record_history: bool = False,
    record_decision_log: bool = False,
    rl_agent: Optional[SchedulingAgent] = None,
    enable_wip_inflow: bool = False,
    max_conversions: Optional[int] = None,
    max_conversions_per_eqp: Optional[int] = None,
    conversion_minutes: Optional[int] = None,
) -> dict:
    """
    동일 입력 데이터로 여러 알고리즘 추론 후 비교용 결과 반환
    """
    results: list[dict] = []
    errors: list[dict] = []

    rl_loaded = rl_agent
    loaded_agents: dict = {}
    if rl_loaded is not None:
        loaded_agents["scheduling_rl"] = rl_loaded
    for algo in [a for a in algorithms if a == "scheduling_rl"]:
        if algo in loaded_agents:
            continue
        try:
            loaded_agents[algo] = SchedulingAgent.load(model_path, env_data=env_data)
        except (FileNotFoundError, ValueError) as exc:
            errors.append({"algorithm": algo, "message": str(exc)})

    for algo in algorithms:
        if algo == "scheduling_rl" and algo not in loaded_agents:
            continue
        try:
            validate_algorithm(algo)
            result = run_inference(
                env_data,
                algorithm=algo,
                agent=loaded_agents.get(algo),
                record_history=record_history,
                record_decision_log=record_decision_log,
                enable_wip_inflow=enable_wip_inflow,
                max_conversions=max_conversions,
                max_conversions_per_eqp=max_conversions_per_eqp,
                conversion_minutes=conversion_minutes,
            )
            result["prod_keys"] = env_data["prod_keys"]
            result["oper_ids"] = env_data["oper_ids"]
            result["eqp_ids"] = env_data["eqp_ids"]
            result["sim_end_minutes"] = env_data["sim_end_minutes"]
            results.append(result)
        except Exception as exc:
            errors.append({"algorithm": algo, "message": str(exc)})

    return {
        "results": results,
        "errors": errors,
        "plan": env_data["plan"],
        "prod_keys": env_data["prod_keys"],
        "oper_ids": env_data["oper_ids"],
        "eqp_ids": env_data["eqp_ids"],
        "sim_end_minutes": env_data["sim_end_minutes"],
    }
