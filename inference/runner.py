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
from utils.helpers import minutes_to_str


def run_inference(
    env_data: dict,
    algorithm: str = "rl",
    agent: Optional[SchedulingAgent] = None,
    model_path: Optional[str] = None,
    deterministic: bool = True,
) -> dict:
    """
    목적: 선택한 알고리즘으로 Post-Scheduling 추론 실행
    Input:
        env_data      (dict): preprocessor.preprocess() 반환값
        algorithm     (str):  "rl" | "minprogress" | "earliest_st"
        agent         (SchedulingAgent|None): RL용 (None이면 model_path로 로드)
        model_path    (str|None): 모델 파일 경로
        deterministic (bool): RL 예측 시 greedy 여부
    Output:
        {
          "schedule", "initial_schedule", "history", "stats", "plan", "algorithm"
        }
    """
    algorithm = validate_algorithm(algorithm)

    run_data = dict(env_data)
    if algorithm == "earliest_st":
        run_data["eqp_selection"] = "min_st"

    if algorithm == "rl":
        if agent is None:
            agent = SchedulingAgent.load(model_path)

    heuristic_agent = None
    if algorithm == "minprogress":
        heuristic_agent = MinProgressAgent(env_data)
    elif algorithm == "earliest_st":
        heuristic_agent = EarliestSTAgent(env_data)

    env = SchedulingEnv(run_data)
    obs, _ = env.reset()
    done = False

    while not done:
        if heuristic_agent is not None:
            eqp_id = env.sim.current_idle_eqp()
            lots = env.sim.available_lots(eqp_id) if eqp_id else []
            action = heuristic_agent.predict(env.sim, eqp_id, lots)
        else:
            action = agent.predict(obs, deterministic=deterministic)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    schedule = env.get_schedule()
    history = env.get_history()
    stats = env.sim.stats
    base_time = env_data["sim_base_time"]

    for rec in schedule:
        rec["START_TM_STR"] = minutes_to_str(rec["START_TM"], base_time)
        rec["END_TM_STR"] = minutes_to_str(rec["END_TM"], base_time)

    return {
        "schedule":         schedule,
        "initial_schedule": env_data["initial_schedule"],
        "history":          history,
        "stats":            {
            "idle_total":    stats["idle_total"],
            "oper_switches": stats["oper_switches"],
            "prod_switches": stats["prod_switches"],
            "completed_qty": {str(k): v for k, v in stats["completed_qty"].items()},
        },
        "plan":      env_data["plan"],
        "algorithm": algorithm,
    }


def save_result(result: dict, output_dir: Path = None) -> Path:
    """
    목적: 추론 결과를 external/output/result.json 으로 저장 (DB insert 전 중간 파일)
    """
    d = output_dir or CONFIG.path.output_dir
    d.mkdir(parents=True, exist_ok=True)

    output_records = [
        {
            "EQP_ID":        r["EQP_ID"],
            "LOT_ID":        r["LOT_ID"],
            "CARRIER_ID":    r["CARRIER_ID"],
            "PLAN_PROD_KEY": r["PLAN_PROD_KEY"],
            "ST":            r["ST"],
            "SEQ":           r["SEQ"],
            "START_TM":      r["START_TM_STR"],
            "END_TM":        r["END_TM_STR"],
        }
        for r in result["schedule"]
    ]

    out_path = d / "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    full_path = d / "result_full.json"
    from api.serializers import serialize_history

    serializable = {
        "schedule":         result["schedule"],
        "initial_schedule": result["initial_schedule"],
        "history":          serialize_history(result.get("history", [])),
        "stats":            result["stats"],
        "plan":             result["plan"],
        "algorithm":        result.get("algorithm", "rl"),
    }
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[runner] 결과 저장 → {out_path}")
    return out_path


def run_inference_compare(
    env_data: dict,
    algorithms: list[str],
    model_path: Optional[str] = None,
) -> dict:
    """
    동일 입력 데이터로 여러 알고리즘 추론 후 비교용 결과 반환
    """
    results: list[dict] = []
    errors: list[dict] = []

    rl_agent = None
    if "rl" in algorithms:
        agent = SchedulingAgent()
        if not agent.model_exists():
            errors.append({
                "algorithm": "rl",
                "message": "학습된 모델이 없습니다.",
            })
        else:
            rl_agent = SchedulingAgent.load(model_path)

    for algo in algorithms:
        if algo == "rl" and rl_agent is None:
            continue
        try:
            validate_algorithm(algo)
            result = run_inference(
                env_data,
                algorithm=algo,
                agent=rl_agent if algo == "rl" else None,
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
        "initial_schedule": env_data["initial_schedule"],
        "plan": env_data["plan"],
        "prod_keys": env_data["prod_keys"],
        "oper_ids": env_data["oper_ids"],
        "eqp_ids": env_data["eqp_ids"],
        "sim_end_minutes": env_data["sim_end_minutes"],
    }
