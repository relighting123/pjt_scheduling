"""
inference/runner.py – 추론 실행 및 결과 저장
학습된 에이전트로 스케줄링을 실행하고, 결과 JSON을 external/output/에 저장합니다.
"""
import json
from datetime import timedelta
from pathlib import Path
from typing import Optional

from config import CONFIG
from agent.rl_agent import SchedulingAgent
from env.scheduling_env import SchedulingEnv
from utils.helpers import minutes_to_str


def run_inference(
    env_data: dict,
    agent: Optional[SchedulingAgent] = None,
    model_path: Optional[str] = None,
    deterministic: bool = True,
) -> dict:
    """
    목적: 학습된 에이전트로 Post-Scheduling 추론 실행
    Input:
        env_data      (dict): preprocessor.preprocess() 반환값
        agent         (SchedulingAgent|None): 이미 로드된 에이전트 (None이면 model_path로 로드)
        model_path    (str|None): 모델 파일 경로
        deterministic (bool): True=결정론적 예측
    Output:
        {
          "schedule":          [...],  # POST 스케줄 결과
          "initial_schedule":  [...],  # 초기 스케줄 (비교용)
          "history":           [...],  # UI 재생용 히스토리
          "stats":             {...},  # 통계
          "plan":              [...],  # 계획 데이터
        }
    """
    if agent is None:
        agent = SchedulingAgent.load(model_path)

    env = SchedulingEnv(env_data)
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    schedule  = env.get_schedule()
    history   = env.get_history()
    stats     = env.sim.stats
    base_time = env_data["sim_base_time"]

    # START_TM / END_TM을 사람이 읽기 쉬운 문자열로도 추가
    for rec in schedule:
        rec["START_TM_STR"] = minutes_to_str(rec["START_TM"], base_time)
        rec["END_TM_STR"]   = minutes_to_str(rec["END_TM"],   base_time)

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
        "plan": env_data["plan"],
    }


def save_result(result: dict, output_dir: Path = None) -> Path:
    """
    목적: 추론 결과를 external/output/result.json 으로 저장 (DB insert 전 중간 파일)
    Input:
        result     (dict): run_inference() 반환값
        output_dir (Path): 저장 디렉터리 (None이면 CONFIG 기본값)
    Output:
        저장된 파일 경로 (Path)
    """
    d = output_dir or CONFIG.path.output_dir
    d.mkdir(parents=True, exist_ok=True)

    # DB insert 용 레코드만 별도 저장 (4개 컬럼 명세)
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

    # 전체 결과 (히스토리 포함) 별도 저장 – UI에서 불러다 쓸 수 있음
    full_path = d / "result_full.json"
    # history는 직렬화 가능하게 tuple key → str
    serializable = {
        "schedule":         result["schedule"],
        "initial_schedule": result["initial_schedule"],
        "stats":            result["stats"],
        "plan":             result["plan"],
        # history는 크기가 클 수 있으므로 저장 여부 선택적
    }
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[runner] 결과 저장 → {out_path}")
    return out_path
