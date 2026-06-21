"""
validation/runner.py – test 데이터셋 기준 모델 검증
"""
from typing import List, Optional

from config import (
    parse_input_folder,
    set_input_folder,
    validate_path_segment,
    list_split_folders,
)
from data.loader import fetch_from_db, load_data, validate_data, preprocess
from agent.rl_agent import SchedulingAgent
from inference.runner import run_inference


def _load_env_data(folder: str) -> dict:
    set_input_folder(folder)
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        raise ValueError("; ".join(errors))
    return preprocess(raw)


def _refresh_test_sql(fac_id: str, folder: str) -> None:
    _, _, period = parse_input_folder(folder)
    if not period:
        raise ValueError(f"test 폴더에 RULE_TIMEKEY가 없습니다: {folder}")
    fetch_from_db(fac_id=fac_id, split="test", period=period)


def run_validation(
    fac_id: str,
    agent: Optional[SchedulingAgent] = None,
    *,
    refresh_sql: bool = True,
) -> dict:
    """
    FAC의 test 데이터 폴더 전체에 대해 RL 추론 검증.

    Returns:
        results: [{folder, stats, schedule_count}, ...]
        errors:  [{folder, message}, ...]
    """
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    folders = list_split_folders(fac_id, "test")
    if not folders:
        raise ValueError(f"test 데이터셋이 없습니다 (FAC_ID={fac_id}).")

    if agent is None:
        agent = SchedulingAgent()
        if not agent.model_exists():
            raise ValueError("학습된 모델이 없습니다. 먼저 train을 실행하세요.")
        agent = SchedulingAgent.load()

    results: List[dict] = []
    errors: List[dict] = []

    for folder in folders:
        try:
            if refresh_sql:
                _refresh_test_sql(fac_id, folder)
            env_data = _load_env_data(folder)
            result = run_inference(
                env_data, algorithm="rl", agent=agent, record_history=False,
            )
            stats = result["stats"]
            results.append({
                "folder":         folder,
                "schedule_count": len(result["schedule"]),
                "stats":          stats,
            })
            print(
                f"  [{folder}] LOT {len(result['schedule'])} · "
                f"oper_sw={stats['oper_switches']} · "
                f"prod_sw={stats['prod_switches']} · "
                f"idle={stats['idle_total']}분"
            )
        except Exception as exc:
            errors.append({"folder": folder, "message": str(exc)})
            print(f"  [{folder}] 오류: {exc}")

    return {"fac_id": fac_id, "results": results, "errors": errors}
