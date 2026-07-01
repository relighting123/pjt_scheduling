"""API 응답용 JSON 직렬화 헬퍼."""
from datetime import datetime
from typing import Any


def _serialize_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(str(k) for k in key)
    return str(key)


def serialize_completed(completed: dict) -> dict[str, int]:
    return {_serialize_key(k): int(v) for k, v in completed.items()}


def serialize_history(history: list[dict]) -> list[dict]:
    out = []
    for snap in history:
        item = dict(snap)
        if "completed" in item:
            item["completed"] = serialize_completed(item["completed"])
        out.append(item)
    return out


def serialize_inference_result(
    result: dict,
    *,
    include_history: bool = True,
    include_event_log: bool | None = None,
    include_decision_log: bool | None = None,
) -> dict:
    if include_event_log is None:
        include_event_log = include_history
    if include_decision_log is None:
        include_decision_log = include_history
    payload = {
        "schedule": result["schedule"],
        "stats": {
            **result["stats"],
            "completed_qty": serialize_completed(result["stats"].get("completed_qty", {})),
        },
        "plan": result["plan"],
        "prod_keys": result.get("prod_keys", []),
        "oper_ids": result.get("oper_ids", []),
        "eqp_ids": result.get("eqp_ids", []),
        "sim_end_minutes": result.get("sim_end_minutes", 0),
        "sim_base_time": result.get("sim_base_time"),
        "algorithm": result.get("algorithm", "rl"),
    }
    if include_history:
        payload["history"] = serialize_history(result.get("history", []))
    else:
        payload["history"] = []
    payload["event_log"] = result.get("event_log", []) if include_event_log else []
    payload["decision_log"] = result.get("decision_log", []) if include_decision_log else []
    payload["conversion_plans"] = result.get("conversion_plans", [])
    return payload


def serialize_compare_response(payload: dict, *, include_history: bool = False) -> dict:
    return {
        "results": [
            serialize_inference_result(
                r,
                include_history=include_history,
                include_event_log=include_history,
                include_decision_log=include_history,
            )
            for r in payload["results"]
        ],
        "errors": payload.get("errors", []),
        "plan": payload["plan"],
        "prod_keys": payload.get("prod_keys", []),
        "oper_ids": payload.get("oper_ids", []),
        "eqp_ids": payload.get("eqp_ids", []),
        "sim_end_minutes": payload.get("sim_end_minutes", 0),
        "sim_base_time": _compare_sim_base_time(payload),
    }


def _compare_sim_base_time(payload: dict) -> str | None:
    results = payload.get("results") or []
    for row in results:
        base = row.get("sim_base_time")
        if base:
            return base if isinstance(base, str) else str(base)
    return None


def empty_data_summary() -> dict:
    """입력 dataset 없을 때 UI용 빈 요약."""
    return {
        "eqp_count": 0,
        "lot_count": 0,
        "prod_count": 0,
        "oper_count": 0,
        "batch_info_count": 0,
        "sim_end_minutes": 0,
        "sim_base_time": "",
        "eqp_ids": [],
        "prod_keys": [],
        "oper_ids": [],
        "batch_info": [],
        "warnings": [],
    }


def env_data_summary(env_data: dict) -> dict:
    base: datetime = env_data["sim_base_time"]
    batch_info_map = env_data.get("batch_info_map", {})
    batch_info = [
        {
            "plan_prod_key": ppk,
            "oper_id": oper_id,
            "lot_cd": info["lot_cd"],
            "temp": info["temp"],
        }
        for (ppk, oper_id), info in sorted(batch_info_map.items())
    ]
    return {
        "eqp_count": len(env_data["eqp_ids"]),
        "lot_count": len(env_data["lots"]),
        "prod_count": len(env_data["prod_keys"]),
        "oper_count": len(env_data["oper_ids"]),
        "batch_info_count": len(batch_info),
        "sim_end_minutes": env_data["sim_end_minutes"],
        "sim_base_time": base.isoformat(sep=" "),
        "eqp_ids": env_data["eqp_ids"],
        "prod_keys": env_data["prod_keys"],
        "oper_ids": env_data["oper_ids"],
        "batch_info": batch_info,
    }
