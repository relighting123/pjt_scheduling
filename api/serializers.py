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


def serialize_inference_result(result: dict, *, include_history: bool = True) -> dict:
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
        "algorithm": result.get("algorithm", "rl"),
    }
    if include_history:
        payload["history"] = serialize_history(result.get("history", []))
        payload["event_log"] = result.get("event_log", [])
    else:
        payload["history"] = []
        payload["event_log"] = []
    return payload


def serialize_compare_response(payload: dict) -> dict:
    return {
        "results": [serialize_inference_result(r) for r in payload["results"]],
        "errors": payload.get("errors", []),
        "plan": payload["plan"],
        "prod_keys": payload.get("prod_keys", []),
        "oper_ids": payload.get("oper_ids", []),
        "eqp_ids": payload.get("eqp_ids", []),
        "sim_end_minutes": payload.get("sim_end_minutes", 0),
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
