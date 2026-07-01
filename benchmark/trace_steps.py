"""스텝별 state / action / reward / 간트 스냅샷 캡처 — PPT 상세 슬라이드용."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from agent.rl_agent import SchedulingAgent
from env.bulkfill_env import BulkFillEnv

DS = ROOT / "data/dataset/SYM_3x3/train/20260629000000/input"
SIM = 480

# bulk 보상 가중치(평가·PPT와 동일)
cfg = CONFIG.reward
cfg.w_bulk_block_bonus = 3.0
cfg.w_dedication_misuse = -4.0
cfg.w_redundant_cover = -5.0
cfg.w_plan_hit = 1.0

REWARD_LABELS = {
    "same_setup": "동일 셋업",
    "pacing": "페이싱",
    "plan_hit": "계획 달성",
    "flow_balance": "흐름 균형",
    "idle": "유휴",
    "conversion": "전환",
    "avoidable_conversion": "회피가능 전환",
    "bulk_block_bonus": "블록 보너스",
    "dedication_misuse": "전용 오용",
    "redundant_cover": "중복 커버",
}


def _schedule_snapshot(schedule: list) -> list:
    return [
        {
            "EQP_ID": r["EQP_ID"],
            "LOT_ID": r.get("LOT_ID", ""),
            "PPK": r["PLAN_PROD_KEY"],
            "OPER": r.get("OPER_ID", ""),
            "START_TM": int(r["START_TM"]),
            "END_TM": int(r["END_TM"]),
        }
        for r in schedule
    ]


def _state_summary(obs: np.ndarray, sim, total_plan: int) -> dict:
    produced = sum(sim.stats["completed_qty"].values())
    return {
        "time_min": int(sim.current_time),
        "progress_pct": round(100 * produced / max(total_plan, 1), 1),
        "produced": int(produced),
        "conversions": int(sim.stats.get("conversions", 0)),
        "idle_eqps": len(sim.get_idle_eqps()),
        "obs_global": {
            "time_norm": round(float(obs[0]), 3),
            "takt_margin": round(float(obs[1]), 3),
            "remaining_lots": round(float(obs[2]), 3),
            "plan_progress": round(float(obs[3]), 3),
        },
    }


def _dedicated_bucket(sim, eqp_id: str, ed: dict) -> int:
    """SYM_3x3 등 대칭 벤치: EQP{i} → PPK{i} 전담 버킷."""
    eqp_ids = ed["eqp_ids"]
    prod_keys = ed["prod_keys"]
    oper_ids = ed["oper_ids"]
    if eqp_id not in eqp_ids:
        return 0
    ei = eqp_ids.index(eqp_id)
    ppk = prod_keys[ei % len(prod_keys)]
    oper = oper_ids[0]
    return sim.ppk_oper_flat_index(oper, ppk)


def _load_agent(ed: dict):
    try:
        return SchedulingAgent.load(env_data=ed, algorithm="bulkfill")
    except FileNotFoundError:
        return None


def main() -> None:
    ed = preprocess(load_data(DS))
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = SIM
    ed["conversion_minutes"] = 60
    agent = _load_agent(ed)

    env = BulkFillEnv(
        ed,
        record_history=True,
        record_event_log=False,
        record_decision_log=True,
        truncate_on_time=False,
    )
    obs, _ = env.reset()
    sim = env.sim
    total_plan = sum(p["d0_plan_qty"] for p in ed["plan"])
    eqp_ids = list(ed["eqp_ids"])

    steps = []
    cum = 0.0
    done = False
    step = 0
    while not done and step < 14:
        t_before = int(sim.current_time)
        eqp_before = sim.current_idle_eqp()
        mask = env.action_masks()
        if agent is not None:
            action = agent.predict(obs, deterministic=True, action_masks=mask, env_data=ed)
        else:
            # 모델 없을 때: SYM_3x3 전담 + 최대 블록(레벨 3) 스크립트 정책
            flat = _dedicated_bucket(sim, eqp_before or "", ed) if eqp_before else 0
            action = np.array([flat, 3], dtype=np.int64)
        arr = np.asarray(action).flatten()
        bucket = int(arr[0])
        level = int(arr[1]) if arr.size > 1 else 0
        ppk, oper = sim.ppk_oper_from_flat(
            bucket % (CONFIG.env.max_oper_count * CONFIG.env.max_prod_count),
        )
        state_before = _state_summary(obs, sim, total_plan)

        obs, reward, terminated, truncated, info = env.step(action)
        cum += float(reward)
        done = terminated or truncated

        log_entry = env.get_decision_log()[-1] if env.get_decision_log() else {}
        breakdown = dict(log_entry.get("reward_breakdown") or sim._last_reward_breakdown or {})
        block_size = int(log_entry.get("block_size") or 0)
        block_start = bool(log_entry.get("block_start"))
        assigned_lot = log_entry.get("assigned_lot_id", "")

        steps.append({
            "step": step + 1,
            "t": t_before,
            "eqp": eqp_before or info.get("current_eqp") or "-",
            "ppk": ppk,
            "oper": oper,
            "action": {
                "bucket": bucket,
                "level": level,
                "block_size": block_size,
                "block_start": block_start,
                "assigned_lot": assigned_lot,
            },
            "state": state_before,
            "reward": round(float(reward), 2),
            "cum": round(float(cum), 2),
            "reward_breakdown": {k: round(float(v), 2) for k, v in breakdown.items()},
            "schedule": _schedule_snapshot(sim.schedule),
        })
        step += 1

    out = {
        "dataset": "SYM_3x3",
        "total_plan": total_plan,
        "sim": SIM,
        "eqp_ids": eqp_ids,
        "prod_keys": list(ed["prod_keys"]),
        "oper_ids": list(ed["oper_ids"]),
        "reward_labels": REWARD_LABELS,
        "rows": [
            {
                "step": s["step"],
                "t": s["t"],
                "eqp": s["eqp"],
                "ppk": s["ppk"],
                "block_lv": s["action"]["level"],
                "reward": s["reward"],
                "cum": s["cum"],
                "prog": round(s["state"]["produced"] / max(total_plan, 1), 3),
                "conv": s["state"]["conversions"],
                "produced": s["state"]["produced"],
            }
            for s in steps
        ],
        "steps": steps,
    }
    out_path = ROOT / "docs/trace_steps.json"
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"저장: {out_path} ({len(steps)} steps)")
    print(f"{'st':>3}{'t':>5}{'eqp':>8}{'ppk':>8}{'lv':>3}{'reward':>8}{'cumΣ':>8}{'bars':>6}")
    for s in steps:
        print(
            f"{s['step']:>3}{s['t']:>5}{s['eqp']:>8}{s['ppk']:>8}"
            f"{s['action']['level']:>3}{s['reward']:>8.2f}{s['cum']:>8.2f}"
            f"{len(s['schedule']):>6}",
        )


if __name__ == "__main__":
    main()
