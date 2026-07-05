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
from benchmark.reward_formula_trace import REWARD_LABELS, build_reward_formula_details

DS = ROOT / "data/dataset/SYM_3x3/train/20260629000000/input"
SIM = 480

cfg = CONFIG.reward
cfg.w_bulk_block_bonus = 3.0
cfg.w_dedication_misuse = -4.0
cfg.w_redundant_cover = -5.0
cfg.w_plan_hit = 1.0


def _schedule_snapshot(schedule: list) -> list:
    return [
        {
            "EQP_ID": r["EQP_ID"],
            "LOT_ID": r.get("LOT_ID", ""),
            "PPK": r["PLAN_PROD_ATTR_VAL"],
            "OPER": r.get("OPER_ID", ""),
            "START_TM": int(r["START_TM"]),
            "END_TM": int(r["END_TM"]),
        }
        for r in schedule
    ]


def _bucket_state(sim, ppk: str, oper: str, eqp_id: str) -> dict:
    """선택된 (ppk, oper, eqp의 model) 버킷 — 13채널 실측치."""
    ed = sim._env_data
    oper_ids = list(ed["oper_ids"])
    prod_keys = list(ed["prod_keys"])
    if oper not in oper_ids or ppk not in prod_keys:
        return {}
    oi = oper_ids.index(oper)
    pi = prod_keys.index(ppk)
    eqp_models = list(ed.get("eqp_models", []))
    eqp_model_map = ed.get("eqp_model_map", {})
    model = eqp_model_map.get(eqp_id)
    mi = eqp_models.index(model) if model in eqp_models else 0
    feats = sim.get_bucket_features()
    if oi >= feats.shape[0] or pi >= feats.shape[1] or mi >= feats.shape[2]:
        return {}
    v = feats[oi, pi, mi]
    r = lambda x: round(float(x), 3)
    return {
        "wip_ratio": f"{r(v[0])}/{r(v[1])}",
        "takt": f"prev={r(v[2])}, post={r(v[3])}",
        "self_st": r(v[4]),
        "plan_urgency": r(v[5]),
        "needs_conversion": r(v[8]),
        "tool_can_assign": r(v[9]),
        "avoidable_frac": r(v[13]),
        "achievable_ratio": r(v[10]),
        "projected_cover_ratio": r(v[11]),
        "starve_time_norm": r(v[12]),
    }


def _state_summary(obs: np.ndarray, sim, total_plan: int, ppk: str = "", oper: str = "", eqp_id: str = "") -> dict:
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
            "conv_idle_ratio": round(float(obs[4]), 3),
            "tool_util": round(float(obs[5]), 3),
        },
        "obs_bucket": _bucket_state(sim, ppk, oper, eqp_id),
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
    done_flag = False
    step = 0
    active_blocks: dict[str, dict] = {}
    while not done_flag and step < 14:
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
        state_before = _state_summary(obs, sim, total_plan, ppk=ppk, oper=oper, eqp_id=eqp_before or "")
        eqp_obj = sim.eqps.get(eqp_before) if eqp_before else None
        done_before = sim.stats["completed_qty"].get((ppk, oper), 0)
        prev_prod = eqp_obj.prev_prod if eqp_obj else None
        prev_oper = eqp_obj.prev_oper if eqp_obj else None

        obs, reward, terminated, truncated, info = env.step(action)
        cum += float(reward)
        done_flag = terminated or truncated

        log_entry = env.get_decision_log()[-1] if env.get_decision_log() else {}
        breakdown = dict(log_entry.get("reward_breakdown") or sim._last_reward_breakdown or {})
        block_size = int(log_entry.get("block_size") or 0)
        block_start = bool(log_entry.get("block_start"))
        assigned_lot = log_entry.get("assigned_lot_id", "")
        wf_qty = 1
        proc_min = 60
        if sim.schedule:
            last = sim.schedule[-1]
            if last.get("EQP_ID") == eqp_before and last.get("PLAN_PROD_ATTR_VAL") == ppk:
                wf_qty = int(last.get("WF_QTY") or 1)
                proc_min = max(int(last.get("END_TM", 0) - last.get("START_TM", 0)), 1)

        eqp_key = eqp_before or info.get("current_eqp") or ""
        if block_start and block_size > 0 and eqp_key and sim.schedule:
            last_bar = sim.schedule[-1]
            start_tm = int(last_bar["START_TM"])
            active_blocks[eqp_key] = {
                "eqp_id": eqp_key,
                "ppk": ppk,
                "oper": oper,
                "total": block_size,
                "done": 1,
                "remaining": max(block_size - 1, 0),
                "proc_min": proc_min,
                "start_tm": start_tm,
                "committed_end_tm": start_tm + block_size * proc_min,
                "scheduled_end_tm": int(last_bar["END_TM"]),
                "block_start": True,
            }
        elif eqp_key in active_blocks and active_blocks[eqp_key]["ppk"] == ppk:
            blk = active_blocks[eqp_key]
            blk["done"] += 1
            blk["remaining"] = max(blk["total"] - blk["done"], 0)
            blk["block_start"] = False
            ends = [
                int(b["END_TM"])
                for b in sim.schedule
                if b["EQP_ID"] == eqp_key and b["PLAN_PROD_ATTR_VAL"] == ppk
            ]
            if ends:
                blk["scheduled_end_tm"] = max(ends)
            if blk["remaining"] <= 0:
                del active_blocks[eqp_key]

        current_block = active_blocks.get(eqp_key)
        blocks_snapshot = [
            {
                **b,
                "scheduled_end_tm": b.get(
                    "scheduled_end_tm",
                    b["start_tm"] + b["done"] * b["proc_min"],
                ),
            }
            for b in active_blocks.values()
        ]

        formula_details = build_reward_formula_details(
            sim,
            ppk=ppk,
            oper_id=oper,
            eqp_id=eqp_before or "",
            wf_qty=wf_qty,
            t=t_before,
            breakdown=breakdown,
            block_start=block_start,
            block_size=block_size,
            eqp_prev_prod=prev_prod,
            eqp_prev_oper=prev_oper,
            done_before=done_before,
            include_zero=True,
        )

        steps.append({
            "step": step + 1,
            "t": t_before,
            "eqp": eqp_before or info.get("current_eqp") or "-",
            "ppk": ppk,
            "oper": oper,
            "action": {
                "bucket": bucket,
                "level": level,
                "block_size": block_size or (current_block or {}).get("total", 0),
                "block_start": block_start,
                "block_continuation": bool(
                    current_block and not block_start and current_block.get("remaining", 0) >= 0,
                ),
                "block_done": (current_block or {}).get("done"),
                "block_total": (current_block or {}).get("total"),
                "assigned_lot": assigned_lot,
                "wf_qty": wf_qty,
            },
            "block": dict(current_block) if current_block else None,
            "blocks": blocks_snapshot,
            "state": state_before,
            "reward": round(float(reward), 2),
            "cum": round(float(cum), 2),
            "reward_breakdown": {k: round(float(v), 2) for k, v in breakdown.items()},
            "reward_formula": formula_details,
            "reward_formula_full": formula_details,
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
