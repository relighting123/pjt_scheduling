"""스텝별 state/reward 트레이스 캡처 — PPT 상세 슬라이드용."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from config import CONFIG
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from agent.rl_agent import SchedulingAgent
from env.bulkfill_env import BulkFillEnv

DS = ROOT / "data/dataset/SYM_3x3/train/20260629000000/input"
SIM = 480
# bulk 보상 가중치(평가와 동일)
cfg = CONFIG.reward
cfg.w_bulk_block_bonus = 3.0; cfg.w_dedication_misuse = -4.0
cfg.w_redundant_cover = -5.0; cfg.w_plan_hit = 1.0

ed = preprocess(load_data(DS)); ed["eqp_selection"] = "order"
ed["sim_end_minutes"] = SIM; ed["conversion_minutes"] = 60
agent = SchedulingAgent.load(env_data=ed, algorithm="bulkfill")

env = BulkFillEnv(ed, record_history=False, record_event_log=False, truncate_on_time=False)
obs, _ = env.reset()
sim = env.sim
total_plan = sum(p["d0_plan_qty"] for p in ed["plan"])

rows = []
cum = 0.0
done = False
step = 0
while not done and step < 14:
    t_before = sim.current_time
    prog_before = obs[3]
    eqp_before = sim.current_idle_eqp()
    mask = env.action_masks()
    action = agent.predict(obs, deterministic=True, action_masks=mask, env_data=ed)
    arr = np.asarray(action).flatten()
    bucket = int(arr[0]); level = int(arr[1]) if arr.size > 1 else 0
    ppk, oper = sim.ppk_oper_from_flat(bucket % (CONFIG.env.max_oper_count*CONFIG.env.max_prod_count))
    obs, reward, terminated, truncated, info = env.step(action)
    cum += reward
    done = terminated or truncated
    produced = sum(sim.stats["completed_qty"].values())
    rows.append({
        "step": step + 1,
        "t": int(t_before),
        "eqp": eqp_before or (info.get("current_eqp") or "-"),
        "ppk": ppk,
        "block_lv": level,
        "reward": round(float(reward), 2),
        "cum": round(float(cum), 2),
        "prog": round(float(produced / max(total_plan, 1)), 3),
        "conv": int(info.get("conversions", 0)),
        "produced": produced,
    })
    step += 1

out = {"total_plan": total_plan, "sim": SIM, "rows": rows}
json.dump(out, open(ROOT / "docs/trace_steps.json", "w"),
          ensure_ascii=False, indent=2)
print(f"{'st':>3}{'t':>5}{'eqp':>8}{'ppk':>8}{'lv':>3}{'reward':>8}{'cumΣ':>8}{'prog':>7}{'conv':>5}")
for r in rows:
    print(f"{r['step']:>3}{r['t']:>5}{r['eqp']:>8}{r['ppk']:>8}{r['block_lv']:>3}"
          f"{r['reward']:>8.2f}{r['cum']:>8.2f}{r['prog']:>7.2f}{r['conv']:>5}")
