"""
BENCH_SUITE 평가 — 10개 데이터셋 × 3 알고리즘 KPI 비교
=========================================================
실행
  python benchmark/bench_suite.py            # 휴리스틱만(빠름) — bulkfill은 저장 모델 사용
  TS=300000 python benchmark/bench_suite.py  # bulkfill 공동 학습 후 평가

산출
  data/dataset/bench_suite_results.json  (PPT 장표용)
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from inference.runner import run_inference

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
META = json.load(open(SUITE_ROOT / "bench_suite_meta.json", encoding="utf-8"))


def load_ed(m):
    ed = preprocess(load_data(Path(m["dir"])))
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = m["sim"]
    ed["conversion_minutes"] = m["conv"]
    return ed


def kpi(result, m):
    stats = result["stats"]
    sched = result["schedule"]
    sim = m["sim"]; st = m["st"]; n = m["n_eqp"]; maxp = m["total"]
    in_time = [r for r in sched if r.get("END_TM", 0) <= sim]
    by_eqp = defaultdict(lambda: defaultdict(int))
    for r in in_time:
        by_eqp[r["EQP_ID"]][r.get("PLAN_PROD_KEY", "?")] += 1
    eqp_ids = sorted(by_eqp.keys()) or sorted({r["EQP_ID"] for r in sched})
    carriers = [sum(by_eqp[e].values()) for e in eqp_ids]
    ded = sum(1 for e in eqp_ids if by_eqp[e]
              and max(by_eqp[e].values()) / max(sum(by_eqp[e].values()), 1) >= 0.8)
    prod = sum(carriers)
    conv = stats.get("conversions", 0)
    util = round(100 * sum(c * st for c in carriers) / max(n * sim, 1), 1)
    return {
        "prod": prod, "max": maxp, "conv": conv, "loss": maxp - prod,
        "ded": ded, "n_eqp": len(eqp_ids), "util": util,
        "prod_sw": stats.get("prod_switches", 0),
    }


def main():
    TS = int(os.environ.get("TS", "0"))
    eds = [load_ed(m) for m in META]

    agent = None
    if TS > 0:
        from agent.rl_agent import SchedulingAgent
        from env.bulkfill_env import BulkFillEnv
        cfg = CONFIG.reward
        cfg.w_bulk_block_bonus = 3.0
        cfg.w_dedication_misuse = -4.0
        cfg.w_redundant_cover = -5.0
        cfg.w_plan_hit = 1.0
        CONFIG.rl.total_timesteps = TS
        CONFIG.rl.n_steps = 2048
        CONFIG.rl.device = "cpu"
        CONFIG.rl.n_envs = 1
        print(f"=== bulkfill 공동 학습 (datasets={len(eds)}, TS={TS:,}) ===")
        agent = SchedulingAgent()
        agent.train(eds, verbose=0, env_cls=BulkFillEnv)
        agent.save(algorithm="bulkfill")
        print("완료\n")
    else:
        try:
            from agent.rl_agent import SchedulingAgent
            agent = SchedulingAgent.load(env_data=eds[0], algorithm="bulkfill")
        except Exception as e:
            print(f"[경고] 저장된 bulkfill 모델 없음 → bulkfill 생략 ({e})")

    results = []
    algos = [("earliest_st", "Earliest-ST"), ("minprogress", "Min-Progress")]
    print(f"{'dataset':<20}{'algo':<14}{'prod':>9}{'conv':>6}{'util':>7}{'전담':>7}")
    for m, ed in zip(META, eds):
        row = {"name": m["name"], "n": m["n_eqp"], "carriers": m["carriers"],
               "st": m["st"], "conv_min": m["conv"], "total": m["total"], "algos": {}}
        for algo, label in algos:
            k = kpi(run_inference(ed, algorithm=algo), m)
            row["algos"][algo] = k
            print(f"{m['name']:<20}{label:<14}{k['prod']:>4}/{k['max']:<4}{k['conv']:>6}{k['util']:>6.0f}%{k['ded']:>4}/{k['n_eqp']}")
        if agent is not None:
            k = kpi(run_inference(ed, algorithm="bulkfill", agent=agent), m)
            row["algos"]["bulkfill"] = k
            print(f"{m['name']:<20}{'Bulk-Fill':<14}{k['prod']:>4}/{k['max']:<4}{k['conv']:>6}{k['util']:>6.0f}%{k['ded']:>4}/{k['n_eqp']}")
        results.append(row)
        print("")

    # 집계
    def agg(algo):
        rows = [r["algos"][algo] for r in results if algo in r["algos"]]
        if not rows:
            return None
        tot_prod = sum(r["prod"] for r in rows)
        tot_max = sum(r["max"] for r in rows)
        tot_conv = sum(r["conv"] for r in rows)
        tot_ded = sum(r["ded"] for r in rows)
        tot_neqp = sum(r["n_eqp"] for r in rows)
        avg_util = round(np.mean([r["util"] for r in rows]), 1)
        return {
            "prod": tot_prod, "max": tot_max,
            "prod_pct": round(100 * tot_prod / max(tot_max, 1), 1),
            "conv": tot_conv, "util": avg_util,
            "ded": tot_ded, "neqp": tot_neqp,
            "ded_pct": round(100 * tot_ded / max(tot_neqp, 1), 1),
            "optimal": sum(1 for r in rows if r["conv"] == 0 and r["prod"] == r["max"]),
            "n_sets": len(rows),
        }

    summary = {a: agg(a) for a in ["earliest_st", "minprogress", "bulkfill"]}
    out = {"datasets": results, "summary": summary}
    out_path = SUITE_ROOT / "bench_suite_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"{'집계':<14}{'생산률':>10}{'총전환':>8}{'평균가동':>9}{'전담률':>9}{'최적달성':>9}")
    for a, lbl in [("earliest_st", "Earliest-ST"), ("minprogress", "Min-Progress"), ("bulkfill", "Bulk-Fill")]:
        s = summary[a]
        if not s:
            continue
        print(f"{lbl:<14}{s['prod_pct']:>9.1f}%{s['conv']:>8}{s['util']:>8.1f}%"
              f"{s['ded_pct']:>8.1f}%{s['optimal']:>6}/{s['n_sets']}")
    print(f"\n저장 → {out_path}")


if __name__ == "__main__":
    main()
