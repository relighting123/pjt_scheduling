"""
BENCH_SUITE 기준 "RL vs Dedication" 비교 러너
=============================================
`bench_suite.py`가 휴리스틱 3종만 비교하던 것과 달리, 이 스크립트는
**개선 목표 기준선인 `dedication`(전담 배분 휴리스틱)** 을 항상 포함하고
데이터셋별 승패(win/tie/loss)까지 집계한다.

실행
  python benchmark/compare_vs_dedication.py                 # 저장된 RL 모델로 평가
  python benchmark/compare_vs_dedication.py --no-rl         # 휴리스틱만
  python benchmark/compare_vs_dedication.py --model PATH    # 특정 모델 zip 지정
  python benchmark/compare_vs_dedication.py --json OUT.json # 결과 JSON 저장

KPI 점수(높을수록 좋음)
  score = 생산량(sim_end 내 완료 carrier) − conv_weight × 전환수
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader.fetch import load_data          # noqa: E402
from data.loader.preprocess import preprocess    # noqa: E402
from inference.runner import run_inference       # noqa: E402

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"

# 비교 대상. dedication 이 기준선(baseline)이다.
ALGOS = [
    ("dedication", "Dedication"),
    ("minprogress", "Min-Progress"),
    ("earliest_st", "Earliest-ST"),
    ("scheduling_rl", "Bulk-Fill RL"),
]
BASELINE = "dedication"


SUITE_META = {
    "bench": ("bench_suite_meta.json", "benchmark/gen_bench_suite.py"),
    "holdout": ("holdout_suite_meta.json", "benchmark/gen_holdout_suite.py"),
    "train_pool": ("train_pool_meta.json", "benchmark/gen_train_pool.py"),
}


def load_meta(suite: str = "bench") -> list:
    filename, generator = SUITE_META.get(suite, SUITE_META["bench"])
    path = SUITE_ROOT / filename
    if not path.exists():
        raise SystemExit(f"{path} 없음 — 먼저 `python {generator}` 를 실행하세요.")
    return json.load(open(path, encoding="utf-8"))


def load_ed(m: dict) -> dict:
    ed = preprocess(load_data(Path(m["dir"])))
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = m["sim"]
    ed["conversion_minutes"] = m["conv"]
    ed["enable_wip_inflow"] = bool(m.get("enable_wip_inflow", False))
    return ed


def kpi(result: dict, m: dict, conv_weight: float = 1.0) -> dict:
    """bench_suite.kpi()와 같은 정의 + score(단일 비교지표)."""
    stats = result["stats"]
    sched = result["schedule"]
    sim, n_eqp, maxp = m["sim"], m["n_eqp"], m["total"]

    in_time = [r for r in sched if r.get("END_TM", 0) <= sim]
    by_eqp: dict = defaultdict(lambda: defaultdict(int))
    busy = defaultdict(int)
    for r in in_time:
        by_eqp[r["EQP_ID"]][r.get("PLAN_PROD_ATTR_VAL", "?")] += 1
        busy[r["EQP_ID"]] += int(r.get("PROC_TIME") or (r["END_TM"] - r["START_TM"]))

    eqp_ids = sorted(by_eqp.keys()) or sorted({r["EQP_ID"] for r in sched})
    prod = sum(sum(v.values()) for v in by_eqp.values())
    conv = stats.get("conversions", 0)
    ded = sum(
        1 for e in eqp_ids
        if by_eqp[e] and max(by_eqp[e].values()) / max(sum(by_eqp[e].values()), 1) >= 0.8
    )
    return {
        "prod": prod,
        "max": maxp,
        "conv": conv,
        "loss": maxp - prod,
        "ded": ded,
        "n_eqp": len(eqp_ids),
        "util": round(100 * sum(busy.values()) / max(n_eqp * sim, 1), 1),
        "score": round(prod - conv_weight * conv, 3),
    }


def run_suite(
    use_rl: bool = True,
    model_path: str | None = None,
    conv_weight: float = 1.0,
    quiet: bool = False,
    suite: str = "bench",
) -> dict:
    meta = load_meta(suite)
    eds = [load_ed(m) for m in meta]

    agent = None
    if use_rl:
        try:
            from agent.rl_agent import SchedulingAgent
            agent = SchedulingAgent.load(model_path, env_data=eds[0])
        except Exception as exc:            # 모델 없음/obs_dim 불일치 등
            print(f"[경고] RL 모델 로드 실패 → scheduling_rl 생략 ({exc})")

    algos = [(a, l) for a, l in ALGOS if a != "scheduling_rl" or agent is not None]

    rows = []
    if not quiet:
        header = f"{'dataset':<13}" + "".join(f"{l:>16}" for _, l in algos)
        print(header)
        print("-" * len(header))

    for m, ed in zip(meta, eds):
        row = {"name": m["id"], "cat": m["cat"], "algos": {}}
        for algo, _label in algos:
            res = run_inference(
                ed, algorithm=algo,
                agent=agent if algo == "scheduling_rl" else None,
                record_history=False,
                enable_wip_inflow=bool(ed.get("enable_wip_inflow", False)),
            )
            row["algos"][algo] = kpi(res, m, conv_weight)
        rows.append(row)
        if not quiet:
            cells = "".join(
                f"{row['algos'][a]['prod']:>7}/{row['algos'][a]['max']:<3}"
                f"c{row['algos'][a]['conv']:<4}"
                for a, _ in algos
            )
            print(f"{m['id']:<13}{cells}")

    summary = {}
    for algo, _label in algos:
        vals = [r["algos"][algo] for r in rows]
        summary[algo] = {
            "prod": sum(v["prod"] for v in vals),
            "max": sum(v["max"] for v in vals),
            "conv": sum(v["conv"] for v in vals),
            "score": round(sum(v["score"] for v in vals), 3),
            "ded": sum(v["ded"] for v in vals),
            "n_eqp": sum(v["n_eqp"] for v in vals),
            "optimal": sum(1 for v in vals if v["conv"] == 0 and v["prod"] == v["max"]),
            "n_sets": len(vals),
        }
        summary[algo]["prod_pct"] = round(
            100 * summary[algo]["prod"] / max(summary[algo]["max"], 1), 1,
        )

    # dedication 대비 승패
    versus = {}
    if BASELINE in summary:
        for algo, _label in algos:
            if algo == BASELINE:
                continue
            win = tie = loss = 0
            for r in rows:
                d = r["algos"][BASELINE]["score"]
                a = r["algos"][algo]["score"]
                win += a > d
                tie += a == d
                loss += a < d
            versus[algo] = {
                "win": win, "tie": tie, "loss": loss,
                "score_delta": round(summary[algo]["score"] - summary[BASELINE]["score"], 3),
                "prod_delta": summary[algo]["prod"] - summary[BASELINE]["prod"],
                "conv_delta": summary[algo]["conv"] - summary[BASELINE]["conv"],
            }

    out = {"suite": suite, "datasets": rows, "summary": summary,
           "vs_dedication": versus, "conv_weight": conv_weight}

    if not quiet:
        print("\n" + "=" * 72)
        print(f"{'집계':<15}{'생산':>12}{'전환':>7}{'점수':>9}{'전담':>9}{'최적':>8}")
        for algo, label in algos:
            s = summary[algo]
            print(f"{label:<15}{s['prod']:>5}/{s['max']:<6}{s['conv']:>7}"
                  f"{s['score']:>9.1f}{s['ded']:>6}/{s['n_eqp']:<3}{s['optimal']:>5}/{s['n_sets']}")
        if versus:
            print("\n" + "-" * 72)
            print("Dedication 기준선 대비")
            for algo, v in versus.items():
                verdict = "WIN " if v["score_delta"] > 0 else ("TIE " if v["score_delta"] == 0 else "LOSS")
                print(f"  {algo:<16}{verdict} 점수 {v['score_delta']:+.1f}  "
                      f"생산 {v['prod_delta']:+d}  전환 {v['conv_delta']:+d}  "
                      f"(데이터셋 승/무/패 {v['win']}/{v['tie']}/{v['loss']})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rl", action="store_true", help="RL 모델 평가 생략")
    ap.add_argument("--model", default=None, help="RL 모델 zip 경로")
    ap.add_argument("--conv-weight", type=float, default=1.0)
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--suite", default="bench", choices=sorted(SUITE_META),
                    help="bench=학습에 쓰는 8종, holdout=학습에 안 쓰는 일반화 검증용")
    args = ap.parse_args()

    out = run_suite(
        use_rl=not args.no_rl, model_path=args.model, conv_weight=args.conv_weight,
        suite=args.suite,
    )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n저장 → {path}")


if __name__ == "__main__":
    main()
