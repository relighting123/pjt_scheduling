"""
benchmark/optimal/runner.py — 검증 가능한 최적해 벤치마크 실행기

benchmark/optimal/cases.py의 각 케이스는 증명된 최적값(OptimalTarget)을
가지고 있다. 이 스크립트는 실제 알고리즘(agent.registry에 등록된 각각)을
inference.runner.run_inference로 실행해 그 값에 도달하는지 채점한다.

실행
  python -m benchmark.optimal.runner
  python -m benchmark.optimal.runner --algo earliest_st --algo minprogress
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agent.registry import ALGORITHMS
from inference.runner import run_inference

from benchmark.optimal.cases import CASES, OptimalCase, measure

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "data/dataset/OPTIMAL_BENCH/optimal_bench_results.json"


def _load_rl_agent(env_data: dict):
    from agent.rl_agent import SchedulingAgent
    return SchedulingAgent.load(env_data=env_data)


def evaluate_case(case: OptimalCase, algorithm: str, agent=None) -> dict:
    """케이스 1건 × 알고리즘 1개를 실행하고 증명된 최적값과 비교해 채점."""
    env_data = case.build()
    result = run_inference(env_data, algorithm=algorithm, agent=agent, record_history=False)
    actual = measure(result, env_data["sim_end_minutes"])
    target = case.optimal
    passed = actual.production == target.production and actual.conversions <= target.conversions
    return {
        "case": case.id,
        "algorithm": algorithm,
        "target": asdict(target),
        "actual": asdict(actual),
        "passed": passed,
    }


def run_optimal_benchmark(algorithms: Optional[list[str]] = None) -> dict:
    algo_ids = list(algorithms) if algorithms else [a["id"] for a in ALGORITHMS]

    rl_agent = None
    if "scheduling_rl" in algo_ids:
        try:
            rl_agent = _load_rl_agent(CASES[0].build())
        except Exception as exc:
            print(f"[경고] 저장된 scheduling_rl 모델 없음 → scheduling_rl 생략 ({exc})")
            algo_ids = [a for a in algo_ids if a != "scheduling_rl"]

    runs = [
        evaluate_case(case, algo, agent=rl_agent if algo == "scheduling_rl" else None)
        for case in CASES
        for algo in algo_ids
    ]

    summary = {
        algo: {
            "passed": sum(1 for r in runs if r["algorithm"] == algo and r["passed"]),
            "total": sum(1 for r in runs if r["algorithm"] == algo),
        }
        for algo in algo_ids
    }
    return {"cases": [c.id for c in CASES], "algorithms": algo_ids, "runs": runs, "summary": summary}


def _print_report(report: dict) -> None:
    print("=" * 68)
    print("OPTIMAL BENCH — 증명된 최적해 대비 알고리즘 채점")
    print("=" * 68)
    for r in report["runs"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(
            f"[{mark}] {r['case']:<28}{r['algorithm']:<16}"
            f"prod {r['actual']['production']}/{r['target']['production']}  "
            f"conv {r['actual']['conversions']}/{r['target']['conversions']}"
        )
    print("-" * 68)
    for algo, s in report["summary"].items():
        print(f"{algo:<16} {s['passed']}/{s['total']} passed")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algo", action="append", dest="algorithms", metavar="ALGORITHM",
        help="채점할 알고리즘 id (반복 지정 가능, 기본: 등록된 전체 알고리즘)",
    )
    args = parser.parse_args()

    report = run_optimal_benchmark(algorithms=args.algorithms)
    _print_report(report)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n저장 → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
