#!/usr/bin/env python3
"""
scripts/run_takt_suite.py – Takt 시나리오 4종: 생성 → 학습 → 추론 → pacing 검증

사용:
    python scripts/run_takt_suite.py
    python scripts/run_takt_suite.py --timesteps 80000 --skip-train  # 추론만
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder
from data.loader import load_data, validate_data
from data.preprocessor import normalize_raw, preprocess
from data.pacing_scenarios import TAKT_SUITE_LAYOUT, bootstrap_takt_suite
from agent.rl_agent import SchedulingAgent
from inference.runner import run_inference
from validation.pacing_metrics import pacing_metrics

ALGOS = ("minprogress", "earliest_st", "rl")


def load_env_for_folder(folder: str) -> dict:
    set_input_folder(folder)
    raw = normalize_raw(load_data())
    errors = validate_data(raw)
    if errors:
        raise RuntimeError(f"{folder}: {errors}")
    return preprocess(raw)


def run_suite(timesteps: int, skip_train: bool, fac_id: str) -> dict:
    print("=" * 60)
    print("[1] Takt 시나리오 JSON 생성 (train 3 + test 1)")
    info = bootstrap_takt_suite(fac_id=fac_id)
    for p in info["paths"]:
        print(f"  · {p['folder']} ({p['scenario']})")

    train_folders = [p["folder"] for p in info["paths"] if p["split"] == "train"]
    all_folders = [p["folder"] for p in info["paths"]]

    print("=" * 60)
    print("[2] 학습 데이터 로드")
    train_data = [load_env_for_folder(f) for f in train_folders]
    print(f"  train 기간: {len(train_data)}개")

    if not skip_train:
        print("=" * 60)
        print(f"[3] RL 학습 (timesteps={timesteps:,})")
        CONFIG.rl.total_timesteps = timesteps
        agent = SchedulingAgent()
        agent.train(train_data, verbose=1)
        agent.save()
        eval_m = agent.evaluate(train_data[0], n_episodes=2)
        print(f"  eval mean_reward={eval_m['mean_reward']:.1f}")
    else:
        print("=" * 60)
        print("[3] 학습 스킵 (--skip-train)")
        agent = SchedulingAgent()
        if not agent.model_exists():
            raise RuntimeError("저장된 RL 모델 없음. --skip-train 제거 후 재실행.")
        agent = SchedulingAgent.load()

    print("=" * 60)
    print("[4] 시나리오별 추론 + pacing 메트릭")
    report = {"fac_id": fac_id, "timesteps": timesteps, "scenarios": []}

    for entry in info["paths"]:
        folder = entry["folder"]
        scenario = entry["scenario"]
        env_data = load_env_for_folder(folder)
        horizon = env_data.get("soft_cutoff_minutes", 1320)
        scen_out = {"scenario": scenario, "folder": folder, "algorithms": []}

        for algo in ALGOS:
            rl_agent = agent if algo == "rl" else None
            result = run_inference(env_data, algorithm=algo, agent=rl_agent, record_history=False)
            metrics = pacing_metrics(result["schedule"], env_data["plan"], horizon=horizon)
            row = {
                "algorithm": algo,
                "mae": round(metrics["mae"], 2),
                "max_dev": round(metrics["max_dev"], 2),
                "final_gap": round(metrics["final_gap"], 3),
                "n_assignments": len(result["schedule"]),
                "idle_total": result["stats"]["idle_total"],
                "by_key": metrics["by_key"],
            }
            scen_out["algorithms"].append(row)
            print(
                f"  {scenario:16} {algo:14} "
                f"MAE={row['mae']:6.1f}  max_dev={row['max_dev']:6.1f}  "
                f"달성={row['final_gap']:.2%}  assigns={row['n_assignments']}"
            )

        best = min(scen_out["algorithms"], key=lambda x: x["mae"])
        scen_out["best_algorithm"] = best["algorithm"]
        scen_out["best_mae"] = best["mae"]
        report["scenarios"].append(scen_out)

    print("=" * 60)
    print("[5] 요약 / 이상 여부")
    issues = []
    for scen in report["scenarios"]:
        sid = scen["scenario"]
        for row in scen["algorithms"]:
            if row["n_assignments"] == 0:
                issues.append(f"{sid}/{row['algorithm']}: 배정 0건")
            if row["final_gap"] < 0.5 and sid != "takt_2ppk":
                issues.append(
                    f"{sid}/{row['algorithm']}: 달성률 {row['final_gap']:.0%} (<50%, takt_2ppk 제외)"
                )
        if sid == "takt_1p1o":
            if scen["best_mae"] > 25:
                issues.append(f"takt_1p1o: 최선 MAE {scen['best_mae']:.1f} > 25 (단순 케이스 이상)")

    rl_wins = sum(
        1 for s in report["scenarios"]
        if s["best_algorithm"] == "rl"
    )
    print(f"  RL이 최저 MAE인 시나리오: {rl_wins}/{len(report['scenarios'])}")
    if issues:
        print("  [주의]")
        for msg in issues:
            print(f"    - {msg}")
        report["status"] = "warning"
    else:
        print("  [OK] 기본 sanity check 통과")
        report["status"] = "ok"
    report["issues"] = issues

    out_path = ROOT / "external" / "dataset" / fac_id / "takt_suite_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  리포트: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Takt pacing 시나리오 전체 파이프라인")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--fac-id", default="FAC_TAKT")
    args = parser.parse_args()
    run_suite(args.timesteps, args.skip_train, args.fac_id)


if __name__ == "__main__":
    main()
