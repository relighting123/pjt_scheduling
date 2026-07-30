"""
BENCH_SUITE 8종 co-train → dedication 기준 벤치마크
====================================================
`benchmark/bench_suite.py`의 `TS=... ` 학습 경로를 독립 스크립트로 뽑아낸 것.
학습이 끝나면 `benchmark/compare_vs_dedication.py`로 곧바로 채점한다.

실행
  python benchmark/train_bench_suite.py --timesteps 400000
  python benchmark/train_bench_suite.py --timesteps 200000 --tag exp1 --seed 1

산출
  models/scheduling_rl.zip            학습 모델
  data/dataset/train_bench_<tag>.json 학습 KPI 히스토리 + 최종 벤치마크
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG                                        # noqa: E402
from benchmark.compare_vs_dedication import (                    # noqa: E402
    load_meta, load_ed, run_suite,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bc-epochs", type=int, default=None)
    ap.add_argument("--anchor-coef", type=float, default=None)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--terminal-throughput", type=float, default=None)
    ap.add_argument("--no-benchmark", action="store_true")
    args = ap.parse_args()

    meta = load_meta()
    eds = [load_ed(m) for m in meta]

    cfg = CONFIG.rl
    cfg.total_timesteps = args.timesteps
    cfg.eval_freq = args.eval_freq
    cfg.device = args.device
    cfg.n_envs = 1
    cfg.seed = args.seed
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.bc_epochs is not None:
        cfg.bc_pretrain_epochs = args.bc_epochs
    if args.anchor_coef is not None:
        cfg.expert_anchor_coef = args.anchor_coef
    if args.ent_coef is not None:
        cfg.ent_coef = args.ent_coef
    if args.terminal_throughput is not None:
        CONFIG.reward.w_terminal_throughput = args.terminal_throughput

    from agent.rl_agent import SchedulingAgent
    from env.scheduling_rl_env import SchedulingRLEnv

    print(f"=== BENCH_SUITE co-train (datasets={len(eds)}, TS={args.timesteps:,}, "
          f"seed={args.seed}) ===", flush=True)
    t0 = time.time()
    agent = SchedulingAgent()
    agent.train(eds, verbose=1, env_cls=SchedulingRLEnv)
    agent.save()
    elapsed = time.time() - t0
    print(f"학습 완료 ({elapsed / 60:.1f}분)\n", flush=True)

    out = {
        "tag": args.tag,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "elapsed_sec": round(elapsed, 1),
        "kpi_history": getattr(agent, "kpi_history", []),
    }
    if not args.no_benchmark:
        out["benchmark"] = run_suite(use_rl=True, conv_weight=1.0)

    path = Path(__file__).parent.parent / "data/dataset" / f"train_bench_{args.tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장 → {path}")


if __name__ == "__main__":
    main()
