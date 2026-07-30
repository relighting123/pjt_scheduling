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
    ap.add_argument("--bc-episodes", type=int, default=None,
                    help="데이터셋당 전문가 시연 에피소드 수 (풀이 크면 낮춘다)")
    ap.add_argument("--pool-envs", type=int, default=None)
    ap.add_argument("--sampling", default=None, choices=["per_env", "random_reset"])
    ap.add_argument("--model-dir", default=None,
                    help="모델 저장 경로(병렬 실험 시 실행별로 분리)")
    ap.add_argument("--experts", default=None,
                    help="BC 시연자 후보 (쉼표 구분, 예: dedication,earliest_st)")
    ap.add_argument("--anchor-coef", type=float, default=None)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--terminal-throughput", type=float, default=None)
    ap.add_argument("--reward", action="append", default=[], metavar="KEY=VALUE",
                    help="RewardConfig 항목 덮어쓰기 (여러 번 지정 가능). "
                         "예: --reward w_redundant_cover=0 --reward w_bulk_block_bonus=0")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument(
        "--train-pool", action="store_true",
        help="BENCH_SUITE 8종에 더해 TRAIN_POOL(도메인 랜덤화 시나리오)로 학습. "
             "KPI 평가·best 선택은 BENCH_SUITE 8종 기준을 유지한다.",
    )
    ap.add_argument("--holdout", action="store_true",
                    help="학습 후 HOLDOUT_SUITE(일반화 검증)도 함께 평가")
    ap.add_argument("--bench-weight", type=int, default=1,
                    help="--train-pool 사용 시 BENCH 8종을 풀에 몇 배로 넣을지. "
                         "1이면 풀 전체에서 BENCH 비중이 1/7로 낮아 벤치 점수가 "
                         "느리게 오른다. 높이면 벤치 쪽으로 치우친다(일반화와 trade-off).")
    args = ap.parse_args()

    bench_meta = load_meta("bench")
    bench_eds = [load_ed(m) for m in bench_meta]

    if args.train_pool:
        pool_meta = load_meta("train_pool")
        weight = max(args.bench_weight, 1)
        eds = bench_eds * weight + [load_ed(m) for m in pool_meta]
        print(f"학습 시나리오: BENCH {len(bench_eds)}종 ×{weight} + TRAIN_POOL "
              f"{len(pool_meta)}종 = {len(eds)}종", flush=True)
    else:
        eds = bench_eds

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
    if args.bc_episodes is not None:
        cfg.bc_episodes_per_dataset = args.bc_episodes
    if args.pool_envs is not None:
        cfg.dataset_pool_envs = args.pool_envs
    if args.sampling is not None:
        cfg.dataset_sampling = args.sampling
    if args.experts is not None:
        cfg.bc_expert_candidates = tuple(
            n.strip() for n in args.experts.split(",") if n.strip()
        )
    if args.model_dir is not None:
        CONFIG.path.model_dir = Path(args.model_dir)
        CONFIG.path.model_dir.mkdir(parents=True, exist_ok=True)
    if args.anchor_coef is not None:
        cfg.expert_anchor_coef = args.anchor_coef
    if args.ent_coef is not None:
        cfg.ent_coef = args.ent_coef
    if args.terminal_throughput is not None:
        CONFIG.reward.w_terminal_throughput = args.terminal_throughput
    for item in args.reward:
        key, _, raw = item.partition("=")
        key = key.strip()
        if not hasattr(CONFIG.reward, key):
            raise SystemExit(f"알 수 없는 RewardConfig 항목: {key}")
        current = getattr(CONFIG.reward, key)
        setattr(CONFIG.reward, key,
                (raw.strip().lower() in ("1", "true", "yes"))
                if isinstance(current, bool) else float(raw))
        print(f"  reward override: {key} = {getattr(CONFIG.reward, key)}", flush=True)

    from agent.rl_agent import SchedulingAgent
    from env.scheduling_rl_env import SchedulingRLEnv

    print(f"=== BENCH_SUITE co-train (datasets={len(eds)}, TS={args.timesteps:,}, "
          f"seed={args.seed}) ===", flush=True)
    t0 = time.time()
    agent = SchedulingAgent()
    agent.train(eds, verbose=1, env_cls=SchedulingRLEnv, eval_datasets=bench_eds)
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
        out["benchmark"] = run_suite(use_rl=True, conv_weight=1.0, suite="bench")
        if args.holdout:
            print("\n" + "#" * 72)
            print("# HOLDOUT_SUITE (학습에 쓰지 않은 시나리오 — 일반화 검증)")
            print("#" * 72)
            out["holdout"] = run_suite(use_rl=True, conv_weight=1.0, suite="holdout")

    path = Path(__file__).parent.parent / "data/dataset" / f"train_bench_{args.tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장 → {path}")


if __name__ == "__main__":
    main()
