"""
CONV_BENCH 평가 스크립트
=========================================================
CONV_BENCH 데이터셋(data/dataset/CONV_BENCH/)을 기반으로
알고리즘별 성능을 비교한다.

평가 지표
  prod  : 총 생산 수량 (최대 24)
  conv  : 전환 횟수   (0이 최적)
  loss  : 전환으로 인한 수량 손실 = 전환 1회당 -1개 (ST=Conv=60분이므로)
  전담  : 1개 PPK가 80%↑인 EQP 비율
  CV    : 장비간 부하 편차 (낮을수록 균등)

이론 최적
  prod=24, conv=0, loss=0, 전담=3/3, CV=0.000

실행 방법
  python benchmark/conv_bench.py           # RL 추론 없이 룰 기반만 비교
  TS=200000 python benchmark/conv_bench.py  # scheduling_rl 학습 후 비교
"""
import os
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import gen_conv_bench  # 데이터셋 자동 생성 (없으면 생성)

from pathlib import Path
from collections import defaultdict
import numpy as np

from config import CONFIG
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from inference.runner import run_inference

DATASET_DIR = Path(__file__).parent.parent / "data/dataset/CONV_BENCH/train/20260629000000"
CONV_MINUTES = 60  # config 기본값과 동일
ST_MINUTES   = 60
SIM_MINUTES  = 480
MAX_PROD     = 24

def load_ed() -> dict:
    ed = preprocess(load_data(DATASET_DIR / "input"))
    ed["eqp_selection"]   = "order"
    ed["sim_end_minutes"] = SIM_MINUTES
    ed["conversion_minutes"] = CONV_MINUTES
    return ed

def summarize(result: dict, label: str) -> dict:
    stats = result["stats"]
    sched = result["schedule"]

    # 시뮬 시간 내 완료된 carrier만 집계 (END_TM <= SIM_MINUTES)
    in_time = [r for r in sched if r.get("END_TM", 0) <= SIM_MINUTES]
    by_eqp: dict = defaultdict(lambda: defaultdict(int))
    for r in in_time:
        by_eqp[r["EQP_ID"]][r.get("PLAN_PROD_ATTR_VAL", "?")] += 1

    eqp_ids  = sorted(by_eqp.keys()) or sorted({r["EQP_ID"] for r in sched})
    carriers = [sum(by_eqp[e].values()) for e in eqp_ids]
    cv  = (np.std(carriers) / np.mean(carriers)) if carriers and np.mean(carriers) > 0 else 0
    ded = sum(1 for e in eqp_ids
              if by_eqp[e] and
              max(by_eqp[e].values()) / max(sum(by_eqp[e].values()), 1) >= 0.8)
    prod = sum(carriers)
    conv = stats.get("conversions", 0)
    loss = MAX_PROD - prod  # 실제 손실(이론 최대 대비)

    print(f"  {label:<18} "
          f"prod={prod:>2d}/{MAX_PROD} "
          f"conv={conv:>2d} "
          f"loss={loss:>2d} "
          f"전담={ded}/{len(eqp_ids)} "
          f"CV={cv:.3f}")
    for e in eqp_ids:
        ppk_dist = dict(sorted(by_eqp[e].items()))
        total    = sum(ppk_dist.values())
        pct      = {k: f"{v}({v/total:.0%})" for k, v in ppk_dist.items()}
        print(f"      {e}: {pct}")
    return {"prod": prod, "conv": conv, "loss": loss, "ded": ded,
            "n_eqp": len(eqp_ids), "cv": cv}

def verdict(results: dict) -> None:
    print("\n" + "="*60)
    print("최적해 판정 (이론: prod=24, conv=0, loss=0, 전담=3/3)")
    print("="*60)
    best = min(results.values(), key=lambda r: (r["conv"], -r["prod"]))
    for label, r in results.items():
        ok_prod = "✓" if r["prod"] == MAX_PROD else f"✗({MAX_PROD-r['prod']}개 손실)"
        ok_conv = "✓" if r["conv"] == 0        else f"✗(전환{r['conv']}회→{r['loss']}개 손실)"
        ok_ded  = "✓" if r["ded"] == r["n_eqp"] else f"△({r['ded']}/{r['n_eqp']})"
        print(f"  {label:<18} 생산:{ok_prod}  전환:{ok_conv}  전담:{ok_ded}")

def main() -> None:
    print("="*60)
    print("CONV_BENCH: 전환 비용 최소화 벤치마크")
    print("  ST=60분, 전환=60분, 시뮬=8h → 전환 1회=캐리어 1개 손실")
    print("  3EQP × 3PPK(각 다른 LOT_CD) × 8carrier = 24 total")
    print("="*60 + "\n")

    ed = load_ed()
    print(f"로드: PPK={ed['prod_keys']}  EQP={ed['eqp_ids']}")
    print(f"      lots={len(ed['lots'])}  sim={SIM_MINUTES}분\n")

    results: dict = {}

    TS = int(os.environ.get("TS", "0"))
    if TS > 0:
        from agent.rl_agent import SchedulingAgent
        from env.scheduling_rl_env import SchedulingRLEnv
        cfg = CONFIG.reward
        cfg.w_bulk_block_bonus  =  3.0
        cfg.w_dedication_misuse = -4.0
        cfg.w_redundant_cover   = -5.0
        cfg.w_plan_hit          =  1.0
        CONFIG.rl.total_timesteps = TS
        CONFIG.rl.n_steps = 2048
        CONFIG.rl.device  = "cpu"
        CONFIG.rl.n_envs  = 1
        print(f"=== scheduling_rl 학습 (TS={TS:,}) ===")
        agent = SchedulingAgent()
        agent.train([ed], verbose=0, env_cls=SchedulingRLEnv)
        agent.save()
        print("완료\n")

    print("=== 알고리즘 비교 ===\n")
    results["minprogress"] = summarize(run_inference(ed, algorithm="minprogress"), "minprogress")
    results["earliest_st"] = summarize(run_inference(ed, algorithm="earliest_st"), "earliest_st")

    if TS > 0:
        results[f"scheduling_rl({TS//1000}k)"] = summarize(
            run_inference(ed, algorithm="scheduling_rl", agent=agent),
            f"scheduling_rl({TS//1000}k)",
        )

    verdict(results)

if __name__ == "__main__":
    main()
