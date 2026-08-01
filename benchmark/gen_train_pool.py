"""
TRAIN_POOL 생성기 — 도메인 랜덤화용 학습 시나리오 풀
=====================================================
BENCH_SUITE 8종만으로 co-train하면 정책이 "전담이라는 규칙"이 아니라 "그 8개
시나리오의 정답 순열"을 외운다 (실측: 학습에 쓴 8종에선 dedication +5점,
학습에 안 쓴 HOLDOUT 6종에선 −7점).

이 스크립트는 같은 생성기(`gen_bench_suite.gen_one`)로 설비 수·제품 수·
캐리어 수·ST·전환시간·분산도를 무작위로 뽑은 시나리오를 대량 생성한다.
`RandomizedSchedulingRLEnv`가 에피소드마다 이 풀에서 하나를 골라 돌면
정책이 시나리오를 외울 수 없어 일반화된다.

주의: 평가용 BENCH_SUITE/HOLDOUT_SUITE와 **id 접두사가 달라** 폴더가 겹치지
않는다. 시드를 고정하므로 재생성해도 같은 풀이 나온다.

실행
  python benchmark/gen_train_pool.py --count 48 --seed 7
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import gen_bench_suite as base  # noqa: E402
from benchmark import gen_tool_change_bench as multistage  # noqa: E402

CATEGORIES = ["대칭 기준", "제품 과잉", "부하 불균등", "전환 과중"]


def sample_spec(rng: random.Random, idx: int) -> dict:
    """4개 카테고리 중 하나를 뽑아 파라미터를 무작위화한 스펙."""
    cat = CATEGORIES[idx % len(CATEGORIES)]
    n_eqp = rng.randint(3, 6)
    st_base = rng.choice([30, 40, 45, 50, 60, 75])

    if cat == "대칭 기준":
        # 설비=제품, 전담하면 전환 0이 최적. mix=0 고정 —
        # gen_bench_suite 주석 참고(mix>0이면 전담 자체가 불가능해진다).
        n_ppk = n_eqp
        carriers = rng.randint(4, 9)
        st = st_base
        conv = st_base
        mix = 0.0
    elif cat == "제품 과잉":
        n_ppk = n_eqp + rng.randint(1, 3)
        carriers = rng.randint(3, 6)
        st = st_base
        conv = st_base
        mix = 0.9
    elif cat == "부하 불균등":
        n_ppk = n_eqp
        if rng.random() < 0.5:                      # 물량 편중
            carriers = sorted(
                (rng.randint(3, 16) for _ in range(n_ppk)), reverse=True,
            )
            st = st_base
        else:                                       # 처리시간 이질
            carriers = rng.randint(6, 10)
            st = [rng.choice([25, 30, 45, 60, 75, 90]) for _ in range(n_ppk)]
        conv = st_base
        mix = 0.9
    else:                                           # 전환 과중
        n_ppk = n_eqp + rng.randint(0, 2)
        carriers = rng.randint(4, 7)
        st = st_base
        conv = int(st_base * rng.choice([2.0, 2.5, 3.0]))
        mix = 0.9

    return dict(
        id=f"TP_{idx:03d}", cat=cat, n_eqp=n_eqp, n_ppk=n_ppk,
        carriers=carriers, st=st, conv=conv, mix=mix, tool=99,
        desc=f"[train pool] {cat} {n_eqp}설비×{n_ppk}제품",
        tests="[train pool] 도메인 랜덤화 학습용",
    )


def sample_multistage_spec(rng: random.Random, idx: int) -> dict:
    """평가용 Tool-Change와 겹치지 않는 2공정 랜덤 학습 스펙."""
    n_ppk = rng.randint(2, 6)
    n_eqp_1 = rng.randint(2, 4)
    n_eqp_2 = rng.randint(2, 4)
    conv = rng.choice([30, 45, 60, 90, 120])
    ppks = [f"PPK{i + 1:03d}" for i in range(n_ppk)]
    carriers = [rng.randint(3, 9) for _ in ppks]
    st_1 = [rng.choice([20, 30, 40, 45, 60, 75]) for _ in ppks]
    st_2 = [rng.choice([20, 30, 40, 45, 60, 75, 90]) for _ in ppks]

    def products(st_values: list[int]) -> list[dict]:
        return [
            {"ppk": ppk, "carriers": qty, "st": st}
            for ppk, qty, st in zip(ppks, carriers, st_values)
        ]

    return {
        "id": f"TP_MS_{idx:03d}",
        "n_ppk": n_ppk,
        "carriers": carriers,
        "conv": conv,
        "stage1": {
            "oper": "OPER001", "n_eqp": n_eqp_1,
            "ppks": products(st_1), "conv": conv, "scramble": True,
        },
        "stage2": {
            "oper": "OPER002", "n_eqp": n_eqp_2,
            "ppks": products(st_2), "conv": conv, "scramble": True,
        },
    }


def generate_multistage(spec: dict) -> dict:
    """2공정 장비공유 데이터를 생성하고 공통 suite 메타 형식으로 반환."""
    multistage.gen_multi_stage(spec["id"], spec["stage1"], spec["stage2"])
    stage_horizons = []
    for stage in (spec["stage1"], spec["stage2"]):
        work = sum(p["carriers"] * p["st"] for p in stage["ppks"])
        unavoidable = max(spec["n_ppk"] - stage["n_eqp"], 0)
        stage_horizons.append(math.ceil(work / stage["n_eqp"]) + unavoidable * spec["conv"])
    sim = int(math.ceil(max(stage_horizons) / 10.0) * 10)
    out = multistage.SUITE_ROOT / spec["id"] / "train" / multistage.TIMEKEY / "input"
    return {
        "id": spec["id"],
        "cat": "다공정 랜덤",
        "n_eqp": spec["stage1"]["n_eqp"] + spec["stage2"]["n_eqp"],
        "n_ppk": spec["n_ppk"],
        "carriers": spec["carriers"],
        "st": [
            [p["st"] for p in spec["stage1"]["ppks"]],
            [p["st"] for p in spec["stage2"]["ppks"]],
        ],
        "conv": spec["conv"],
        "sim": sim,
        "total": 2 * sum(spec["carriers"]),
        "min_conv": sum(
            max(spec["n_ppk"] - stage["n_eqp"], 0)
            for stage in (spec["stage1"], spec["stage2"])
        ),
        "mix": 1.0,
        "tool": multistage.TOOL_MAX,
        "desc": (
            f"[train pool] 2공정 랜덤 장비공유 "
            f"{spec['stage1']['n_eqp']}+{spec['stage2']['n_eqp']}설비×{spec['n_ppk']}제품"
        ),
        "tests": "[train pool] 미학습 다공정 일반화",
        "dir": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument(
        "--multi-stage-count", type=int, default=12,
        help="별도로 추가할 2공정 랜덤 장비공유 시나리오 수",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta = []
    for i in range(args.count):
        m = base.gen_one(sample_spec(rng, i))
        meta.append(m)
    for i in range(max(args.multi_stage_count, 0)):
        meta.append(generate_multistage(sample_multistage_spec(rng, i)))

    out = base.SUITE_ROOT / "train_pool_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    by_cat: dict = {}
    for m in meta:
        by_cat[m["cat"]] = by_cat.get(m["cat"], 0) + 1
    print(f"TRAIN_POOL {len(meta)}종 생성 완료 → {out}")
    for cat, n in by_cat.items():
        print(f"  {cat:<10} {n}종")


if __name__ == "__main__":
    main()
