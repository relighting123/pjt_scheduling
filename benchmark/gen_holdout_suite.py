"""
HOLDOUT_SUITE 생성기 — 학습에 쓰지 않은 시나리오로 일반화 성능 확인
====================================================================
`gen_bench_suite.py`가 만드는 BENCH_SUITE 8종은 co-train(같은 데이터로 학습 후
같은 데이터로 평가)에 쓰이므로, 그 점수만으로는 "정책을 배운 것"과 "그 8개를
외운 것"을 구분할 수 없다. 이 스크립트는 같은 4개 카테고리를 유지하면서
설비 수·제품 수·캐리어 수·ST·전환시간을 바꾼 **학습에 쓰지 않는** 시나리오를
만든다.

실행
  python benchmark/gen_holdout_suite.py
  python benchmark/compare_vs_dedication.py --suite holdout
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import gen_bench_suite as base  # noqa: E402

# BENCH_SUITE와 같은 카테고리 구성이되, 규모·파라미터가 전부 다른 시나리오.
SPECS = [
    dict(id="H_SYM_4x4", cat="대칭 기준", n_eqp=4, n_ppk=4, carriers=7, st=50, conv=50,
         mix=0.0, tool=99, desc="4설비·4제품·각 7캐리어 대칭.",
         tests="[holdout] 전담 최적해 일반화"),
    dict(id="H_SYM_6x6", cat="대칭 기준", n_eqp=6, n_ppk=6, carriers=5, st=42, conv=42,
         mix=0.0, tool=99, desc="6설비·6제품·각 5캐리어. BENCH보다 큰 대칭.",
         tests="[holdout] 규모 확장 일반화"),
    dict(id="H_OVER_6p4", cat="제품 과잉", n_eqp=4, n_ppk=6, carriers=5, st=55, conv=55,
         mix=0.9, tool=99, desc="4설비가 6제품 → 최소 2회 전환.",
         tests="[holdout] 불가피 전환 최소화"),
    dict(id="H_LOAD_skew2", cat="부하 불균등", n_eqp=5, n_ppk=5,
         carriers=[16, 10, 6, 4, 4], st=55, conv=55, mix=0.9, tool=99,
         desc="제품별 계획량 16·10·6·4·4로 편중.",
         tests="[holdout] 물량 편중 일반화"),
    dict(id="H_LOAD_stmix2", cat="부하 불균등", n_eqp=3, n_ppk=3, carriers=9,
         st=[25, 50, 75], conv=50, mix=0.9, tool=99,
         desc="제품별 처리시간 25·50·75로 이질.",
         tests="[holdout] 처리시간 이질성 일반화"),
    dict(id="H_CONV_x25", cat="전환 과중", n_eqp=5, n_ppk=5, carriers=5, st=40, conv=100,
         mix=0.9, tool=99, desc="전환시간(100)=처리시간(40)의 2.5배.",
         tests="[holdout] 큰 전환비용 일반화"),
]


def main() -> None:
    meta = []
    for spec in SPECS:
        m = base.gen_one(spec)
        meta.append(m)
        print(f"  {m['id']:<14} [{m['cat']:<6}] {m['n_eqp']}설비×{m['n_ppk']}제품 "
              f"총{m['total']}캐리어 sim={m['sim']} 최소전환={m['min_conv']}")
    import json
    out = base.SUITE_ROOT / "holdout_suite_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nHOLDOUT_SUITE {len(meta)}종 생성 완료 → {out}")


if __name__ == "__main__":
    main()
