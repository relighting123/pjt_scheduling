"""
여러 학습 실행 결과 중 최종 모델을 고른다.
=============================================
`train_bench_suite.py`를 설정·시드를 바꿔 여러 번 돌린 뒤, 각 실행의 모델을
BENCH_SUITE(목표)와 HOLDOUT_SUITE(일반화)로 다시 채점하고 하나를 고른다.

선택 규칙 (기본): **BENCH 점수 최고**, 동점이면 HOLDOUT 점수 최고.
`--require-holdout` 을 주면 HOLDOUT에서 dedication 기준선에 못 미치는
모델은 후보에서 제외한다 — 벤치마크만 외운 모델을 배포하지 않기 위해서다.

실행
  python benchmark/select_best_model.py /tmp/m_full_s0 /tmp/m_bench_s0 --install
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG                                    # noqa: E402
from benchmark.compare_vs_dedication import run_suite        # noqa: E402

BASELINE = "dedication"
RL = "scheduling_rl"


def candidate_zips(run_dir: Path) -> list[Path]:
    """실행 디렉터리에서 평가할 모델 zip 후보 (best 우선).

    restore_best=True로 학습하면 `{model_name}.zip`은 `best/best_model.zip`을
    복원해 저장한 것이라 내용이 같다 — 같은 모델을 두 번 채점하지 않도록
    해시로 중복을 거른다.
    """
    out: list[Path] = []
    seen: set[str] = set()
    for rel in ("best/best_model.zip", f"{CONFIG.rl.model_name}.zip"):
        path = run_dir / rel
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(path)
    return out


def score_model(path: Path, suites: list[str]) -> dict:
    result: dict = {"model": str(path)}
    for suite in suites:
        out = run_suite(use_rl=True, model_path=str(path), suite=suite, quiet=True)
        rl = out["summary"].get(RL)
        base = out["summary"].get(BASELINE)
        if rl is None or base is None:
            result[suite] = None
            continue
        result[suite] = {
            "score": rl["score"], "prod": rl["prod"], "max": rl["max"],
            "conv": rl["conv"],
            "baseline_score": base["score"],
            "delta": round(rl["score"] - base["score"], 3),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="학습 실행별 모델 디렉터리")
    ap.add_argument("--install", action="store_true",
                    help="선택된 모델을 CONFIG.path.model_dir 로 복사")
    ap.add_argument("--require-holdout", action="store_true",
                    help="HOLDOUT에서 dedication 미만인 모델은 후보에서 제외")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    suites = ["bench", "holdout"]
    rows = []
    for d in args.run_dirs:
        run_dir = Path(d)
        for zip_path in candidate_zips(run_dir):
            row = score_model(zip_path, suites)
            row["run"] = run_dir.name
            rows.append(row)
            b, h = row.get("bench"), row.get("holdout")
            print(
                f"{run_dir.name:<14} {zip_path.name:<16} "
                f"bench {b['score']:>6.1f} ({b['delta']:+.1f})  "
                f"holdout {h['score']:>6.1f} ({h['delta']:+.1f})"
                if b and h else f"{run_dir.name}: 평가 실패",
                flush=True,
            )

    eligible = [r for r in rows if r.get("bench")]
    if args.require_holdout:
        kept = [r for r in eligible if r.get("holdout") and r["holdout"]["delta"] >= 0]
        if kept:
            eligible = kept
        else:
            print("\n[경고] HOLDOUT에서 dedication을 넘는 모델이 없어 조건을 완화합니다.")

    if not eligible:
        raise SystemExit("평가 가능한 모델이 없습니다.")

    best = max(
        eligible,
        key=lambda r: (r["bench"]["score"],
                       r["holdout"]["score"] if r.get("holdout") else -1e9),
    )
    print(f"\n선택 → {best['model']}")
    print(f"  BENCH   점수 {best['bench']['score']:.1f} "
          f"(dedication {best['bench']['baseline_score']:.1f}, {best['bench']['delta']:+.1f})")
    if best.get("holdout"):
        print(f"  HOLDOUT 점수 {best['holdout']['score']:.1f} "
              f"(dedication {best['holdout']['baseline_score']:.1f}, "
              f"{best['holdout']['delta']:+.1f})")

    if args.install:
        dest_dir = CONFIG.path.model_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{CONFIG.rl.model_name}.zip"
        shutil.copy2(best["model"], dest)
        best_dir = dest_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["model"], best_dir / "best_model.zip")
        print(f"  설치 → {dest}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"candidates": rows, "selected": best}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  저장 → {args.json}")


if __name__ == "__main__":
    main()
