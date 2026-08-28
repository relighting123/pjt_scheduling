"""
결과 일관성(Consistency) 계측
=============================
5분 주기 재실행 운영에서 "상황이 조금만 달라져도 스케줄이 크게 뒤바뀌는" 정도를
수치화한다. `compare_vs_dedication.py`가 품질(생산/전환)을 재는 것과 달리, 이
스크립트는 **입력 미세 변화에 대한 출력의 안정성**을 잰다.

방법
  각 데이터셋마다 baseline 을 한 번 돌린 뒤, 아래 perturbation 을 하나씩 적용해
  다시 돌리고 baseline 스케줄과 비교한다.
    - `wip-1:<PPK>/<OPER>` : 해당 버킷의 재공 1건 감소 (5분 사이 1장 빠진 상황)
    - `horizon-N`          : sim_end/soft_cutoff 를 N분 단축 (N분 경과한 상황)
    - `eqp-down:<EQP>`     : 설비 1대 제외 (축 목록이 바뀌는 상황 — 남은 설비의
                             배정이 유지되는지가 축 고정(axis_map)의 효과를 가른다)

지표 (높을수록 안정적)
  seq_agree   : EQP별 (PPK/OPER) 배정 순서를 앞에서부터 대조한 일치율  ← 주 지표
  first_agree : 각 EQP의 첫 배정이 baseline과 같은 비율
  conv_range  : 전환 횟수의 (min, max). baseline보다 작아졌다 커졌다 하면 비단조.

실행
  python benchmark/consistency_bench.py                          # 저장된 RL 모델
  python benchmark/consistency_bench.py --model PATH --json OUT.json
  python benchmark/consistency_bench.py --algorithm dedication   # 휴리스틱 기준선
  python benchmark/consistency_bench.py --suite holdout --limit 3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG                                    # noqa: E402
from data.loader.axis_map import save_axis_map                # noqa: E402
from inference.runner import run_inference                   # noqa: E402
from data.loader.fetch import load_data                       # noqa: E402
from data.loader.preprocess import preprocess                 # noqa: E402
from benchmark.compare_vs_dedication import load_meta, load_ed   # noqa: E402

ROOT = Path(__file__).parent.parent

DEFAULT_HORIZON_SHIFTS = (5, 10)


# ── perturbation ────────────────────────────────────────────────────────────

def _wip_buckets(ed: dict) -> list:
    """재공을 1건 뺄 수 있는 (ppk, oper) 버킷 목록 (많은 순)."""
    pool = ed.get("abstract_wip_init") or {}
    items = [
        (key, int(v.get("wip_qty", 0)))
        for key, v in pool.items()
        if isinstance(v, dict) and int(v.get("wip_qty", 0)) > 1
    ]
    return [k for k, _ in sorted(items, key=lambda kv: (-kv[1], str(kv[0])))]


def perturb_drop_wip(ed: dict, key) -> dict:
    """지정 버킷에서 재공(carrier) 1건 제거. wip_qty/lot_ids/lots/meta 를 함께 정리."""
    p = copy.deepcopy(ed)
    bucket = p["abstract_wip_init"][key]
    lot_ids = list(bucket.get("lot_ids", []))
    if not lot_ids:
        return p
    victim = lot_ids[-1]          # 마지막 1건 — 결정적으로 고르기 위함
    bucket["lot_ids"] = lot_ids[:-1]
    bucket["wip_qty"] = max(int(bucket.get("wip_qty", 1)) - 1, 0)
    if "wip_qty_init" in bucket:
        bucket["wip_qty_init"] = bucket["wip_qty"]

    if isinstance(p.get("lots"), list):
        p["lots"] = [l for l in p["lots"] if l.get("lot_id") != victim]
    for k in ("abstract_lot_meta", "lot_attrs"):
        if isinstance(p.get(k), dict):
            p[k] = {kk: vv for kk, vv in p[k].items() if kk != victim}
    return p


def perturb_drop_eqp(m: dict, eqp_id: str) -> dict:
    """설비 1대를 원시 입력에서 빼고 다시 preprocess.

    `eqp_ids` 목록 자체가 바뀌므로 축 인덱스가 재배치될 수 있는 유일한
    perturbation이다. axis_map 고정의 효과가 여기서 드러난다.
    """
    raw = load_data(ROOT / m["dir"])
    for key in ("discrete_arrange", "eqp_initial_state", "eqp_queue_init",
                "eqp_down", "eqp_conv_plan"):
        if isinstance(raw.get(key), list):
            raw[key] = [r for r in raw[key] if r.get("EQP_ID") != eqp_id]
    ed = preprocess(raw)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = m["sim"]
    ed["conversion_minutes"] = m["conv"]
    ed["enable_wip_inflow"] = bool(m.get("enable_wip_inflow", False))
    if m.get("discrete_wait_enabled") is not None:
        ed["discrete_wait_enabled"] = bool(m["discrete_wait_enabled"])
    return ed


def perturb_horizon(ed: dict, minutes: int) -> dict:
    """horizon 을 minutes 만큼 단축 = 그만큼 시간이 지난 상황."""
    p = copy.deepcopy(ed)
    p["sim_end_minutes"] = int(p["sim_end_minutes"]) - minutes
    soft = int(p.get("soft_cutoff_minutes", CONFIG.env.soft_cutoff_minutes))
    p["soft_cutoff_minutes"] = soft - minutes
    return p


# ── 비교 ────────────────────────────────────────────────────────────────────

def _bucket_seq(result: dict) -> dict:
    """EQP별 (PPK/OPER) 배정 순서."""
    seq: dict = {}
    for rec in sorted(result["schedule"], key=lambda r: (r["EQP_ID"], r["START_TM"])):
        seq.setdefault(rec["EQP_ID"], []).append(
            f"{rec.get('PLAN_PROD_ATTR_VAL', '?')}/{rec.get('OPER_ID', '?')}"
        )
    return seq


def compare(base_seq: dict, pert_seq: dict) -> dict:
    """baseline 대비 시퀀스/첫 배정 일치율. 공통 EQP 에 대해서만 센다."""
    common = sorted(set(base_seq) & set(pert_seq))
    total = matched = 0
    first_total = first_matched = 0
    for eqp in common:
        a, b = base_seq[eqp], pert_seq[eqp]
        n = min(len(a), len(b))
        total += n
        matched += sum(1 for i in range(n) if a[i] == b[i])
        if a and b:
            first_total += 1
            first_matched += int(a[0] == b[0])
    return {
        "seq_agree": round(100 * matched / total, 1) if total else None,
        "first_agree": round(100 * first_matched / first_total, 1) if first_total else None,
        "compared_rows": total,
    }


# ── 러너 ────────────────────────────────────────────────────────────────────

def run_dataset(ed: dict, m: dict, *, algorithm: str, agent, max_wip_perturb: int,
                horizon_shifts, eqp_down: bool = True,
                pin_axis: bool = False) -> dict:
    def go(e):
        return run_inference(
            e, algorithm=algorithm,
            agent=agent if algorithm == "scheduling_rl" else None,
            record_history=False,
            enable_wip_inflow=bool(e.get("enable_wip_inflow", False)),
        )

    base = go(ed)
    base_seq = _bucket_seq(base)
    base_conv = base["stats"].get("conversions", 0)

    cases = []
    for key in _wip_buckets(ed)[:max_wip_perturb]:
        cases.append((f"wip-1:{key[0]}/{key[1]}", perturb_drop_wip(ed, key)))
    for mins in horizon_shifts:
        cases.append((f"horizon-{mins}", perturb_horizon(ed, mins)))
    if eqp_down:
        victim = sorted(ed.get("eqp_ids", []))[-1:]        # 결정적으로 마지막 1대
        # pin_axis: 이 데이터셋의 baseline 축 순서를 고정해 두고 perturbation을
        # 돌린다 = 운영에서 "정상 상태에서 축을 고정해 둔 뒤 설비가 빠지는" 상황.
        # 축 맵은 FAC 단위 계약이라 데이터셋마다 따로 고정해야 비교가 성립한다.
        tmpdir = tempfile.mkdtemp(prefix="axis_") if pin_axis else None
        prev_env = os.environ.get("AXIS_MAP_CONFIG")
        try:
            if pin_axis:
                axis_path = Path(tmpdir) / "axis_map.json"
                save_axis_map(ed, timestamp="00000000000000", path=axis_path)
                os.environ["AXIS_MAP_CONFIG"] = str(axis_path)
            for e in victim:
                try:
                    cases.append((f"eqp-down:{e}", perturb_drop_eqp(m, e)))
                except Exception:
                    pass                                    # 설비 1대뿐인 데이터셋 등
        finally:
            if prev_env is None:
                os.environ.pop("AXIS_MAP_CONFIG", None)
            else:
                os.environ["AXIS_MAP_CONFIG"] = prev_env

    results = []
    for label, ped in cases:
        try:
            r = go(ped)
        except Exception as exc:                       # 데이터가 깨지는 조합은 건너뜀
            results.append({"perturbation": label, "error": str(exc)})
            continue
        cmp_ = compare(base_seq, _bucket_seq(r))
        cmp_["perturbation"] = label
        cmp_["conv"] = r["stats"].get("conversions", 0)
        cmp_["conv_delta"] = cmp_["conv"] - base_conv
        results.append(cmp_)

    ok = [r for r in results if r.get("seq_agree") is not None]
    convs = [r["conv"] for r in results if "conv" in r]
    return {
        "id": m.get("id", "?"),
        "base_conv": base_conv,
        "base_rows": len(base["schedule"]),
        "seq_agree_mean": round(sum(r["seq_agree"] for r in ok) / len(ok), 1) if ok else None,
        "seq_agree_min": min((r["seq_agree"] for r in ok), default=None),
        "first_agree_mean": round(
            sum(r["first_agree"] for r in ok if r["first_agree"] is not None)
            / max(sum(1 for r in ok if r["first_agree"] is not None), 1), 1,
        ) if ok else None,
        "conv_range": [min(convs, default=base_conv), max(convs, default=base_conv)],
        # 전환이 baseline 아래위로 모두 튀면 비단조 — 과최적화 징후
        "conv_nonmonotonic": bool(convs) and min(convs) < base_conv < max(convs),
        "cases": results,
    }


def run_consistency(*, suite: str = "bench", algorithm: str = "scheduling_rl",
                    model_path: str | None = None, limit: int | None = None,
                    max_wip_perturb: int = 4,
                    horizon_shifts=DEFAULT_HORIZON_SHIFTS,
                    eqp_down: bool = True,
                    pin_axis: bool = False,
                    quiet: bool = False) -> dict:
    meta = load_meta(suite)
    if limit:
        meta = meta[:limit]
    eds = [load_ed(m) for m in meta]

    agent = None
    if algorithm == "scheduling_rl":
        from agent.rl_agent import SchedulingAgent
        agent = SchedulingAgent.load(model_path, env_data=eds[0])

    rows = []
    if not quiet:
        print(f"{'dataset':<22}{'base_conv':>10}{'seq_agree':>11}{'min':>7}"
              f"{'first':>8}{'conv_range':>13}")
        print("-" * 71)
    for m, ed in zip(meta, eds):
        r = run_dataset(ed, m, algorithm=algorithm, agent=agent,
                        max_wip_perturb=max_wip_perturb, horizon_shifts=horizon_shifts,
                        eqp_down=eqp_down, pin_axis=pin_axis)
        rows.append(r)
        if not quiet:
            cr = f"{r['conv_range'][0]}~{r['conv_range'][1]}"
            flag = " *비단조" if r["conv_nonmonotonic"] else ""
            print(f"{r['id']:<22}{r['base_conv']:>10}{_fmt(r['seq_agree_mean']):>11}"
                  f"{_fmt(r['seq_agree_min']):>7}{_fmt(r['first_agree_mean']):>8}"
                  f"{cr:>13}{flag}")

    vals = [r["seq_agree_mean"] for r in rows if r["seq_agree_mean"] is not None]
    firsts = [r["first_agree_mean"] for r in rows if r["first_agree_mean"] is not None]
    summary = {
        "suite": suite,
        "pin_axis": pin_axis,
        "algorithm": algorithm,
        "n_datasets": len(rows),
        "seq_agree_mean": round(sum(vals) / len(vals), 1) if vals else None,
        "seq_agree_worst": min(
            (r["seq_agree_min"] for r in rows if r["seq_agree_min"] is not None),
            default=None,
        ),
        "first_agree_mean": round(sum(firsts) / len(firsts), 1) if firsts else None,
        "total_base_conv": sum(r["base_conv"] for r in rows),
        "n_nonmonotonic": sum(1 for r in rows if r["conv_nonmonotonic"]),
    }
    if not quiet:
        print("\n" + "=" * 71)
        print(f"시퀀스 일치율 평균 {_fmt(summary['seq_agree_mean'])}%  "
              f"(최악 {_fmt(summary['seq_agree_worst'])}%)   "
              f"첫배정 일치 {_fmt(summary['first_agree_mean'])}%")
        print(f"전환 합계 {summary['total_base_conv']}   "
              f"비단조 데이터셋 {summary['n_nonmonotonic']}/{summary['n_datasets']}")
    return {"summary": summary, "datasets": rows}


def _fmt(v) -> str:
    return "-" if v is None else f"{v}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="bench", choices=["bench", "holdout", "train_pool"])
    ap.add_argument("--algorithm", default="scheduling_rl")
    ap.add_argument("--model", default=None, help="RL 모델 zip 경로")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개 데이터셋만")
    ap.add_argument("--max-wip-perturb", type=int, default=4,
                    help="데이터셋당 재공 감소 perturbation 개수 상한")
    ap.add_argument("--no-eqp-down", action="store_true",
                    help="설비 down perturbation 생략")
    ap.add_argument("--pin-axis", action="store_true",
                    help="데이터셋별 baseline 축 순서를 axis_map으로 고정한 뒤 측정")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    out = run_consistency(
        suite=args.suite, algorithm=args.algorithm, model_path=args.model,
        limit=args.limit, max_wip_perturb=args.max_wip_perturb,
        eqp_down=not args.no_eqp_down, pin_axis=args.pin_axis,
    )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n저장 → {path}")


if __name__ == "__main__":
    main()
