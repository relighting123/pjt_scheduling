"""
sametool_setup 설계서(2026-08-14-sametool-setup-dedication-design.md)의
50개 테스트셋 기대값을 산출하는 참조 계산기.

설계서에 정의된 알고리즘(A: target_eqp_count, B: gap/grade 기반 STAY/SWITCH
판단)을 그대로 구현해, 각 시나리오의 기대 출력(target_eqp_count, grade,
선택된 버킷, TOOL 전환 여부, sametool_setup 여부)을 계산한다.

이 스크립트는 구현 코드가 아니라 "설계서에 적을 숫자가 맞는지" 검증하기
위한 1회성 참조 도구다. 실제 구현(simulator.py)은 별도 계획(writing-plans)을
거쳐 작성한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

GRADE_MAX = 3


@dataclass
class Bucket:
    ppk: str
    oper: str
    remaining_target: float          # achievable_qty - completed_qty
    batch: Tuple[str, str]           # (LOT_CD, TEMP) = BATCHID
    dedicated: List[Tuple[str, float]] = field(default_factory=list)
    # dedicated: [(eqp_id, ST_for_this_bucket), ...] 현재 prev_prod/prev_oper가
    # 이 버킷과 일치하는 장비들과 그 장비가 이 버킷을 처리할 때의 ST(분/장)


def covered_by_others(bucket: Bucket, t_avail: float, exclude_eqp: Optional[str]) -> float:
    total = 0.0
    for eid, st in bucket.dedicated:
        if eid == exclude_eqp or st <= 0:
            continue
        total += t_avail / st
    return total


def gap(bucket: Bucket, t_avail: float, exclude_eqp: Optional[str]) -> float:
    return bucket.remaining_target - covered_by_others(bucket, t_avail, exclude_eqp)


def target_eqp_count(bucket: Bucket, t_avail: float, avg_st: float) -> int:
    """리포트용 '적정 장비댓수' — 버킷 전체를 처리하는 데 필요한 총 대수."""
    if bucket.remaining_target <= 0 or avg_st <= 0 or t_avail <= 0:
        return 0
    capacity_per_eqp = t_avail / avg_st
    return max(math.ceil(bucket.remaining_target / capacity_per_eqp), 0)


def grade_for_candidate(bucket: Bucket, t_avail: float, eqp_id: str, st: float) -> Tuple[float, int]:
    g = gap(bucket, t_avail, exclude_eqp=eqp_id)
    if st <= 0 or t_avail <= 0:
        return g, 0
    capacity_per_eqp = t_avail / st
    needed_more = math.ceil(g / capacity_per_eqp) if g > 0 else 0
    grade = max(0, min(needed_more, GRADE_MAX))
    return g, grade


@dataclass
class Candidate:
    bucket: Bucket
    st: float                # 이 장비가 이 버킷을 처리할 때 ST


def decide_next_bucket(
    eqp_id: str,
    eqp_prev_ppk: Optional[str],
    eqp_prev_oper: Optional[str],
    eqp_batch: Optional[Tuple[str, str]],
    candidates: List[Candidate],
    t_avail: float,
) -> Dict:
    scored = []
    for c in candidates:
        g, grade = grade_for_candidate(c.bucket, t_avail, eqp_id, c.st)
        same_ppk_oper = (c.bucket.ppk == eqp_prev_ppk and c.bucket.oper == eqp_prev_oper)
        same_batch = (c.bucket.batch == eqp_batch)
        scored.append({
            "bucket": c.bucket, "gap": g, "grade": grade,
            "same_ppk_oper": same_ppk_oper, "same_batch": same_batch,
        })

    def best(pool):
        return max(pool, key=lambda s: (s["grade"], s["gap"]))

    tier1 = [s for s in scored if s["same_ppk_oper"] and s["grade"] >= 1]
    tier2 = [s for s in scored if s["same_batch"] and s["grade"] >= 1]
    tier3 = [s for s in scored if s["grade"] >= 1]

    if tier1:
        chosen, tier = best(tier1), 1
    elif tier2:
        chosen, tier = best(tier2), 2
    elif tier3:
        chosen, tier = best(tier3), 3
    else:
        chosen, tier = best(scored), 4

    b = chosen["bucket"]
    tool_conversion = (eqp_batch is not None and b.batch != eqp_batch)
    sametool_setup = (
        not tool_conversion
        and eqp_batch is not None
        and not (b.ppk == eqp_prev_ppk and b.oper == eqp_prev_oper)
    )
    return {
        "chosen_ppk_oper": (b.ppk, b.oper), "tier": tier,
        "grade": chosen["grade"], "gap": round(chosen["gap"], 2),
        "tool_conversion": tool_conversion, "sametool_setup": sametool_setup,
        "all_grades": {(s["bucket"].ppk, s["bucket"].oper): s["grade"] for s in scored},
    }


# ──────────────────────────────────────────────────────────────────────────
# 50개 시나리오 생성 (5그룹 × 10)
# ──────────────────────────────────────────────────────────────────────────

def fmt_row(idx, group, desc, result, extra=""):
    b = result["chosen_ppk_oper"]
    return (
        f"| {idx} | {group} | {desc} | {result['grade']} | {result['gap']} | "
        f"{b[0]}/{b[1]} (Tier{result['tier']}) | "
        f"{'O' if result['tool_conversion'] else '-'} | "
        f"{'O' if result['sametool_setup'] else '-'} | {extra} |"
    )


rows = []
idx = 1

# ---- Group 1: 기본 단일 버킷 (target_eqp_count / grade 계산 정확성) ----
BATCH_A = ("LOT_A", "T1")
for i in range(10):
    remaining = 50 + i * 40          # 50..410
    t_avail = 480                    # 8시간
    st = 8 + (i % 4) * 2             # 8,10,12,14 순환
    dedicated_n = i % 3               # 0,1,2 기존 배치 장비
    bucket = Bucket("PPK1", "OPER1", remaining, BATCH_A,
                     dedicated=[(f"E{d}", st) for d in range(dedicated_n)])
    tgt = target_eqp_count(bucket, t_avail, st)
    cand = [Candidate(bucket, st)]
    res = decide_next_bucket("E_NEW", None, None, None, cand, t_avail)
    desc = f"remain={remaining}, T={t_avail}, ST={st}, 기배치={dedicated_n}대"
    rows.append(fmt_row(idx, "G1-기본", desc, res, extra=f"target_eqp_count={tgt}"))
    idx += 1

# ---- Group 2: 동일 BATCH, 다중 PPK/OPER 공유 (sametool_setup 검증) ----
# E1 자신 외에 별도 장비 E_other가 OPER1을 이미 커버하는 케이스를 절반 섞어
# grade(OPER1)=0이 되어 Tier2(sametool_setup)로 넘어가는 경우도 나오게 한다.
BATCH_B = ("LOT_B", "T2")
for i in range(10):
    t_avail = 300
    st1 = 10
    st2 = 10 + (i % 3) * 5    # 10,15,20
    other_covers = (i % 2 == 1)   # 홀수 인덱스: 다른 장비가 OPER1을 충분히 커버
    remain1 = 20 + (i % 2) * 10
    remain2 = 40 + i * 12
    dedicated1 = [("E1", st1)]
    if other_covers:
        dedicated1.append(("E_other", st1))   # 별도 장비가 이미 OPER1 커버
    b1 = Bucket("PPK2", "OPER1", remain1, BATCH_B, dedicated=dedicated1)
    b2 = Bucket("PPK2", "OPER2", remain2, BATCH_B, dedicated=[])
    cand = [Candidate(b1, st1), Candidate(b2, st2)]
    # 이 장비는 방금 PPK2/OPER1을 끝내고 idle, 배치는 BATCH_B 유지 중
    res = decide_next_bucket("E1", "PPK2", "OPER1", BATCH_B, cand, t_avail)
    desc = (
        f"OPER1 remain={remain1}(E_other 커버:{other_covers}), "
        f"OPER2 remain={remain2}, ST2={st2}"
    )
    rows.append(fmt_row(idx, "G2-동일배치", desc, res))
    idx += 1

# ---- Group 3: BATCH 전환 필요 (TOOL 전환 최소화) ----
# 같은 배치 버킷의 기배치 대수를 0~2대로 섞어, "같은 배치 안에서 해결되는 경우"와
# "정말 불가피하게 전환해야 하는 경우"가 모두 나오게 한다.
BATCH_C = ("LOT_C", "T1")
BATCH_D = ("LOT_D", "T3")
for i in range(10):
    t_avail = 240
    st_same = 12
    st_other = 9 + (i % 4)
    same_dedicated_n = i % 3          # 0,1,2대 순환
    remain_same_batch = 40 + (i % 2) * 40
    remain_other_batch = 80 + i * 20
    b_same = Bucket(
        "PPK3", "OPER1", remain_same_batch, BATCH_C,
        dedicated=[(f"Ex{k}", st_same) for k in range(same_dedicated_n)],
    )
    b_other = Bucket("PPK4", "OPER2", remain_other_batch, BATCH_D, dedicated=[])
    cand = [Candidate(b_same, st_same), Candidate(b_other, st_other)]
    res = decide_next_bucket("E9", "PPK3", "OPER1", BATCH_C, cand, t_avail)
    desc = (
        f"동일배치 remain={remain_same_batch}(기배치{same_dedicated_n}대), "
        f"타배치 remain={remain_other_batch}"
    )
    rows.append(fmt_row(idx, "G3-배치전환", desc, res))
    idx += 1

# ---- Group 4: 상류 유입 변화에 따른 target 동적 변화 (3스텝 시계열) ----
for i in range(10):
    t_avail0 = 360
    st = 10
    base_remain = 30 + i * 10
    # 3개 시점: 유입 전/유입 후 급증/소진 후 감소
    steps = [base_remain, base_remain + 90, max(base_remain + 90 - 70, 0)]
    t_avails = [t_avail0, t_avail0 - 60, t_avail0 - 150]
    targets = []
    for r, t in zip(steps, t_avails):
        b = Bucket("PPK5", "OPER3", r, BATCH_A, dedicated=[])
        targets.append(target_eqp_count(b, t, st))
    desc = f"remain 추이={steps}, T_avail 추이={t_avails} -> target 추이={targets}"
    rows.append(f"| {idx} | G4-유입변화 | {desc} | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |")
    idx += 1

# ---- Group 5: 등급 경계/동률 (진동 방지) ----
for i in range(10):
    t_avail = 200
    st = 10
    capacity = t_avail / st  # 한 대당 처리 가능량
    # gap을 등급 경계(±0.4 단위) 바로 근처로 촘촘히 배치해 grade가 안정적인지 확인
    remain = capacity * (1.0 + 0.05 * (i - 5))   # grade=1 근방에서 미세 변동
    b = Bucket("PPK6", "OPER4", remain, BATCH_A, dedicated=[])
    cand = [Candidate(b, st)]
    res = decide_next_bucket("E_X", "PPK6", "OPER4", BATCH_A, cand, t_avail)
    desc = f"remain={remain:.1f} (capacity/eqp={capacity:.1f} 근방 미세변동 {i-5:+d}칸)"
    rows.append(fmt_row(idx, "G5-경계동률", desc, res))
    idx += 1

header = (
    "| # | 그룹 | 입력 요약 | grade | gap | 선택 버킷(Tier) | TOOL전환 | sametool_setup | 비고 |\n"
    "|---|---|---|---|---|---|---|---|---|"
)
out_path = __file__.rsplit("\\", 1)[0] + "\\scenarios_output.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for r in rows:
        f.write(r + "\n")

# 요약 집계 (설계서 본문에 인용)
n_tool_conv = sum(1 for r in rows if " O | - |" in r)
n_sametool = sum(1 for r in rows if " - | O |" in r)
print(f"wrote {len(rows)} rows to {out_path}")
print(f"tool_conversion=O count: {n_tool_conv}, sametool_setup=O count: {n_sametool}")
