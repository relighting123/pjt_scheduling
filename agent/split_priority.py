"""
agent/split_priority.py – split sub-lot / 기할당 LOT 우선순위

동일 parent(원 LOT)에서 이미 배정한 sub-lot이 있으면
같은 parent의 다음 split(LOT001__S02 등)을 우선 선택합니다.
"""
from typing import Dict, Optional, Tuple


def parent_lot_id(lot_id: str, lot_by_id: Optional[Dict[str, dict]] = None) -> str:
    if lot_by_id and lot_id in lot_by_id:
        return lot_by_id[lot_id].get("parent_lot_id") or lot_id
    if "__S" in lot_id:
        return lot_id.rsplit("__S", 1)[0]
    return lot_id


def split_suffix_index(lot_id: str) -> int:
    if "__S" not in lot_id:
        return 0
    try:
        return int(lot_id.rsplit("__S", 1)[1])
    except ValueError:
        return 0


def assigned_split_priority(
    prev_lot_id: Optional[str],
    lot_id: str,
    lot_by_id: Optional[Dict[str, dict]] = None,
) -> Tuple[int, int, int]:
    """
    정렬 키 (작을수록 우선)
      0. 직전 배정 LOT과 동일 parent
      1. 바로 다음 split 순번 (S01 → S02)
      2. split 순번
    """
    if not prev_lot_id:
        return (1, 0, split_suffix_index(lot_id))

    prev_parent = parent_lot_id(prev_lot_id, lot_by_id)
    lot_parent = parent_lot_id(lot_id, lot_by_id)

    if prev_parent != lot_parent:
        return (1, 0, split_suffix_index(lot_id))

    prev_idx = split_suffix_index(prev_lot_id)
    lot_idx = split_suffix_index(lot_id)

    if lot_idx == prev_idx + 1:
        continuity = 0
    elif lot_idx > prev_idx:
        continuity = 1
    else:
        continuity = 2

    return (0, continuity, lot_idx)


def prev_assigned_lot_id(sim, eqp_id: Optional[str]) -> Optional[str]:
    if not eqp_id:
        return None
    eqp = sim.eqps.get(eqp_id)
    return eqp.prev_lot_id if eqp else None
