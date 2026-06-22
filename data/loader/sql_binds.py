"""
data/loader/sql_binds.py – Oracle fetch 공통 바인드 (FAC_ID, RULE_TIMEKEY, LOT_CD)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from config import CONFIG, normalize_rule_timekey, validate_path_segment


def resolve_lot_cd(value: Optional[str] = None) -> Optional[str]:
    """
    LOT_CD 필터 값 해석.
    우선순위: 인자 > SQL_LOT_CD > COLLECTOR_LOT_CD > None(전체)
    """
    if value is not None:
        text = str(value).strip()
        return text or None
    for key in ("SQL_LOT_CD", "COLLECTOR_LOT_CD"):
        text = os.environ.get(key, "").strip()
        if text:
            return text
    return None


def merge_fetch_binds(
    fac_id: str,
    period: Optional[str] = None,
    *,
    lot_cd: Optional[str] = None,
    extra_binds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """fetch_from_db 에 전달할 바인드 dict (LOT_CD 는 미지정 시 NULL)."""
    binds: Dict[str, Any] = {
        "FAC_ID": validate_path_segment(fac_id, "FAC_ID"),
        "LOT_CD": resolve_lot_cd(lot_cd),
    }
    if period:
        binds["RULE_TIMEKEY"] = normalize_rule_timekey(period)
    if extra_binds:
        binds.update(extra_binds)
    if CONFIG.oracle.extra_binds:
        binds.update(CONFIG.oracle.extra_binds)
    return binds
