"""
data/loader/sql_binds.py – Oracle fetch 공통 바인드 (FAC_ID, RULE_TIMEKEY)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import CONFIG, normalize_rule_timekey, validate_path_segment


def merge_fetch_binds(
    fac_id: str,
    period: Optional[str] = None,
    *,
    extra_binds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """fetch_from_db 에 전달할 바인드 dict."""
    binds: Dict[str, Any] = {
        "FAC_ID": validate_path_segment(fac_id, "FAC_ID"),
    }
    if period:
        binds["RULE_TIMEKEY"] = normalize_rule_timekey(period)
    if extra_binds:
        binds.update(extra_binds)
    if CONFIG.oracle.extra_binds:
        binds.update(CONFIG.oracle.extra_binds)
    return binds
