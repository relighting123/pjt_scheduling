"""
data/loader – 입력 데이터 fetch (Oracle SQL / JSON) 및 전처리

writer 반대 축:
  loader:  Oracle SQL → data/dataset/.../input/*.json → env_data
  writer:  추론 결과 → data/dataset/.../output/output.json + sql/*.sql
"""
from data.loader.fetch import (
    load_data,
    validate_data,
    fetch_from_db,
    fetch_period_range,
)
from data.loader.preprocess import preprocess
from data.loader.sql_binds import merge_fetch_binds, resolve_lot_cd
from data.loader.rule_timekey_query import (
    fetch_latest_rule_timekey,
    fetch_recent_rule_timekeys,
    fetch_rule_timekey_list,
    resolve_collect_periods,
    resolve_snapshot_rule_timekey,
)

__all__ = [
    "load_data",
    "validate_data",
    "fetch_from_db",
    "fetch_period_range",
    "preprocess",
    "merge_fetch_binds",
    "resolve_lot_cd",
    "fetch_latest_rule_timekey",
    "fetch_recent_rule_timekeys",
    "fetch_rule_timekey_list",
    "resolve_collect_periods",
    "resolve_snapshot_rule_timekey",
]
