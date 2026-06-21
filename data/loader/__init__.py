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

__all__ = [
    "load_data",
    "validate_data",
    "fetch_from_db",
    "fetch_period_range",
    "preprocess",
]
