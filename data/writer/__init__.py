"""
data/writer – 추론 결과 Oracle 적재용 출력 (loader 반대)

loader:  Oracle SQL → dataset/.../input/*.json
writer:  추론 결과 → dataset/.../output/output.json + sql/*.sql
"""
from data.writer.rts_json import build_rts_output, resolve_writer_meta
from data.writer.rts_sql import build_writer_sql_scripts, write_sql
from data.writer.write import write_inference_result, write_sql_from_json
from data.writer.db_load import (
    apply_output_ddl,
    load_output_json,
    load_output_sql_files,
    resolve_output_dir,
    split_sql_statements,
)

__all__ = [
    "build_rts_output",
    "resolve_writer_meta",
    "build_writer_sql_scripts",
    "write_sql",
    "write_inference_result",
    "write_sql_from_json",
    "apply_output_ddl",
    "load_output_json",
    "load_output_sql_files",
    "resolve_output_dir",
    "split_sql_statements",
]
