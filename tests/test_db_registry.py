"""DB alias 레지스트리·collector 단위 테스트 (Oracle 연결 없음)."""
from pathlib import Path

from data.collector import TrainingDataCollector, build_arg_parser
from data.db_registry import (
    DbRegistry,
    load_db_aliases_from_env,
    parse_sql_db_alias,
)


def test_parse_sql_db_alias_from_header():
    sql = """-- @db: plan
-- plan.sql
SELECT 1 FROM dual
"""
    assert parse_sql_db_alias(sql) == "plan"


def test_parse_sql_db_alias_default():
    sql = "-- plan.sql\nSELECT 1 FROM dual"
    assert parse_sql_db_alias(sql, "main") == "main"


def test_load_db_aliases_from_env():
    env = {
        "DB_MAIN_USER": "u1",
        "DB_MAIN_PASSWORD": "p1",
        "DB_MAIN_DSN": "d1",
        "DB_PLAN_USER": "u2",
        "DB_PLAN_PASSWORD": "p2",
        "DB_PLAN_DSN": "d2",
    }
    aliases = load_db_aliases_from_env(env)
    assert set(aliases) == {"main", "plan"}
    assert aliases["main"].user == "u1"
    assert aliases["plan"].dsn == "d2"


def test_oracle_legacy_maps_to_main():
    env = {
        "ORACLE_USER": "legacy",
        "ORACLE_PASSWORD": "secret",
        "ORACLE_DSN": "host:1521/xe",
    }
    aliases = load_db_aliases_from_env(env)
    assert aliases["main"].user == "legacy"


def test_db_registry_unknown_alias_raises():
    registry = DbRegistry(
        aliases=load_db_aliases_from_env({"DB_MAIN_USER": "u", "DB_MAIN_PASSWORD": "p", "DB_MAIN_DSN": "d"}),
    )
    try:
        registry.get_credentials("missing")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "missing" in str(e)


def test_sql_files_declare_db_alias():
    sql_dir = Path(__file__).resolve().parent.parent / "external" / "sql"
    for path in sql_dir.glob("*.sql"):
        text = path.read_text(encoding="utf-8")
        assert "@db:" in text.splitlines()[0], f"{path.name} 에 -- @db: 헤더 필요"


def test_collector_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--once", "--fac-id", "FAC001"])
    assert args.once is True
    assert args.fac_id == "FAC001"


def test_collector_resolve_range_prevdays():
    c = TrainingDataCollector(fac_id="FAC001", prevdays=1)
    start, end = c._resolve_range()
    assert len(start) == 14
    assert len(end) == 14
