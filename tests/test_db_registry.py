"""DB alias 레지스트리·collector 단위 테스트 (Oracle 연결 없음)."""
from pathlib import Path

from data.collector import TrainingDataCollector, build_arg_parser
from data.db_registry import (
    DbRegistry,
    alias_to_env_prefix,
    load_db_aliases_from_env,
    parse_sql_db_alias,
    resolve_db_credentials,
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
    assert resolve_db_credentials("main", aliases).user == "u1"
    assert resolve_db_credentials("plan", aliases).dsn == "d2"


def test_hierarchical_alias_env_and_sql():
    env = {
        "DB_FAB__MES_USER": "mes_user",
        "DB_FAB__MES_PASSWORD": "mes_pw",
        "DB_FAB__MES_DSN": "mes-dsn",
        "DB_FAB__MES__PLAN_DSN": "plan-dsn",
    }
    buckets = load_db_aliases_from_env(env)
    assert set(buckets) == {"fab.mes", "fab.mes.plan"}

    mes = resolve_db_credentials("fab.mes", buckets)
    assert mes.user == "mes_user"
    assert mes.dsn == "mes-dsn"

    plan = resolve_db_credentials("fab.mes.plan", buckets)
    assert plan.user == "mes_user"
    assert plan.password == "mes_pw"
    assert plan.dsn == "plan-dsn"

    sql = "-- @db: fab.mes.plan\nSELECT 1"
    assert parse_sql_db_alias(sql) == "fab.mes.plan"


def test_alias_to_env_prefix():
    assert alias_to_env_prefix("fab.mes") == "FAB__MES"
    assert alias_to_env_prefix("main") == "MAIN"


def test_oracle_legacy_maps_to_main():
    env = {
        "ORACLE_USER": "legacy",
        "ORACLE_PASSWORD": "secret",
        "ORACLE_DSN": "host:1521/xe",
    }
    buckets = load_db_aliases_from_env(env)
    assert resolve_db_credentials("main", buckets).user == "legacy"


def test_db_registry_unknown_alias_raises():
    registry = DbRegistry(
        alias_buckets=load_db_aliases_from_env({
            "DB_MAIN_USER": "u", "DB_MAIN_PASSWORD": "p", "DB_MAIN_DSN": "d",
        }),
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
