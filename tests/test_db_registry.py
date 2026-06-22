"""DB alias 레지스트리·collector 단위 테스트 (Oracle 연결 없음)."""
import textwrap
from pathlib import Path

import yaml

from data.collector import TrainingDataCollector, build_arg_parser
from data.db_registry import (
    DbRegistry,
    default_db_alias,
    load_db_aliases,
    load_db_aliases_from_yaml,
    parse_sql_db_alias,
    resolve_db_credentials,
)


def test_parse_sql_db_alias_from_header():
    sql = """-- @db: Prd
-- plan.sql
SELECT 1 FROM dual
"""
    assert parse_sql_db_alias(sql) == "prd"


def test_parse_sql_db_alias_hierarchical():
    sql = "-- @db: Prd.Plan\nSELECT 1"
    assert parse_sql_db_alias(sql) == "prd.plan"


def test_parse_sql_db_alias_default():
    sql = "-- plan.sql\nSELECT 1 FROM dual"
    assert parse_sql_db_alias(sql, "Prd") == "prd"


def test_yaml_hierarchical_prd_dev(tmp_path):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        textwrap.dedent("""
        default: Prd

        Prd:
          User: prd_user
          Pw: prd_pw
          Dsn: prd-dsn

        Dev:
          user: dev_user
          password: dev_pw
          dsn: dev-dsn
        """),
        encoding="utf-8",
    )
    buckets, yaml_default = load_db_aliases_from_yaml(cfg)
    assert yaml_default == "prd"
    assert set(buckets) == {"prd", "dev"}

    prd = resolve_db_credentials("Prd", buckets)
    assert prd.user == "prd_user"
    assert prd.password == "prd_pw"
    assert prd.dsn == "prd-dsn"

    dev = resolve_db_credentials("dev", buckets)
    assert dev.user == "dev_user"


def test_yaml_nested_inheritance(tmp_path):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        yaml.dump({
            "default": "Prd",
            "Prd": {
                "user": "base_u",
                "password": "base_p",
                "dsn": "base-dsn",
                "Plan": {"dsn": "plan-dsn"},
            },
        }),
        encoding="utf-8",
    )
    buckets, _ = load_db_aliases_from_yaml(cfg)
    plan = resolve_db_credentials("Prd.Plan", buckets)
    assert plan.user == "base_u"
    assert plan.password == "base_p"
    assert plan.dsn == "plan-dsn"


def test_default_alias_priority(tmp_path, monkeypatch):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text("default: Dev\nDev:\n  user: u\n  password: p\n  dsn: d\n", encoding="utf-8")
    monkeypatch.setenv("DB_DEFAULT_ALIAS", "Prd")
    _, yaml_def = load_db_aliases_from_yaml(cfg)
    assert default_db_alias(yaml_def) == "prd"


def test_oracle_legacy_maps_to_main():
    buckets, _ = load_db_aliases(
        yaml_path=Path("/nonexistent/databases.yaml"),
        environ={
            "ORACLE_USER": "legacy",
            "ORACLE_PASSWORD": "secret",
            "ORACLE_DSN": "host:1521/xe",
        },
    )
    assert resolve_db_credentials("main", buckets).user == "legacy"


def test_db_registry_unknown_alias_raises():
    registry = DbRegistry(
        alias_buckets={"main": {"user": "u", "password": "p", "dsn": "d"}},
        default_alias="main",
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
