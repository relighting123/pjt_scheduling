"""출력 테이블별 -- @db: 헤더로 다른 DB에 적재할 수 있는지 검증.

data/sql/rts_output_tables.sql 에 테이블별 헤더를 두면(입력 SQL과 동일한
표기), load_output_sql_files()/load_output_json() 이 테이블마다 그 alias로
연결해 실행해야 한다. 헤더가 없으면(기존 파일) 기존처럼 db_alias 하나만
그대로 쓴다.
"""
from pathlib import Path

import data.writer.db_load as db_load
from data.db_registry import parse_per_table_db_aliases


def test_parse_per_table_db_aliases_single_header_applies_to_all():
    sql = (
        "-- @db: Prd\n"
        "CREATE TABLE RTS_RSLT_MAS (X NUMBER);\n"
        "CREATE TABLE RTS_RSLT_HIS (X NUMBER);\n"
    )
    assert parse_per_table_db_aliases(sql) == {
        "RTS_RSLT_MAS": "prd",
        "RTS_RSLT_HIS": "prd",
    }


def test_parse_per_table_db_aliases_new_header_overrides_following_tables():
    sql = (
        "-- @db: Prd\n"
        "CREATE TABLE RTS_RSLT_MAS (X NUMBER);\n"
        "-- @db: Dev\n"
        "CREATE TABLE RTS_PERFMON_HIS (X NUMBER);\n"
        "CREATE TABLE RTS_VALIDATION (X NUMBER);\n"
    )
    assert parse_per_table_db_aliases(sql) == {
        "RTS_RSLT_MAS": "prd",
        "RTS_PERFMON_HIS": "dev",
        "RTS_VALIDATION": "dev",
    }


def test_parse_per_table_db_aliases_no_header_uses_default():
    sql = "CREATE TABLE RTS_RSLT_MAS (X NUMBER);\n"
    assert parse_per_table_db_aliases(sql, default_alias="main") == {
        "RTS_RSLT_MAS": "main",
    }


class _FakeConn:
    def __init__(self, alias):
        self.alias = alias
        self.executed: list = []

    def cursor(self):
        return self

    def execute(self, stmt):
        self.executed.append(stmt)
        self.rowcount = 1

    def close(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


class _FakeRegistry:
    """DbRegistry(alias)-> 커넥션 대신, alias별로 연결을 흉내내는 더미."""

    def __init__(self, *a, **kw):
        self.default_alias = "prd"
        self._connections: dict = {}

    def connect(self, alias=None):
        key = alias or self.default_alias
        if key not in self._connections:
            self._connections[key] = _FakeConn(key)
        return self._connections[key]

    def close_all(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close_all()


def test_load_output_sql_files_routes_each_table_to_its_own_alias(tmp_path, monkeypatch):
    # DDL: RTS_RSLT_MAS/HIS -> Prd(기본), RTS_PERFMON_HIS -> Dev
    ddl_dir = tmp_path / "data" / "sql"
    ddl_dir.mkdir(parents=True)
    (ddl_dir / "rts_output_tables.sql").write_text(
        "-- @db: Prd\n"
        "CREATE TABLE RTS_RSLT_MAS (X NUMBER);\n"
        "CREATE TABLE RTS_EQPCONVPLAN_INF (X NUMBER);\n"
        "-- @db: Dev\n"
        "CREATE TABLE RTS_PERFMON_HIS (X NUMBER);\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(db_load, "BASE_DIR", tmp_path)

    out_dir = tmp_path / "output"
    sql_dir = out_dir / "sql"
    sql_dir.mkdir(parents=True)
    (sql_dir / "rts_rslt_mas.sql").write_text("INSERT INTO RTS_RSLT_MAS VALUES (1);", encoding="utf-8")
    (sql_dir / "rts_eqpconvplan_inf.sql").write_text("INSERT INTO RTS_EQPCONVPLAN_INF VALUES (1);", encoding="utf-8")
    (sql_dir / "rts_perfmon_his.sql").write_text("INSERT INTO RTS_PERFMON_HIS VALUES (1);", encoding="utf-8")

    fake_registry = _FakeRegistry()
    monkeypatch.setattr(db_load, "DbRegistry", lambda *a, **kw: fake_registry)

    executed = db_load.load_output_sql_files(
        out_dir,
        include_history=False,
        script_names=["rts_rslt_mas.sql", "rts_eqpconvplan_inf.sql", "rts_perfmon_his.sql"],
    )

    assert len(executed) == 3
    # RTS_RSLT_MAS/RTS_EQPCONVPLAN_INF는 prd 커넥션에, RTS_PERFMON_HIS는 dev
    # 커넥션에 각각 실행됐어야 한다 (헤더 기준 테이블별 라우팅).
    assert "prd" in fake_registry._connections
    assert "dev" in fake_registry._connections
    prd_sql = "\n".join(fake_registry._connections["prd"].executed)
    dev_sql = "\n".join(fake_registry._connections["dev"].executed)
    assert "RTS_RSLT_MAS" in prd_sql
    assert "RTS_EQPCONVPLAN_INF" in prd_sql
    assert "RTS_PERFMON_HIS" not in prd_sql
    assert "RTS_PERFMON_HIS" in dev_sql
