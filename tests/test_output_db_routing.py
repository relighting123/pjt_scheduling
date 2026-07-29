"""출력 테이블별로 다른 DB에 적재할 수 있는지 검증 (config/output_db_routing.yaml).

load_output_sql_files()/load_output_json() 은 이 파일에 적힌 테이블만 그
alias로 연결해 실행하고, 적히지 않은 테이블이나 파일 자체가 없으면 기존처럼
db_alias(또는 registry 기본 alias) 하나만 그대로 쓴다.
"""
from pathlib import Path

import data.writer.db_load as db_load


def test_load_output_db_routing_returns_empty_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    assert db_load.load_output_db_routing(missing) == {}


def test_load_output_db_routing_parses_table_to_alias_map(tmp_path):
    path = tmp_path / "output_db_routing.yaml"
    path.write_text(
        "RTS_PERFMON_HIS: Dev\n"
        "rts_validation: Dev\n",  # 소문자로 적어도 대문자 테이블명으로 정규화
        encoding="utf-8",
    )
    assert db_load.load_output_db_routing(path) == {
        "RTS_PERFMON_HIS": "Dev",
        "RTS_VALIDATION": "Dev",
    }


def test_load_output_db_routing_ignores_default_key(tmp_path):
    path = tmp_path / "output_db_routing.yaml"
    path.write_text("default: Prd\nRTS_VALIDATION: Dev\n", encoding="utf-8")
    assert db_load.load_output_db_routing(path) == {"RTS_VALIDATION": "Dev"}


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
    routing_path = tmp_path / "output_db_routing.yaml"
    routing_path.write_text("RTS_PERFMON_HIS: dev\n", encoding="utf-8")
    monkeypatch.setattr(db_load, "_output_db_routing_path", lambda: routing_path)

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
        db_alias="prd",
        include_history=False,
        script_names=["rts_rslt_mas.sql", "rts_eqpconvplan_inf.sql", "rts_perfmon_his.sql"],
    )

    assert len(executed) == 3
    # RTS_RSLT_MAS/RTS_EQPCONVPLAN_INF는 명시적으로 넘긴 db_alias="prd"로,
    # RTS_PERFMON_HIS는 output_db_routing.yaml 을 따라 "dev"로 실행됐어야 한다.
    assert "prd" in fake_registry._connections
    assert "dev" in fake_registry._connections
    prd_sql = "\n".join(fake_registry._connections["prd"].executed)
    dev_sql = "\n".join(fake_registry._connections["dev"].executed)
    assert "RTS_RSLT_MAS" in prd_sql
    assert "RTS_EQPCONVPLAN_INF" in prd_sql
    assert "RTS_PERFMON_HIS" not in prd_sql
    assert "RTS_PERFMON_HIS" in dev_sql


def test_load_output_sql_files_uses_single_alias_when_no_routing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(db_load, "_output_db_routing_path", lambda: tmp_path / "missing.yaml")

    out_dir = tmp_path / "output"
    sql_dir = out_dir / "sql"
    sql_dir.mkdir(parents=True)
    (sql_dir / "rts_rslt_mas.sql").write_text("INSERT INTO RTS_RSLT_MAS VALUES (1);", encoding="utf-8")
    (sql_dir / "rts_perfmon_his.sql").write_text("INSERT INTO RTS_PERFMON_HIS VALUES (1);", encoding="utf-8")

    fake_registry = _FakeRegistry()
    monkeypatch.setattr(db_load, "DbRegistry", lambda *a, **kw: fake_registry)

    db_load.load_output_sql_files(
        out_dir,
        db_alias="prd",
        include_history=False,
        script_names=["rts_rslt_mas.sql", "rts_perfmon_his.sql"],
    )

    # 라우팅 파일이 없으면 모든 테이블이 넘긴 db_alias 하나로만 실행된다
    assert list(fake_registry._connections.keys()) == ["prd"]
