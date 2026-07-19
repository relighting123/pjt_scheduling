"""
tests/test_db_load_logging.py

execute_sql_text()로 실행되는 INSERT/DELETE/DDL 문도 SELECT(sql_fetch_*.log)와
마찬가지로 logs/sql_load_*.log 파일에 남아야 한다.
"""
from datetime import datetime

import data.writer.db_load as db_load


class _FakeCursor:
    def __init__(self):
        self.executed: list = []
        self.rowcount = 0
        self.closed = False

    def execute(self, stmt):
        self.executed.append(stmt)
        self.rowcount = 3 if stmt.strip().upper().startswith("INSERT") else 1

    def close(self):
        self.closed = True


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _log_path(tmp_path):
    today = datetime.now().strftime("%Y%m%d")
    return tmp_path / "logs" / f"sql_load_{today}.log"


def _reset_sql_logger(monkeypatch, tmp_path):
    import logging

    monkeypatch.setattr(db_load, "BASE_DIR", tmp_path)
    monkeypatch.setattr(db_load, "_sql_logger", None)
    logging.getLogger("sql_load").handlers.clear()


def test_insert_and_delete_statements_are_logged(tmp_path, monkeypatch):
    _reset_sql_logger(monkeypatch, tmp_path)

    sql_text = (
        "DELETE FROM RTS_EQPCONVPLAN_INF WHERE FAC_ID = 'FAC001';\n"
        "INSERT INTO RTS_EQPCONVPLAN_INF (FAC_ID) VALUES ('FAC001');\n"
    )
    conn = _FakeConn()
    count = db_load.execute_sql_text(conn, sql_text, label="rts_eqpconvplan_inf.sql")

    assert count == 2
    assert conn.committed is True

    log_path = _log_path(tmp_path)
    assert log_path.is_file()
    contents = log_path.read_text(encoding="utf-8")
    assert "DELETE FROM RTS_EQPCONVPLAN_INF" in contents
    assert "INSERT INTO RTS_EQPCONVPLAN_INF" in contents
    assert "rts_eqpconvplan_inf.sql" in contents


def test_ddl_statements_are_also_logged(tmp_path, monkeypatch):
    _reset_sql_logger(monkeypatch, tmp_path)

    conn = _FakeConn()
    db_load.execute_sql_text(conn, "CREATE TABLE FOO (ID NUMBER);\n", label="ddl.sql")

    log_path = _log_path(tmp_path)
    assert "CREATE TABLE FOO" in log_path.read_text(encoding="utf-8")
