"""
tests/test_db_load_logging.py

execute_sql_text()로 실행되는 INSERT/DELETE/DDL 문도 SELECT(sql_fetch.log)와
마찬가지로 logs/sql_load.log 파일에 남아야 한다(자정 회전, 백업 3일치 보관).
실행 실패(FAILED)는 터미널(stderr)에도 "[ERROR] 시각 …" 한 줄 요약으로 출력된다.
"""
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


class _FailingCursor(_FakeCursor):
    def execute(self, stmt):
        self.executed.append(stmt)
        if "BAD" in stmt.upper():
            raise Exception("ORA-01843: not a valid month")
        self.rowcount = 1


class _FailingConn(_FakeConn):
    def __init__(self):
        super().__init__()
        self.cursor_obj = _FailingCursor()


def _log_path(tmp_path):
    return tmp_path / "logs" / "sql_load.log"


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


def test_failing_statement_is_logged_before_the_exception_propagates(tmp_path, monkeypatch):
    _reset_sql_logger(monkeypatch, tmp_path)

    sql_text = (
        "INSERT INTO GOOD_TABLE (ID) VALUES (1);\n"
        "INSERT INTO BAD_TABLE (ID) VALUES (2);\n"
    )
    conn = _FailingConn()
    try:
        db_load.execute_sql_text(conn, sql_text, label="mixed.sql")
        assert False, "expected exception to propagate"
    except Exception as exc:
        assert "not a valid month" in str(exc)

    assert conn.rolled_back is True
    assert conn.committed is False

    contents = _log_path(tmp_path).read_text(encoding="utf-8")
    assert "INSERT INTO GOOD_TABLE" in contents
    assert "FAILED" in contents
    assert "BAD_TABLE" in contents
    assert "not a valid month" in contents


def test_failing_statement_is_echoed_to_terminal(tmp_path, monkeypatch, capsys):
    """실행 실패는 파일 외에 stderr에도 '[ERROR] 시각 [label] FAILED: …' 한 줄로 출력."""
    import re

    _reset_sql_logger(monkeypatch, tmp_path)

    conn = _FailingConn()
    try:
        db_load.execute_sql_text(
            conn, "INSERT INTO BAD_TABLE (ID) VALUES (2);\n", label="mixed.sql"
        )
        assert False, "expected exception to propagate"
    except Exception:
        pass

    err = capsys.readouterr().err.strip()
    assert re.match(
        r"^\[ERROR\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
        r"\[mixed\.sql\] FAILED: ORA-01843: not a valid month$",
        err,
    )
    assert "INSERT INTO" not in err  # SQL 본문은 파일 전용
