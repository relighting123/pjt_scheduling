"""data.writer.db_load – SQL 분리 / 적재 헬퍼 테스트."""
from data.writer.db_load import split_sql_statements


def test_split_sql_statements_skips_comments_and_blank():
    sql = """
-- header
CREATE TABLE T1 (ID NUMBER);

-- comment only

INSERT INTO T1 (ID) VALUES (1);
"""
    stmts = split_sql_statements(sql)
    assert len(stmts) == 2
    assert stmts[0].startswith("CREATE TABLE")
    assert stmts[1].startswith("INSERT INTO")


def test_split_sql_statements_multiple_ddl():
    sql = "CREATE TABLE A (X NUMBER); CREATE INDEX IX_A ON A (X);"
    stmts = split_sql_statements(sql)
    assert len(stmts) == 2
