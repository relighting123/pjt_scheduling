"""
tests/test_file_logger.py

utils.file_logger.get_daily_file_logger()는:
  - logs/{base_filename}에 기록하는 TimedRotatingFileHandler(자정 회전)를 붙인다.
  - 자정 회전 백업은 3일치까지 보관하고(backupCount=3) 그보다 오래된 것은 삭제한다.
  - ERROR 이상은 터미널(stderr)에도 "[ERROR] YYYY-MM-DD HH:MM:SS 메시지" 한 줄로 출력한다.
  - 이전 방식(prefix_YYYYMMDD.log)의 잔여 로그 파일은 정리한다.
"""
import logging
import logging.handlers

from utils.file_logger import get_daily_file_logger


def _clear_logger(name: str):
    logging.getLogger(name).handlers.clear()


def _file_handler(logger: logging.Logger) -> logging.handlers.TimedRotatingFileHandler:
    return next(
        h for h in logger.handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    )


def test_creates_timed_rotating_handler_with_3day_backup(tmp_path):
    _clear_logger("test_logger_a")
    log_dir = tmp_path / "logs"
    logger = get_daily_file_logger("test_logger_a", log_dir, "sql_fetch.log")

    handler = _file_handler(logger)
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 3

    logger.info("hello")
    assert (log_dir / "sql_fetch.log").is_file()
    assert "hello" in (log_dir / "sql_fetch.log").read_text(encoding="utf-8")


def test_does_not_add_duplicate_handlers_on_repeated_calls(tmp_path):
    _clear_logger("test_logger_b")
    log_dir = tmp_path / "logs"
    get_daily_file_logger("test_logger_b", log_dir, "sql_load.log")
    logger = get_daily_file_logger("test_logger_b", log_dir, "sql_load.log")
    assert len(logger.handlers) == 2


def test_cleans_up_legacy_dated_log_files(tmp_path):
    _clear_logger("test_logger_c")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "sql_fetch_20260101.log").write_text("old", encoding="utf-8")
    (log_dir / "sql_fetch_20260715.log").write_text("old", encoding="utf-8")
    (log_dir / "sql_fetch_notes.log").write_text("keep me", encoding="utf-8")

    get_daily_file_logger("test_logger_c", log_dir, "sql_fetch.log")

    remaining = sorted(p.name for p in log_dir.glob("*.log"))
    assert "sql_fetch_20260101.log" not in remaining
    assert "sql_fetch_20260715.log" not in remaining
    assert "sql_fetch_notes.log" in remaining


def test_midnight_rollover_keeps_dated_backup(tmp_path):
    """자정 회전(doRollover) 시 이전 내용은 dated 백업으로 남아야 한다(3일 보관)."""
    _clear_logger("test_logger_d")
    log_dir = tmp_path / "logs"
    logger = get_daily_file_logger("test_logger_d", log_dir, "sql_fetch.log")
    handler = _file_handler(logger)

    logger.info("yesterday's entry")
    assert "yesterday's entry" in (log_dir / "sql_fetch.log").read_text(encoding="utf-8")

    handler.doRollover()

    backups = [p for p in log_dir.glob("sql_fetch.log.*")]
    assert len(backups) == 1, f"expected one dated backup, got: {sorted(p.name for p in log_dir.glob('sql_fetch*'))}"
    assert "yesterday's entry" in backups[0].read_text(encoding="utf-8")

    logger.info("today's entry")
    contents = (log_dir / "sql_fetch.log").read_text(encoding="utf-8")
    assert "yesterday's entry" not in contents
    assert "today's entry" in contents


def test_file_log_format_is_level_then_timestamp(tmp_path):
    _clear_logger("test_logger_e")
    log_dir = tmp_path / "logs"
    logger = get_daily_file_logger("test_logger_e", log_dir, "sql_load.log")

    logger.warning("rows=0")
    line = (log_dir / "sql_load.log").read_text(encoding="utf-8").strip()
    # [WARNING] 2026-07-10 10:10:10 rows=0
    import re
    assert re.match(r"^\[WARNING\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} rows=0$", line)


def test_error_is_echoed_to_terminal_as_single_line(tmp_path, capsys):
    """ERROR는 stderr에 '[ERROR] 시각 메시지 첫 줄' 형태로만 출력, 이후 줄(SQL문)은 파일 전용."""
    _clear_logger("test_logger_f")
    log_dir = tmp_path / "logs"
    logger = get_daily_file_logger("test_logger_f", log_dir, "sql_load.log")

    logger.error("[rts_eqpconvplan_inf.sql] FAILED: ORA-00942\nINSERT INTO RTS_EQPCONVPLAN_INF (...)")

    err = capsys.readouterr().err.strip()
    import re
    assert re.match(
        r"^\[ERROR\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[rts_eqpconvplan_inf\.sql\] FAILED: ORA-00942$",
        err,
    )
    assert "INSERT INTO" not in err

    file_text = (log_dir / "sql_load.log").read_text(encoding="utf-8")
    assert "INSERT INTO RTS_EQPCONVPLAN_INF" in file_text


def test_info_is_not_echoed_to_terminal(tmp_path, capsys):
    _clear_logger("test_logger_g")
    log_dir = tmp_path / "logs"
    logger = get_daily_file_logger("test_logger_g", log_dir, "sql_load.log")

    logger.info("rows=1\nINSERT INTO RTS_RSLT_MAS (...)")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
