"""
tests/test_file_logger.py

utils.file_logger.get_daily_file_logger()는:
  - logs/{base_filename}에 기록하는 TimedRotatingFileHandler(자정 회전)를 붙인다.
  - 이전 방식(prefix_YYYYMMDD.log)의 잔여 로그 파일은 정리한다.
  - 디스크에 파일이 무한정 쌓이지 않도록 backup_count를 강제한다.
"""
import logging
import logging.handlers

from utils.file_logger import get_daily_file_logger


def _clear_logger(name: str):
    logging.getLogger(name).handlers.clear()


def test_creates_timed_rotating_handler_with_backup_count(tmp_path):
    _clear_logger("test_logger_a")
    log_dir = tmp_path / "logs"
    logger = get_daily_file_logger("test_logger_a", log_dir, "sql_fetch.log")

    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert isinstance(handler, logging.handlers.TimedRotatingFileHandler)
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 1

    logger.info("hello")
    assert (log_dir / "sql_fetch.log").is_file()
    assert "hello" in (log_dir / "sql_fetch.log").read_text(encoding="utf-8")


def test_does_not_add_duplicate_handlers_on_repeated_calls(tmp_path):
    _clear_logger("test_logger_b")
    log_dir = tmp_path / "logs"
    get_daily_file_logger("test_logger_b", log_dir, "sql_load.log")
    logger = get_daily_file_logger("test_logger_b", log_dir, "sql_load.log")
    assert len(logger.handlers) == 1


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
