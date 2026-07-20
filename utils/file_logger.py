"""
utils/file_logger.py – 자정 기준 자동 회전(rotate) 파일 로거 공통 유틸

logs/ 아래 각 로거는 당일치 파일 하나만 남기고, 자정 회전 시 이전 내용은
백업 없이 바로 삭제되어 디스크 사용량이 하루치 이상으로 늘어나지 않는다.
장기 실행 프로세스(API 서버 등)가 자정을 넘겨도 재시작 없이 자동으로
회전·삭제된다.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path


def _cleanup_legacy_daily_logs(log_dir: Path, prefix: str) -> None:
    """이전 방식(prefix_YYYYMMDD.log, 파일당 하루치 고정 이름)의 잔여 로그 파일을 정리."""
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{8}}\.log$")
    for f in log_dir.glob(f"{prefix}_*.log"):
        if pattern.match(f.name):
            try:
                f.unlink()
            except OSError:
                pass


def _discard_rotated_log(source: str, dest: str) -> None:
    """자정 회전 시 백업(dest)을 만들지 않고 이전 로그(source)를 바로 삭제한다.

    TimedRotatingFileHandler.rotator 훅으로 등록 — 기본 동작(rename)을 대신해
    파일을 지우기만 하므로 오래된 로그가 하루치를 넘어 쌓이지 않는다.
    """
    try:
        os.remove(source)
    except OSError:
        pass


def get_daily_file_logger(
    logger_name: str,
    log_dir: Path,
    base_filename: str,
) -> logging.Logger:
    """logs/{base_filename}에 당일치만 기록 — 자정마다 회전하며 이전 내용은 삭제.

    TimedRotatingFileHandler(when="midnight")가 자정에 회전을 트리거하되,
    rotator 훅으로 백업 파일 생성을 건너뛰고 이전 로그를 즉시 삭제한다.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = base_filename.rsplit(".", 1)[0]
    _cleanup_legacy_daily_logs(log_dir, prefix)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.handlers.TimedRotatingFileHandler(
            log_dir / base_filename,
            when="midnight",
            backupCount=1,
            encoding="utf-8",
        )
        fh.rotator = _discard_rotated_log
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger
