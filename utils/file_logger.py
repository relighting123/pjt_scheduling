"""
utils/file_logger.py – 자정 기준 자동 회전(rotate) 파일 로거 공통 유틸

logs/ 아래 각 로거는 당일 파일 하나 + 최소 백업 파일만 남기고 자동 삭제되어
디스크 사용량이 무한정 늘어나지 않는다. 장기 실행 프로세스(API 서버 등)가
자정을 넘겨도 재시작 없이 자동으로 회전·정리된다.
"""
from __future__ import annotations

import logging
import logging.handlers
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


def get_daily_file_logger(
    logger_name: str,
    log_dir: Path,
    base_filename: str,
    *,
    backup_count: int = 1,
) -> logging.Logger:
    """logs/{base_filename}에 기록, 자정마다 회전하고 backup_count개까지만 보관.

    TimedRotatingFileHandler(when="midnight")가 자정에 파일을 회전시키고
    backup_count를 넘는 과거 파일은 자동 삭제한다.
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
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger
