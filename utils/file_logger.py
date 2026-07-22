"""
utils/file_logger.py – 자정 기준 자동 회전(rotate) 파일 로거 공통 유틸

logs/ 아래 각 로거는 당일치 파일 + 자정 회전 백업 3일치까지만 보관하고,
그보다 오래된 백업은 자동 삭제되어 디스크 사용량이 계속 늘어나지 않는다.
장기 실행 프로세스(API 서버 등)가 자정을 넘겨도 재시작 없이 자동으로
회전·정리된다.

ERROR 이상 레코드는 파일과 별도로 터미널(stderr)에도 출력된다 —
`[ERROR] 2026-07-10 10:10:10 <메시지 첫 줄>` 형태의 한 줄 요약이며,
전체 SQL문 등 상세 내용은 파일 로그에서 확인한다.
"""
from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_LOG_FORMAT = "[%(levelname)s] %(asctime)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# 자정 회전 백업 보관 일수 — 당일 파일 외에 최근 3일치 백업만 남기고 삭제
_BACKUP_DAYS = 3


class _FirstLineFormatter(logging.Formatter):
    """터미널에는 레코드의 첫 줄(에러 요약)만 출력한다 — 이어지는 SQL문 등은 파일 전용."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).splitlines()[0]


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
) -> logging.Logger:
    """logs/{base_filename}에 기록 — 자정마다 회전, 백업 3일치만 보관.

    TimedRotatingFileHandler(when="midnight", backupCount=3)가 자정에
    {base_filename}.YYYY-MM-DD 백업을 만들고, 3일이 지난 백업은 자동 삭제한다.
    ERROR 이상은 StreamHandler(stderr)로 터미널에도 한 줄 요약이 출력된다.
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
            backupCount=_BACKUP_DAYS,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(_FirstLineFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(ch)

    return logger
