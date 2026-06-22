"""
data/collector.py – 주기적 학습 데이터 수집 (Oracle SQL → dataset JSON)

환경 변수 (선택, CLI 인자가 우선):
    COLLECTOR_FAC_ID=FAC001
    COLLECTOR_SPLIT=train
    COLLECTOR_INTERVAL_SEC=3600
    COLLECTOR_PREVDAYS=1

사용 예:
    python -m data.collector --fac-id FAC001 --once
    python -m data.collector --fac-id FAC001 --interval 3600
    python main.py collect --fac-id FAC001 --prevdays 3 --once
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    resolve_train_period_range,
    rule_timekey_now,
    validate_path_segment,
)
from data.loader.fetch import fetch_from_db, fetch_period_range


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


class TrainingDataCollector:
    """FAC_ID·RULE_TIMEKEY 기준 학습 입력 JSON 수집."""

    def __init__(
        self,
        fac_id: str,
        split: str = "train",
        prevdays: int = 1,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
    ):
        self.fac_id = validate_path_segment(fac_id, "FAC_ID")
        self.split = validate_path_segment(split, "split")
        self.prevdays = prevdays
        self.from_key = from_key
        self.to_key = to_key

    def _resolve_range(self) -> tuple[str, str]:
        if self.from_key and self.to_key:
            return self.from_key, self.to_key
        start, end = resolve_train_period_range(prevdays=self.prevdays)
        return start, end

    def collect_period_range(
        self,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
    ) -> List[Path]:
        start, end = from_key or self.from_key, to_key or self.to_key
        if not start or not end:
            start, end = self._resolve_range()
        print(
            f"[collector] {self.fac_id}/{self.split} "
            f"RULE_TIMEKEY {start} ~ {end}",
        )
        return fetch_period_range(
            fac_id=self.fac_id,
            from_timekey=start,
            to_timekey=end,
            split=self.split,
        )

    def collect_snapshot(self, period: Optional[str] = None) -> Path:
        per = period or rule_timekey_now()
        print(f"[collector] {self.fac_id}/{self.split}/{per} 단일 스냅샷")
        return fetch_from_db(
            fac_id=self.fac_id,
            split=self.split,
            period=per,
        )

    def collect_once(
        self,
        *,
        snapshot_only: bool = False,
        period: Optional[str] = None,
    ) -> List[Path]:
        """1회 수집. snapshot_only=True 이면 현재 시각 1폴더만."""
        if snapshot_only:
            return [self.collect_snapshot(period=period)]
        return self.collect_period_range()

    def run_loop(
        self,
        interval_sec: int,
        *,
        snapshot_only: bool = False,
    ) -> None:
        """interval_sec 마다 collect_once 반복."""
        if interval_sec < 1:
            raise ValueError("interval_sec 는 1 이상이어야 합니다.")
        print(
            f"[collector] 주기 수집 시작 – every {interval_sec}s "
            f"fac={self.fac_id} split={self.split}",
        )
        while True:
            started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[collector] tick {started}")
            try:
                paths = self.collect_once(snapshot_only=snapshot_only)
                print(f"[collector] 완료 – {len(paths)}개 폴더")
            except Exception as exc:
                print(f"[collector] 오류: {exc}")
            time.sleep(interval_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="주기적 학습 데이터 수집 (SQL @db alias → JSON)",
    )
    parser.add_argument(
        "--fac-id",
        default=_env_str("COLLECTOR_FAC_ID", "FAC001"),
        help="FAB ID (기본: COLLECTOR_FAC_ID 또는 FAC001)",
    )
    parser.add_argument(
        "--split",
        default=_env_str("COLLECTOR_SPLIT", "train"),
        choices=("train", "test", "infer"),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=_env_int("COLLECTOR_INTERVAL_SEC", 3600),
        help="수집 주기(초). --once 와 함께 쓰면 1회만 실행",
    )
    parser.add_argument(
        "--prevdays",
        type=int,
        default=_env_int("COLLECTOR_PREVDAYS", 1),
        help="최근 N일 RULE_TIMEKEY 구간 수집",
    )
    parser.add_argument("--from", dest="from_key", help="시작 RULE_TIMEKEY")
    parser.add_argument("--to", dest="to_key", help="종료 RULE_TIMEKEY")
    parser.add_argument(
        "--once",
        action="store_true",
        help="1회만 수집 후 종료",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="현재 시각 단일 스냅샷만 수집 (--prevdays 무시)",
    )
    parser.add_argument(
        "--period",
        help="--snapshot 시 사용할 RULE_TIMEKEY (미지정 시 now)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    collector = TrainingDataCollector(
        fac_id=args.fac_id,
        split=args.split,
        prevdays=args.prevdays,
        from_key=args.from_key,
        to_key=args.to_key,
    )
    if args.once or args.interval <= 0:
        collector.collect_once(
            snapshot_only=args.snapshot,
            period=args.period,
        )
        return 0
    collector.run_loop(args.interval, snapshot_only=args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
