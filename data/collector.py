"""
data/collector.py – 주기적 학습 데이터 수집 (Oracle SQL → dataset JSON)

환경 변수 (선택, CLI 인자가 우선):
    COLLECTOR_FAC_ID=FAC001
    COLLECTOR_SPLIT=train
    COLLECTOR_INTERVAL_SEC=3600
    COLLECTOR_PREVDAYS=1
    COLLECTOR_LOT_CD=LC001   # 선택: SQL :LOT_CD 바인드 (discrete_arrange 제외)

사용 예:
    python -m data.collector --facid FAC001 --once
    python -m data.collector --facid FAC001 --once --lotcd LC001
    python -m data.collector --facid FAC001 --once --preflight
    python -m data.collector --facid FAC001 --once --dry-run -v
    python -m data.collector --facid FAC001 --once --debug
    python main.py collect --facid FAC001 --prevdays 3 --once

RULE_TIMEKEY (DB 메타 SQL 필수, external/sql/rule_timekey_*.sql):
    수집 폴더명 = DB 에서 조회한 실제 RULE_TIMEKEY (로컬 시각 생성 없음)
    rule_timekey_latest.sql  – 최신 1건 (--snapshot 기본값)
    rule_timekey_recent.sql  – 최근 N개 (--prevdays)
    rule_timekey_list.sql    – FROM~TO 구간 (--from/--to)
    cp external/sql.example/rule_timekey_*.sql external/sql/

디버그 순서 (오류 시):
    1. python main.py db-check
    2. python -m data.collector --facid FAC001 --once --preflight
    3. python -m data.collector --facid FAC001 --once --dry-run -v
    4. python -m data.collector --facid FAC001 --once --debug
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CONFIG,
    list_split_folders,
    resolve_dataset_path,
    resolve_train_folders,
    resolve_train_period_range,
    validate_path_segment,
)
from data.db_registry import diagnose_db_config, print_db_config_report
from data.loader.fetch import fetch_from_db, fetch_period_range
from data.loader.sql_binds import merge_fetch_binds, resolve_lot_cd
from data.loader.rule_timekey_query import (
    resolve_collect_periods,
    resolve_snapshot_rule_timekey,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


@dataclass
class CollectorOptions:
    verbose: bool = False
    dry_run: bool = False
    debug: bool = False
    preflight: bool = False


def add_debug_arguments(parser: argparse.ArgumentParser) -> None:
    """collector / main.py collect 공용 디버그 플래그."""
    group = parser.add_argument_group("디버그")
    group.add_argument(
        "--preflight",
        action="store_true",
        help="DB·SQL·수집 경로만 점검 (Oracle 연결 없음)",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="수집 계획·SQL alias 확인만 (DB 연결·JSON 저장 없음)",
    )
    group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="단계별 상세 로그",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="오류 시 전체 traceback 출력",
    )


def collector_options_from_args(args: argparse.Namespace) -> CollectorOptions:
    return CollectorOptions(
        verbose=getattr(args, "verbose", False),
        dry_run=getattr(args, "dry_run", False),
        debug=getattr(args, "debug", False),
        preflight=getattr(args, "preflight", False),
    )


def paths_to_folder_keys(fac_id: str, split: str, paths: List[Path]) -> List[str]:
    """수집 output 경로 → dataset 폴더 키 (FAC/split/RULE_TIMEKEY)."""
    fac = validate_path_segment(fac_id, "FAC_ID")
    sp = validate_path_segment(split, "split")
    return [f"{fac}/{sp}/{path.parent.name}" for path in paths]


def collect_dataset(
    fac_id: str,
    *,
    split: str = "train",
    prevdays: int = 1,
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
    lot_cd: Optional[str] = None,
    options: Optional[CollectorOptions] = None,
) -> List[Path]:
    """Oracle SQL → dataset JSON 수집 (collect/train 공용 진입점)."""
    collector = TrainingDataCollector(
        fac_id=fac_id,
        split=split,
        prevdays=prevdays,
        from_key=from_key,
        to_key=to_key,
        lot_cd=lot_cd,
    )
    options = options or CollectorOptions()
    if from_key and to_key:
        return collector.collect_period_range(
            from_key=from_key,
            to_key=to_key,
            options=options,
        )
    return collector.collect_period_range(options=options)


def ensure_train_folders(
    fac_id: str,
    *,
    prevdays: Optional[int] = None,
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
    lot_cd: Optional[str] = None,
    nodb: bool = False,
) -> List[str]:
    """
    train 학습용 폴더 목록 확보.
    로컬 dataset 이 없고 nodb=False 이면 TrainingDataCollector 로 DB 수집.
    """
    start_key, end_key = resolve_train_period_range(
        prevdays=prevdays, from_key=from_key, to_key=to_key,
    )
    folders = resolve_train_folders(fac_id, start_key, end_key, prevdays=prevdays)
    if folders:
        return folders
    if nodb:
        return []

    print("[train] train 데이터 없음 → collector 수집")
    paths = collect_dataset(
        fac_id,
        split="train",
        prevdays=prevdays or 1,
        from_key=from_key,
        to_key=to_key,
        lot_cd=lot_cd,
    )
    if not paths:
        return []

    folders_after = resolve_train_folders(fac_id, start_key, end_key, prevdays=prevdays)
    if folders_after:
        return folders_after
    return paths_to_folder_keys(fac_id, "train", paths)


class TrainingDataCollector:
    """FAC_ID·RULE_TIMEKEY 기준 학습 입력 JSON 수집."""

    def __init__(
        self,
        fac_id: str,
        split: str = "train",
        prevdays: int = 1,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
        lot_cd: Optional[str] = None,
    ):
        self.fac_id = validate_path_segment(fac_id, "FAC_ID")
        self.split = validate_path_segment(split, "split")
        self.prevdays = prevdays
        self.from_key = from_key
        self.to_key = to_key
        self.lot_cd = resolve_lot_cd(lot_cd)

    def _resolve_periods(self) -> tuple[List[str], str]:
        periods, source = resolve_collect_periods(
            self.fac_id,
            prevdays=self.prevdays,
            from_key=self.from_key,
            to_key=self.to_key,
            require_db=True,
        )
        if not periods:
            raise ValueError("수집할 RULE_TIMEKEY 가 없습니다.")
        return periods, source

    def _fetch_kwargs(self, options: CollectorOptions) -> dict:
        return {
            "verbose": options.verbose or options.dry_run or options.preflight,
            "dry_run": options.dry_run or options.preflight,
        }

    def run_preflight(
        self,
        *,
        snapshot_only: bool = False,
        period: Optional[str] = None,
        options: Optional[CollectorOptions] = None,
    ) -> List[Path]:
        """DB 설정·SQL·출력 경로 사전 점검."""
        options = options or CollectorOptions(preflight=True)
        print("[preflight] === 1) DB 설정 ===")
        report = diagnose_db_config()
        print_db_config_report(report)

        print("[preflight] === 2) 수집 계획 ===")
        print(f"  fac_id={self.fac_id}  split={self.split}")
        if self.lot_cd:
            print(f"  lot_cd={self.lot_cd}")
        print(f"  sql_dir={CONFIG.path.sql_dir}")

        fetch_kwargs = self._fetch_kwargs(options)
        if snapshot_only:
            per, source = resolve_snapshot_rule_timekey(
                self.fac_id, period, require_db=True,
            )
            out_dir, _ = resolve_dataset_path(self.fac_id, self.split, per)
            print(f"  mode=snapshot  period={per} ({source})")
            print(f"  output_dir={out_dir}")
            fetch_from_db(
                fac_id=self.fac_id,
                split=self.split,
                period=per,
                lot_cd=self.lot_cd,
                **fetch_kwargs,
            )
            return []

        periods, source = self._resolve_periods()
        sample = periods[0]
        print(
            f"  mode=range  RULE_TIMEKEY {periods[0]} ~ {periods[-1]} "
            f"({source}, {len(periods)}건)",
        )
        out_dir, _ = resolve_dataset_path(self.fac_id, self.split, sample)
        print(f"  sample_period={sample}")
        print(f"  sample_output_dir={out_dir}")
        fetch_from_db(
            fac_id=self.fac_id,
            split=self.split,
            period=sample,
            lot_cd=self.lot_cd,
            **fetch_kwargs,
        )
        print("[preflight] 완료 – 문제 없으면 --once 로 실제 수집하세요.")
        return []

    def collect_period_range(
        self,
        from_key: Optional[str] = None,
        to_key: Optional[str] = None,
        *,
        options: Optional[CollectorOptions] = None,
    ) -> List[Path]:
        options = options or CollectorOptions()
        if from_key and to_key:
            periods, source = resolve_collect_periods(
                self.fac_id,
                from_key=from_key,
                to_key=to_key,
                require_db=True,
            )
        else:
            periods, source = self._resolve_periods()
        print(
            f"[collector] {self.fac_id}/{self.split} "
            f"RULE_TIMEKEY {periods[0]} ~ {periods[-1]} ({source}, {len(periods)}건)",
        )
        if options.verbose:
            print(f"[collector] sql_dir={CONFIG.path.sql_dir}")
        return fetch_period_range(
            fac_id=self.fac_id,
            split=self.split,
            periods=periods,
            lot_cd=self.lot_cd,
            **self._fetch_kwargs(options),
        )

    def collect_snapshot(
        self,
        period: Optional[str] = None,
        *,
        options: Optional[CollectorOptions] = None,
    ) -> Path:
        options = options or CollectorOptions()
        per, source = resolve_snapshot_rule_timekey(
            self.fac_id, period, require_db=True,
        )
        print(f"[collector] {self.fac_id}/{self.split}/{per} 단일 스냅샷 ({source})")
        if options.verbose:
            out_dir, _ = resolve_dataset_path(self.fac_id, self.split, per)
            print(f"[collector] output_dir={out_dir}")
        return fetch_from_db(
            fac_id=self.fac_id,
            split=self.split,
            period=per,
            lot_cd=self.lot_cd,
            **self._fetch_kwargs(options),
        )

    def collect_once(
        self,
        *,
        snapshot_only: bool = False,
        period: Optional[str] = None,
        options: Optional[CollectorOptions] = None,
    ) -> List[Path]:
        """1회 수집. snapshot_only=True 이면 현재 시각 1폴더만."""
        options = options or CollectorOptions()
        if options.preflight:
            return self.run_preflight(
                snapshot_only=snapshot_only,
                period=period,
                options=options,
            )
        if snapshot_only:
            return [self.collect_snapshot(period=period, options=options)]
        return self.collect_period_range(options=options)

    def run_loop(
        self,
        interval_sec: int,
        *,
        snapshot_only: bool = False,
        options: Optional[CollectorOptions] = None,
    ) -> None:
        """interval_sec 마다 collect_once 반복."""
        options = options or CollectorOptions()
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
                paths = self.collect_once(
                    snapshot_only=snapshot_only,
                    options=options,
                )
                print(f"[collector] 완료 – {len(paths)}개 폴더")
            except Exception as exc:
                print(f"[collector] 오류: {exc}")
                if options.debug:
                    traceback.print_exc()
            time.sleep(interval_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="주기적 학습 데이터 수집 (SQL @db alias → JSON)",
    )
    parser.add_argument(
        "--facid",
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
        "--lotcd",
        default=os.environ.get("COLLECTOR_LOT_CD", "").strip() or None,
        help="SQL :LOT_CD 바인드 (discrete_arrange 제외, 기본: COLLECTOR_LOT_CD)",
    )
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
        help="--snapshot 시 사용할 RULE_TIMEKEY (미지정 시 DB 최신 → 현재 시각)",
    )
    add_debug_arguments(parser)
    return parser


def run_collector_cli(args: argparse.Namespace) -> int:
    options = collector_options_from_args(args)
    collector = TrainingDataCollector(
        fac_id=args.facid,
        split=args.split,
        prevdays=args.prevdays,
        from_key=args.from_key,
        to_key=args.to_key,
        lot_cd=args.lotcd,
    )
    try:
        if args.once or args.interval <= 0:
            collector.collect_once(
                snapshot_only=args.snapshot,
                period=args.period,
                options=options,
            )
        else:
            collector.run_loop(
                args.interval,
                snapshot_only=args.snapshot,
                options=options,
            )
    except Exception as exc:
        print(f"[collector] 오류: {exc}", file=sys.stderr)
        if options.debug:
            traceback.print_exc()
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    return run_collector_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
