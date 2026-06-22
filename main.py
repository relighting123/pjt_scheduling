"""
main.py - 운영 CLI

사용 예:
    python main.py train --facid FAC001 --prevdays 3
    python main.py train --facid FAC001 --prevdays 3 --nodb
    python main.py train --facid FAC001 --from 20260621170000 --to 20260623170000
    python main.py validate --facid FAC001
    python main.py infer --facid FAC001
    python main.py infer --facid FAC001 --ruletimekey 20260621170000
    python main.py collect --facid FAC001 --prevdays 1 --once
    python main.py collect --facid FAC001 --once --preflight
    python -m data.collector --facid FAC001 --once --dry-run -v --debug
    python main.py db-check
    python main.py ui
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CONFIG,
    set_input_folder,
    validate_path_segment,
    resolve_train_period_range,
    resolve_infer_rule_timekey,
    list_split_folders,
)
from data.collector import (
    add_debug_arguments,
    ensure_train_folders,
    run_collector_cli,
)
from data.db_registry import diagnose_db_config, print_db_config_report
from data.loader import fetch_from_db, load_data, validate_data, preprocess
from data.loader.sql_binds import resolve_lot_cd
from agent.rl_agent import SchedulingAgent
from inference.runner import run_inference, save_result
from validation.runner import run_validation


def _load_env_data(folder: str) -> dict:
    set_input_folder(folder)
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        for e in errors:
            print(f"  [오류] {e}")
        sys.exit(1)
    env_data = preprocess(raw)
    print(
        f"  EQP: {len(env_data['eqp_ids'])}대  "
        f"LOT: {len(env_data['lots'])}개  "
        f"제품: {len(env_data['prod_keys'])}종  "
        f"공정: {len(env_data['oper_ids'])}종"
    )
    return env_data


def _load_many(folders: List[str]) -> List[dict]:
    print("=" * 60)
    print(f"[load] {len(folders)}개 기간")
    datasets = []
    for folder in folders:
        print(f"  · {folder}")
        set_input_folder(folder)
        raw = load_data()
        errors = validate_data(raw)
        if errors:
            for e in errors:
                print(f"  [오류] {e}")
            sys.exit(1)
        datasets.append(preprocess(raw))
    if datasets:
        d0 = datasets[0]
        print(
            f"  (대표) EQP: {len(d0['eqp_ids'])}대  "
            f"LOT: {len(d0['lots'])}개  "
            f"제품: {len(d0['prod_keys'])}종  "
            f"공정: {len(d0['oper_ids'])}종"
        )
    return datasets


def cmd_train(
    fac_id: str,
    prevdays: int = None,
    from_key: str = None,
    to_key: str = None,
    *,
    nodb: bool = False,
    lot_cd: str = None,
):
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    start_key, end_key = resolve_train_period_range(
        prevdays=prevdays, from_key=from_key, to_key=to_key,
    )

    print("=" * 60)
    print(f"[train] FAC={fac_id}  RULE_TIMEKEY {start_key} ~ {end_key}")
    if nodb:
        print("[train] --nodb: 기존 JSON만 사용 (자동 수집 없음)")

    train_folders = ensure_train_folders(
        fac_id,
        prevdays=prevdays,
        from_key=from_key,
        to_key=to_key,
        lot_cd=lot_cd,
        nodb=nodb,
    )
    if not train_folders:
        available = list_split_folders(fac_id, "train")
        print("[오류] 학습용 train 폴더가 없습니다.")
        print(f"  요청 구간: {start_key} ~ {end_key}")
        if available:
            print(f"  사용 가능한 train 폴더: {', '.join(available)}")
        else:
            print("  collect 로 train 데이터를 수집하거나 --nodb 없이 train 을 실행하세요.")
        sys.exit(1)

    print(f"[train] train 폴더 {len(train_folders)}개 사용")
    for f in train_folders:
        print(f"  · {f}")

    print("=" * 60)
    print("[train] 데이터 로드 및 전처리")
    env_data = _load_many(train_folders) if len(train_folders) > 1 else _load_env_data(train_folders[0])

    print("=" * 60)
    print("[train] 모델 학습")
    datasets = env_data if isinstance(env_data, list) else [env_data]
    if len(datasets) > 1:
        print(f"  학습 기간: {len(datasets)}개 (VecEnv)")
    print(f"  Total Timesteps: {CONFIG.rl.total_timesteps:,}")

    agent = SchedulingAgent()
    agent.train(env_data, verbose=1)
    agent.save()
    print(f"  모델 저장: {CONFIG.path.model_dir / CONFIG.rl.model_name}.zip")

    print("=" * 60)
    print("[train] validation (test 전체)")
    run_validation(fac_id, agent=agent, refresh_sql=not nodb)


def cmd_validate(fac_id: str, *, nodb: bool = False):
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    print("=" * 60)
    print(f"[validate] FAC={fac_id} (test 전체)")
    if nodb:
        print("[validate] --nodb: 기존 JSON 사용 (Oracle 조회 생략)")
    payload = run_validation(fac_id, refresh_sql=not nodb)
    if payload["errors"]:
        print(f"\n[validate] {len(payload['errors'])}개 폴더 오류")
        sys.exit(1)
    print(f"\n[validate] 완료 ({len(payload['results'])}개 test)")


def cmd_inference(fac_id: str, rule_timekey: str = None, *, nodb: bool = False, lot_cd: str = None):
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    rtk = resolve_infer_rule_timekey(fac_id, rule_timekey)

    print("=" * 60)
    print(f"[inference] FAC={fac_id}  RULE_TIMEKEY={rtk}")
    lcd = resolve_lot_cd(lot_cd)
    if lcd:
        print(f"[inference] LOT_CD={lcd}")
    if nodb:
        print("[inference] --nodb: 기존 JSON 사용 (Oracle 조회 생략)")
    else:
        print("[inference] Oracle SQL → JSON (infer)")
        fetch_from_db(fac_id=fac_id, split="infer", period=rtk, lot_cd=lcd)
    set_input_folder(f"{fac_id}/infer")

    agent = SchedulingAgent()
    if not agent.model_exists():
        print("[오류] 학습된 모델이 없습니다. 먼저 train을 실행하세요.")
        sys.exit(1)
    agent = SchedulingAgent.load()

    print("=" * 60)
    print("[inference] 데이터 로드 및 전처리")
    env_data = _load_env_data(f"{fac_id}/infer")

    print("=" * 60)
    print("[inference] 추론 실행")
    result = run_inference(env_data, algorithm="rl", agent=agent)
    path = save_result(result, env_data=env_data)
    stats = result["stats"]
    print(f"  배정 LOT 수:    {len(result['schedule'])}")
    print(f"  공정 전환 횟수: {stats['oper_switches']}")
    print(f"  제품 전환 횟수: {stats['prod_switches']}")
    print(f"  Idle 합계:      {stats['idle_total']}분")
    print(f"  결과 파일:      {path}")


def cmd_ui():
    print("=" * 60)
    print("[ui] React UI + API 서버 실행")
    print("  API:      http://127.0.0.1:8000")
    print("  Frontend: http://localhost:5173")
    print("  종료: Ctrl+C")

    frontend_dir = ROOT / "frontend"
    if not (frontend_dir / "node_modules").exists():
        print("\n  [ui] frontend 의존성 설치 중 (npm install)...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True, shell=True)

    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "api.server:app",
            "--host", "127.0.0.1", "--port", "8000", "--reload",
            "--reload-exclude", "models",
            "--reload-exclude", "data/dataset",
        ],
        cwd=str(ROOT),
    )

    health_url = "http://127.0.0.1:8000/api/health"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as res:
                if res.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    else:
        api_proc.terminate()
        print("[오류] API 서버가 시작되지 않았습니다.")
        sys.exit(1)

    fe_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir),
        shell=True,
        env={**os.environ, "BROWSER": "none"},
    )

    time.sleep(2)
    webbrowser.open("http://localhost:5173")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n[ui] 종료 중...")
    finally:
        fe_proc.terminate()
        api_proc.terminate()
        fe_proc.wait(timeout=5)
        api_proc.wait(timeout=5)


def cmd_collect(
    fac_id: str,
    split: str = "train",
    interval: int = 0,
    prevdays: int = 1,
    from_key: str = None,
    to_key: str = None,
    once: bool = False,
    snapshot: bool = False,
    period: str = None,
    lot_cd: str = None,
    verbose: bool = False,
    dry_run: bool = False,
    debug: bool = False,
    preflight: bool = False,
):
    args = argparse.Namespace(
        facid=fac_id,
        split=split,
        interval=interval,
        prevdays=prevdays,
        from_key=from_key,
        to_key=to_key,
        once=once,
        snapshot=snapshot,
        period=period,
        lotcd=lot_cd,
        verbose=verbose,
        dry_run=dry_run,
        debug=debug,
        preflight=preflight,
    )
    code = run_collector_cli(args)
    if code:
        sys.exit(code)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scheduling RL 운영 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="dataset train JSON 로드 → 학습 → validation")
    train_p.add_argument("--facid", required=True, help="공장 ID")
    train_p.add_argument(
        "--prevdays", type=int, default=None,
        help="현재 기준 최근 N일 train 데이터",
    )
    train_p.add_argument(
        "--from", dest="from_key", metavar="RULE_TIMEKEY", default=None,
        help="학습 시작 RULE_TIMEKEY",
    )
    train_p.add_argument(
        "--to", dest="to_key", metavar="RULE_TIMEKEY", default=None,
        help="학습 종료 RULE_TIMEKEY",
    )
    train_p.add_argument(
        "--nodb", action="store_true",
        help="자동 수집·validation DB 조회 생략, dataset 기존 JSON만 사용",
    )
    train_p.add_argument(
        "--lotcd",
        default=None,
        help="자동 수집 시 discrete_arrange LOT_CD 필터",
    )
    val_p = sub.add_parser("validate", help="test 데이터 전체 검증")
    val_p.add_argument("--facid", required=True, help="공장 ID")
    val_p.add_argument(
        "--nodb", action="store_true",
        help="Oracle 조회 생략, dataset 기존 JSON 사용",
    )

    inf_p = sub.add_parser("infer", help="SQL 조회 → 추론")
    inf_p.add_argument("--facid", required=True, help="공장 ID")
    inf_p.add_argument(
        "--ruletimekey", default=None,
        help="추론 RULE_TIMEKEY (미지정 시 최신)",
    )
    inf_p.add_argument(
        "--lotcd",
        default=None,
        help="discrete_arrange LOT_CD 필터 (기본: SQL_LOT_CD / COLLECTOR_LOT_CD)",
    )
    inf_p.add_argument(
        "--nodb", action="store_true",
        help="Oracle 조회 생략, dataset 기존 JSON 사용",
    )

    sub.add_parser("db-check", help="DB alias 설정 진단 (databases.yaml / .env)")

    sub.add_parser("ui", help="React UI + API 서버 실행")

    collect_p = sub.add_parser(
        "collect", help="주기적 학습 데이터 수집 (SQL @db alias → JSON)",
    )
    collect_p.add_argument("--facid", required=True, help="공장 ID")
    collect_p.add_argument(
        "--split", default="train", choices=("train", "test", "infer"),
    )
    collect_p.add_argument(
        "--interval", type=int, default=0,
        help="수집 주기(초). 0 또는 --once 이면 1회",
    )
    collect_p.add_argument("--prevdays", type=int, default=1)
    collect_p.add_argument("--from", dest="from_key", metavar="RULE_TIMEKEY")
    collect_p.add_argument("--to", dest="to_key", metavar="RULE_TIMEKEY")
    collect_p.add_argument("--once", action="store_true")
    collect_p.add_argument("--snapshot", action="store_true")
    collect_p.add_argument("--period", help="--snapshot 시 RULE_TIMEKEY")
    collect_p.add_argument(
        "--lotcd",
        default=None,
        help="discrete_arrange LOT_CD 필터 (기본: COLLECTOR_LOT_CD / SQL_LOT_CD)",
    )
    add_debug_arguments(collect_p)

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "train":
            if args.prevdays is None and not (args.from_key and args.to_key):
                print("[오류] --prevdays 또는 --from/--to 가 필요합니다.")
                sys.exit(1)
            if args.prevdays is not None and (args.from_key or args.to_key):
                print("[오류] --prevdays 와 --from/--to 는 함께 쓸 수 없습니다.")
                sys.exit(1)
            cmd_train(
                fac_id=args.facid,
                prevdays=args.prevdays,
                from_key=args.from_key,
                to_key=args.to_key,
                nodb=args.nodb,
                lot_cd=args.lotcd,
            )

        elif args.command == "validate":
            cmd_validate(fac_id=args.facid, nodb=args.nodb)

        elif args.command == "infer":
            cmd_inference(
                fac_id=args.facid,
                rule_timekey=args.ruletimekey,
                nodb=args.nodb,
                lot_cd=args.lotcd,
            )

        elif args.command == "collect":
            cmd_collect(
                fac_id=args.facid,
                split=args.split,
                interval=args.interval,
                prevdays=args.prevdays,
                from_key=args.from_key,
                to_key=args.to_key,
                once=args.once,
                snapshot=args.snapshot,
                period=args.period,
                lot_cd=args.lotcd,
                verbose=args.verbose,
                dry_run=args.dry_run,
                debug=args.debug,
                preflight=args.preflight,
            )

        elif args.command == "db-check":
            report = diagnose_db_config()
            print_db_config_report(report)
            if not report["ok"]:
                sys.exit(1)

        elif args.command == "ui":
            cmd_ui()

    except KeyError as e:
        print(f"[오류] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[오류] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
