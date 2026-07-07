"""
main.py - 운영 CLI

사용 예:
    python main.py train --facid FAC001 --prevcnt 3
    python main.py train --facid FAC001 --prevcnt 3 --db
    python main.py train --facid FAC001 --from 20260621170000 --to 20260623170000
    python main.py test --facid FAC001
    python main.py test --facid FAC001 --prevcnt 3 --lotcd LC001
    python main.py test --facid FAC001 --from 20260621170000 --to 20260623170000
    python main.py infer --facid FAC001
    python main.py infer --facid FAC001 --ruletimekey 20260621170000
    python main.py infer --facid FAC001 --from 20260621170000 --to 20260623170000
    python main.py infer --facid FAC001 --prevcnt 3
    python main.py infer --facid FAC001 --nodb --decision-log
    python main.py collect --facid FAC001 --prevcnt 1 --once
    python main.py collect --facid FAC001 --once --preflight
    python -m data.collector --facid FAC001 --once --dry-run -v --debug
    python main.py db-check
    python main.py db-load --ddl-only
    python main.py db-load --facid FAC001 --split infer
    python main.py infer --facid FAC001 --db-load
    python main.py sample --facid FAC001 --bootstrap
    python main.py sample --facid FAC001 --split test --period 20260621070000
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
    normalize_rule_timekey,
    resolve_train_period_range,
    resolve_infer_rule_timekey,
    list_split_folders,
)
from data.collector import (
    add_debug_arguments,
    ensure_test_folders,
    ensure_train_folders,
    run_collector_cli,
)
from data.db_registry import diagnose_db_config, print_db_config_report
from data.generator import generate_sample, list_sample_scenarios
from data.loader import fetch_from_db, load_data, validate_data, preprocess
from data.loader.rule_timekey_query import resolve_collect_periods
from data.loader.sql_binds import resolve_lot_cd
from agent.rl_agent import SchedulingAgent
from agent.training_report import save_training_convergence_report
from inference.runner import run_inference, save_result
from data.writer.db_load import (
    apply_output_ddl,
    load_output_json,
    load_output_sql_files,
    resolve_output_dir,
)
from validation.runner import run_validation
from validation.output_checks import validate_schedule_output


def _refresh_invalid_folders(
    fac_id: str,
    folders: List[str],
    lot_cd: str | None = None,
) -> List[str]:
    """
    폴더별 validate_data() 검사 → 실패한 폴더만 DB 재수집 후 유효 폴더 목록 반환.
    유효한 폴더는 기존 JSON을 그대로 사용하고 재수집하지 않는다.
    """
    from config import parse_input_folder, resolve_dataset_path

    valid: List[str] = []
    to_refetch: List[str] = []

    print("[train] 폴더별 데이터 유효성 검사")
    for folder in folders:
        set_input_folder(folder)
        try:
            raw = load_data()
            errors = validate_data(raw)
        except Exception as exc:
            errors = [str(exc)]

        if errors:
            print(f"  [재수집 필요] {folder}")
            for e in errors:
                print(f"    · {e}")
            to_refetch.append(folder)
        else:
            print(f"  [OK] {folder}")
            valid.append(folder)

    if not to_refetch:
        print("[train] 모든 폴더 유효 – 재수집 없음")
        return valid

    print(f"\n[train] {len(to_refetch)}개 폴더 재수집 시작")
    for folder in to_refetch:
        _, split, period = parse_input_folder(folder)
        out_dir, _ = resolve_dataset_path(fac_id, split, period)
        try:
            fetch_from_db(
                fac_id=fac_id,
                split=split,
                period=period,
                lot_cd=lot_cd,
                output_dir=out_dir,
                verbose=True,
            )
            # 재수집 후 재검증
            set_input_folder(folder)
            raw = load_data()
            errors = validate_data(raw)
            if errors:
                print(f"  [경고] 재수집 후에도 오류 – 학습에서 제외: {folder}")
                for e in errors:
                    print(f"    · {e}")
            else:
                print(f"  [재수집 완료] {folder}")
                valid.append(folder)
        except Exception as exc:
            print(f"  [오류] 재수집 실패 – 학습에서 제외: {folder}\n    {exc}")

    return valid


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


def _validate_period_selectors(
    args: argparse.Namespace,
    *,
    has_ruletimekey: bool = False,
    require_one: bool = False,
    one_of_label: str = "--prevcnt, --from/--to",
) -> None:
    """train/test/infer 공통: --ruletimekey(있으면)/--prevcnt/--from,--to 상호배타 검증."""
    rtk = getattr(args, "ruletimekey", None) if has_ruletimekey else None
    if bool(args.from_key) != bool(args.to_key):
        print("[오류] --from 와 --to 를 함께 지정하세요.")
        sys.exit(1)
    if rtk and (args.prevcnt is not None or args.from_key or args.to_key):
        print("[오류] --ruletimekey 는 --prevcnt, --from/--to 와 함께 쓸 수 없습니다.")
        sys.exit(1)
    if args.prevcnt is not None and args.from_key and args.to_key:
        print("[오류] --prevcnt 와 --from/--to 는 함께 쓸 수 없습니다.")
        sys.exit(1)
    if require_one and rtk is None and args.prevcnt is None and not (args.from_key and args.to_key):
        print(f"[오류] {one_of_label} 중 하나가 필요합니다.")
        sys.exit(1)


def cmd_train(
    fac_id: str,
    prevcnt: int = None,
    from_key: str = None,
    to_key: str = None,
    rule_timekey: str = None,
    *,
    nodb: bool = True,
    lot_cd: str = None,
    all_folders: bool = False,
    algorithm: str = "scheduling_rl",
):
    fac_id = validate_path_segment(fac_id, "FAC_ID")

    if all_folders:
        train_folders = list_split_folders(fac_id, "train")
        print("=" * 60)
        print(f"[train] FAC={fac_id}  --all: train 폴더 전체 {len(train_folders)}개")
        if nodb:
            print("[train] dataset JSON 사용 (Oracle 조회·자동 수집 없음)")
        if not train_folders:
            print("[오류] train 폴더가 없습니다. collect 로 데이터를 먼저 수집하세요.")
            sys.exit(1)
    else:
        if rule_timekey:
            key = normalize_rule_timekey(rule_timekey)
            start_key = end_key = key
            range_source = "cli"
        elif nodb:
            start_key, end_key = resolve_train_period_range(
                prevcnt=prevcnt, from_key=from_key, to_key=to_key,
            )
            range_source = "local"
        else:
            periods, range_source = resolve_collect_periods(
                fac_id,
                prevcnt=prevcnt or 1,
                from_key=from_key,
                to_key=to_key,
                require_db=True,
            )
            start_key, end_key = periods[0], periods[-1]

        print("=" * 60)
        print(
            f"[train] FAC={fac_id}  RULE_TIMEKEY {start_key} ~ {end_key}"
            f" ({range_source})",
        )
        if nodb:
            print("[train] dataset JSON 사용 (Oracle 조회·자동 수집 없음)")
        else:
            print("[train] Oracle RULE_TIMEKEY 조회·자동 수집 사용 (--db)")

        train_folders = ensure_train_folders(
            fac_id,
            prevcnt=prevcnt,
            from_key=from_key,
            to_key=to_key,
            period=rule_timekey,
            lot_cd=lot_cd,
            nodb=nodb,
        )
    if not train_folders:
        available = list_split_folders(fac_id, "train")
        print("[오류] 학습용 train 폴더가 없습니다.")
        if not all_folders:
            print(f"  요청 구간: {start_key} ~ {end_key}")
        if available:
            print(f"  사용 가능한 train 폴더: {', '.join(available)}")
        else:
            print("  collect 로 train 데이터를 수집하거나 --db 로 Oracle 조회·수집을 사용하세요.")
        sys.exit(1)

    print(f"[train] train 폴더 {len(train_folders)}개 사용")
    for f in train_folders:
        print(f"  · {f}")

    print("=" * 60)
    if not nodb:
        train_folders = _refresh_invalid_folders(fac_id, train_folders, lot_cd=lot_cd)
        if not train_folders:
            print("[오류] 유효한 train 폴더가 없습니다. 재수집 결과를 확인하세요.")
            sys.exit(1)

    print("=" * 60)
    print("[train] 데이터 로드 및 전처리")
    env_data = _load_many(train_folders) if len(train_folders) > 1 else _load_env_data(train_folders[0])

    print("=" * 60)
    print("[train] 모델 학습")
    datasets = env_data if isinstance(env_data, list) else [env_data]
    if len(datasets) > 1:
        print(f"  학습 기간: {len(datasets)}개 (VecEnv)")
    print(f"  Total Timesteps: {CONFIG.rl.total_timesteps:,}")

    from env.scheduling_rl_env import SchedulingRLEnv
    env_cls = SchedulingRLEnv
    print("  알고리즘: scheduling_rl (SchedulingRLEnv)")

    agent = SchedulingAgent()
    agent.train(env_data, verbose=1, env_cls=env_cls)
    agent.save()
    print(f"  모델 저장: {CONFIG.path.model_dir / CONFIG.rl.model_name}.zip")

    report = save_training_convergence_report(CONFIG.path.model_dir, algorithm=algorithm)
    print(f"  수렴 리포트: {report['json_path']}")
    if report["png_path"]:
        print(f"  수렴 차트: {report['png_path']}")
    print(f"  판정: {report['verdict']} — {report['note']}")

    print("=" * 60)
    print("[train] validation (test 전체)")
    run_validation(fac_id, agent=agent, refresh_sql=not nodb)


def cmd_test(
    fac_id: str,
    prevcnt: int = None,
    from_key: str = None,
    to_key: str = None,
    *,
    nodb: bool = False,
    lot_cd: str = None,
):
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    print("=" * 60)

    folders = None
    if prevcnt is not None or (from_key and to_key) or lot_cd:
        folders = ensure_test_folders(
            fac_id,
            prevcnt=prevcnt,
            from_key=from_key,
            to_key=to_key,
            lot_cd=lot_cd,
            nodb=nodb,
        )
        print(f"[test] FAC={fac_id}  test 폴더 {len(folders)}개")
        for f in folders:
            print(f"  · {f}")
        if not folders:
            print("[오류] 조건에 맞는 test 폴더가 없습니다.")
            sys.exit(1)
    else:
        print(f"[test] FAC={fac_id} (test 전체)")

    if nodb:
        print("[test] --nodb: 기존 JSON 사용 (Oracle 조회 생략)")
    payload = run_validation(fac_id, folders=folders, lot_cd=lot_cd, refresh_sql=not nodb)
    if payload["errors"]:
        print(f"\n[test] {len(payload['errors'])}개 폴더 오류")
        sys.exit(1)
    print(f"\n[test] 완료 ({len(payload['results'])}개 test)")


def cmd_db_load(
    *,
    ddl_only: bool = False,
    apply_ddl: bool = False,
    fac_id: str = None,
    split: str = "infer",
    period: str = None,
    output_dir: str = None,
    db_alias: str = None,
    json_path: str = None,
    no_history: bool = False,
    regenerate_sql: bool = False,
):
    if ddl_only or apply_ddl:
        apply_output_ddl(db_alias=db_alias)

    if ddl_only:
        return

    include_history = not no_history

    if json_path:
        load_output_json(
            json_path,
            db_alias=db_alias,
            include_history=include_history,
        )
        return

    if not fac_id and not output_dir:
        raise ValueError(
            "적재 대상이 필요합니다: --facid/--split, --output-dir, 또는 --json 중 하나"
        )

    out = resolve_output_dir(
        fac_id=fac_id or CONFIG.path.fac_id,
        split=split,
        period=period,
        output_dir=Path(output_dir) if output_dir else None,
    )
    load_output_sql_files(
        out,
        db_alias=db_alias,
        include_history=include_history,
        regenerate_sql=regenerate_sql,
    )


def cmd_inference(
    fac_id: str,
    rule_timekey: str = None,
    from_key: str = None,
    to_key: str = None,
    prevcnt: int = None,
    *,
    nodb: bool = False,
    lot_cd: str = None,
    algorithm: str = "scheduling_rl",
    decision_log: bool = False,
    enable_wip_inflow: bool = False,
    include_history: bool = False,
    db_load: bool = False,
    db_alias: str = None,
    no_history: bool = False,
    max_conversions: int = None,
    max_conversions_per_eqp: int = None,
    conversion_minutes: int = None,
    strict_validate: bool = False,
):
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    rtk = resolve_infer_rule_timekey(
        fac_id, rule_timekey,
        from_key=from_key, to_key=to_key, prevcnt=prevcnt, nodb=nodb,
    )

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

    try:
        agent = SchedulingAgent.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    print("=" * 60)
    print("[inference] 데이터 로드 및 전처리")
    env_data = _load_env_data(f"{fac_id}/infer")

    print("=" * 60)
    print("[inference] 추론 실행")
    print(
        "[inference] 유입 재공 이벤트: "
        + ("ON (다음 공정 flow 유입)" if enable_wip_inflow else "OFF (현재 재공만)")
    )
    if decision_log:
        print("[inference] 결정 로그: ON (step별 EQP/PPK/OPER·미할당 사유)")
    if max_conversions is not None:
        print(f"[inference] 전환 상한(전체): {max_conversions}")
    if max_conversions_per_eqp is not None:
        print(f"[inference] 전환 상한(EQP별): {max_conversions_per_eqp}")
    if conversion_minutes is not None:
        print(f"[inference] 전환 소요 시간: {conversion_minutes}분")
    result = run_inference(
        env_data,
        algorithm=algorithm,
        agent=agent,
        record_history=include_history,
        record_decision_log=decision_log,
        enable_wip_inflow=enable_wip_inflow,
        max_conversions=max_conversions,
        max_conversions_per_eqp=max_conversions_per_eqp,
        conversion_minutes=conversion_minutes,
    )
    print("=" * 60)
    print("[inference] 결과 검증 (장비 투입 가능성 · 처리시간 · 배정 완전성)")
    validation = validate_schedule_output(result, env_data)
    result["validation"] = validation
    summary = validation["summary"]
    if validation["ok"]:
        print(f"  [OK] 이상 없음 (배정 {summary['total_scheduled']}건)")
    else:
        print(
            f"  [경고] 투입불가 {summary['eligibility_violation_count']}건 · "
            f"처리시간불일치 {summary['proc_time_mismatch_count']}건 · "
            f"미배정LOT {summary['unassigned_lot_count']}건"
        )
        for v in validation["eligibility_violations"][:10]:
            print(f"    · [투입불가] LOT={v['lot_id']} EQP={v['eqp_id']} OPER={v['oper_id']}")
        for m in validation["proc_time_mismatches"][:10]:
            print(
                f"    · [처리시간불일치] LOT={m['lot_id']} EQP={m['eqp_id']} "
                f"기대={m['expected_proc_time']}분 실제={m['actual_proc_time']}분"
            )
        for u in validation["unassigned_lots"][:10]:
            print(f"    · [미배정] LOT={u['lot_id']} PPK={u['PLAN_PROD_ATTR_VAL']} OPER={u['oper_id']}")

    path = save_result(result, env_data=env_data)
    stats = result["stats"]
    print(f"  배정 LOT 수:    {len(result['schedule'])}")
    print(f"  공정 전환 횟수: {stats['oper_switches']}")
    print(f"  제품 전환 횟수: {stats['prod_switches']}")
    print(f"  Idle 합계:      {stats['idle_total']}분")
    print(f"  결과 파일:      {path}")
    if decision_log:
        log = result.get("decision_log", [])
        print(f"  결정 로그:      {len(log)}건 → result_full.json 의 decision_log")

    if strict_validate and not validation["ok"]:
        print("[오류] --strict-validate: 검증 실패로 종료합니다.")
        sys.exit(1)

    if db_load:
        print("=" * 60)
        print("[inference] Oracle output 적재")
        load_output_sql_files(
            CONFIG.path.output_dir,
            db_alias=db_alias,
            include_history=not no_history,
        )


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


def cmd_sample(
    fac_id: str,
    split: str = "train",
    scenario: str = "default",
    *,
    bootstrap: bool = False,
    period: str = None,
    from_key: str = None,
    to_key: str = None,
    use_period_count: bool = False,
):
    print("=" * 60)
    print(f"[sample] 샘플 데이터 생성 (FAC: {fac_id}, split: {split}, 시나리오: {scenario})")
    if bootstrap:
        print("  모드: bootstrap (train/test/infer 일괄 생성)")
    elif period:
        print(f"  기간: {period}")
    elif from_key and to_key:
        print(f"  기간 범위: {from_key} ~ {to_key}")

    try:
        from data.generator import generate_sample_data

        if period and not bootstrap:
            path = generate_sample_data(
                scenario=scenario,
                fac_id=fac_id,
                split=split,
                period=period,
            )
            set_input_folder(
                f"{fac_id}/{split}/{period}" if split in ("train", "test") else f"{fac_id}/{split}"
            )
            result = {"path": path}
        else:
            result = generate_sample(
                scenario=scenario,
                fac_id=fac_id,
                split=split,
                bootstrap=bootstrap,
                from_date=from_key,
                to_date=to_key,
                use_period_count=use_period_count,
                verbose=True,
            )
    except ValueError as e:
        print(f"[오류] {e}")
        scenarios = ", ".join(s["id"] for s in list_sample_scenarios())
        print(f"  사용 가능 시나리오: {scenarios}")
        sys.exit(1)

    path = result["path"]
    print(f"  생성 완료: {path}")
    print(f"  입력 경로: {CONFIG.path.input_folder_key}")
    print(f"  input_dir: {CONFIG.path.input_dir}")


def cmd_collect(
    fac_id: str,
    split: str = "train",
    interval: int = 0,
    prevcnt: int = 1,
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
        prevcnt=prevcnt,
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
        "--prevcnt", type=int, default=None,
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
        "--ruletimekey", default=None,
        help="단일 RULE_TIMEKEY 학습 (미지정 시 --prevcnt 또는 --from/--to 필요)",
    )
    train_p.add_argument(
        "--db", action="store_true",
        help="Oracle RULE_TIMEKEY 조회·자동 수집·무효 폴더 재수집·validation SQL 갱신 (기본: dataset JSON만)",
    )
    train_p.add_argument(
        "--nodb", action="store_true",
        help="dataset 기존 JSON만 사용 (기본값, 명시용)",
    )
    train_p.add_argument(
        "--all", dest="all_folders", action="store_true",
        help="train 폴더 전체 학습 (--prevcnt/--from/--to/--ruletimekey 불필요)",
    )
    train_p.add_argument(
        "--lotcd",
        default=None,
        help="자동 수집 시 SQL :LOT_CD 바인드 (discrete_arrange 제외)",
    )
    test_p = sub.add_parser("test", help="test 데이터 검증")
    test_p.add_argument("--facid", required=True, help="공장 ID")
    test_p.add_argument(
        "--prevcnt", type=int, default=None,
        help="현재 기준 최근 N일 test 데이터",
    )
    test_p.add_argument(
        "--from", dest="from_key", metavar="RULE_TIMEKEY", default=None,
        help="검증 시작 RULE_TIMEKEY",
    )
    test_p.add_argument(
        "--to", dest="to_key", metavar="RULE_TIMEKEY", default=None,
        help="검증 종료 RULE_TIMEKEY",
    )
    test_p.add_argument(
        "--lotcd",
        default=None,
        help="자동 수집 시 SQL :LOT_CD 바인드 (discrete_arrange 제외)",
    )
    test_p.add_argument(
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
        "--from", dest="from_key", metavar="RULE_TIMEKEY", default=None,
        help="구간 시작 RULE_TIMEKEY (BETWEEN 조회 후 최신값 사용, --ruletimekey와 함께 쓸 수 없음)",
    )
    inf_p.add_argument(
        "--to", dest="to_key", metavar="RULE_TIMEKEY", default=None,
        help="구간 종료 RULE_TIMEKEY (BETWEEN 조회 후 최신값 사용, --ruletimekey와 함께 쓸 수 없음)",
    )
    inf_p.add_argument(
        "--prevcnt", type=int, default=None,
        help="최신 기준 최근 N개 RULE_TIMEKEY 조회 후 최신값 사용 (--ruletimekey와 함께 쓸 수 없음)",
    )
    inf_p.add_argument(
        "--lotcd",
        default=None,
        help="SQL :LOT_CD 바인드 (discrete_arrange 제외, 기본: SQL_LOT_CD / COLLECTOR_LOT_CD)",
    )
    inf_p.add_argument(
        "--nodb", action="store_true",
        help="Oracle 조회 생략, dataset 기존 JSON 사용",
    )
    inf_p.add_argument(
        "--decision-log", action="store_true",
        help="step별 EQP/PPK/OPER 결정 및 미할당 사유를 result_full.json에 기록",
    )
    inf_p.add_argument(
        "--include-history",
        action="store_true",
        help="UI 재생용 history/event snapshot을 생성합니다. 기본은 빠른 schedule 결과만 생성.",
    )
    inf_p.add_argument(
        "--enable-wip-inflow",
        action="store_true",
        help="공정 완료 시 다음 공정 flow 재공 유입 이벤트를 켭니다. 기본은 현재 재공만 배정.",
    )
    inf_p.add_argument(
        "--db-load",
        action="store_true",
        help="추론 후 output/sql 을 Oracle RTS 테이블에 적재",
    )
    inf_p.add_argument(
        "--db",
        default=None,
        help="db-load 시 DB alias (미지정 시 databases.yaml default)",
    )
    inf_p.add_argument(
        "--no-history",
        action="store_true",
        help="db-load 시 HIS 테이블 적재 생략",
    )
    inf_p.add_argument(
        "--max-conversions",
        type=int,
        default=None,
        metavar="N",
        help="시뮬 전체 전환(컨버전) 상한",
    )
    inf_p.add_argument(
        "--max-conversions-per-eqp",
        type=int,
        default=None,
        metavar="N",
        help="EQP별 전환(컨버전) 상한",
    )
    inf_p.add_argument(
        "--conversion-minutes",
        type=int,
        default=None,
        metavar="MIN",
        help="LOT_CD/TEMP 전환 1회 소요 시간(분, 기본: config.env.conversion_minutes)",
    )
    inf_p.add_argument(
        "--strict-validate",
        action="store_true",
        help="결과 검증(장비 투입 가능성·처리시간·배정 완전성) 실패 시 종료코드 1로 종료",
    )

    db_load_p = sub.add_parser(
        "db-load",
        help="RTS output.json / sql → Oracle 적재 (추론 결과 DB 반영)",
    )
    db_load_p.add_argument(
        "--ddl-only",
        action="store_true",
        help="output 테이블 CREATE 만 실행 (data/sql/rts_output_tables.sql)",
    )
    db_load_p.add_argument(
        "--ddl",
        action="store_true",
        help="적재 전 DDL도 실행 (테이블 없을 때)",
    )
    db_load_p.add_argument("--facid", help="dataset FAC_ID")
    db_load_p.add_argument(
        "--split", default="infer", choices=("train", "test", "infer"),
    )
    db_load_p.add_argument("--period", help="RULE_TIMEKEY (train/test)")
    db_load_p.add_argument(
        "--output-dir",
        help="output 폴더 직접 지정 (dataset 경로 대신)",
    )
    db_load_p.add_argument(
        "--json",
        dest="json_path",
        help="output.json 경로 (SQL 생성 후 즉시 적재)",
    )
    db_load_p.add_argument(
        "--db",
        default=None,
        help="DB alias (미지정 시 SQL @db 또는 default)",
    )
    db_load_p.add_argument(
        "--no-history",
        action="store_true",
        help="HIS 테이블 적재 생략",
    )
    db_load_p.add_argument(
        "--regenerate-sql",
        action="store_true",
        help="output.json에서 sql/*.sql 재생성 후 적재",
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
    collect_p.add_argument("--prevcnt", type=int, default=1)
    collect_p.add_argument("--from", dest="from_key", metavar="RULE_TIMEKEY")
    collect_p.add_argument("--to", dest="to_key", metavar="RULE_TIMEKEY")
    collect_p.add_argument("--once", action="store_true")
    collect_p.add_argument("--snapshot", action="store_true")
    collect_p.add_argument("--period", help="--snapshot 시 RULE_TIMEKEY")
    collect_p.add_argument(
        "--lotcd",
        default=None,
        help="SQL :LOT_CD 바인드 (discrete_arrange 제외, 기본: COLLECTOR_LOT_CD / SQL_LOT_CD)",
    )
    add_debug_arguments(collect_p)

    sample_p = sub.add_parser(
        "sample",
        help="Oracle 없이 샘플 dataset JSON 생성 (train/test/infer)",
    )
    sample_p.add_argument("--facid", required=True, help="공장 ID (예: FAC001)")
    sample_p.add_argument(
        "--split", default="train", choices=("train", "test", "infer"),
        help="생성할 split (기본: train)",
    )
    sample_p.add_argument(
        "--scenario", default="default",
        help="시나리오 ID (default, pacing_steady, random 등)",
    )
    sample_p.add_argument(
        "--bootstrap", action="store_true",
        help="train 3일 + test 1일 + infer 를 한 번에 생성",
    )
    sample_p.add_argument(
        "--period", metavar="RULE_TIMEKEY",
        help="특정 RULE_TIMEKEY 폴더에 생성 (예: 20260621070000)",
    )
    sample_p.add_argument("--from", dest="from_key", metavar="RULE_TIMEKEY")
    sample_p.add_argument("--to", dest="to_key", metavar="RULE_TIMEKEY")
    sample_p.add_argument(
        "--use-period-count", action="store_true",
        help="설정된 train/test 기간 수만큼 연속 생성",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "train":
            if not args.all_folders:
                _validate_period_selectors(
                    args, has_ruletimekey=True, require_one=True,
                    one_of_label="--ruletimekey, --prevcnt, --from/--to, --all",
                )
            if args.nodb and args.db:
                print("[오류] --db 와 --nodb 는 함께 쓸 수 없습니다.")
                sys.exit(1)
            cmd_train(
                fac_id=args.facid,
                prevcnt=args.prevcnt,
                from_key=args.from_key,
                to_key=args.to_key,
                rule_timekey=args.ruletimekey,
                nodb=not args.db,
                lot_cd=args.lotcd,
                all_folders=args.all_folders,
            )

        elif args.command == "test":
            _validate_period_selectors(args)
            cmd_test(
                fac_id=args.facid,
                prevcnt=args.prevcnt,
                from_key=args.from_key,
                to_key=args.to_key,
                nodb=args.nodb,
                lot_cd=args.lotcd,
            )

        elif args.command == "infer":
            _validate_period_selectors(args, has_ruletimekey=True)
            cmd_inference(
                fac_id=args.facid,
                rule_timekey=args.ruletimekey,
                from_key=args.from_key,
                to_key=args.to_key,
                prevcnt=args.prevcnt,
                nodb=args.nodb,
                lot_cd=args.lotcd,
                decision_log=args.decision_log,
                enable_wip_inflow=args.enable_wip_inflow,
                include_history=args.include_history,
                db_load=args.db_load,
                db_alias=args.db,
                no_history=args.no_history,
                max_conversions=args.max_conversions,
                max_conversions_per_eqp=args.max_conversions_per_eqp,
                conversion_minutes=args.conversion_minutes,
                strict_validate=args.strict_validate,
            )

        elif args.command == "db-load":
            cmd_db_load(
                ddl_only=args.ddl_only,
                apply_ddl=args.ddl,
                fac_id=args.facid,
                split=args.split,
                period=args.period,
                output_dir=args.output_dir,
                db_alias=args.db,
                json_path=args.json_path,
                no_history=args.no_history,
                regenerate_sql=args.regenerate_sql,
            )

        elif args.command == "collect":
            cmd_collect(
                fac_id=args.facid,
                split=args.split,
                interval=args.interval,
                prevcnt=args.prevcnt,
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

        elif args.command == "sample":
            cmd_sample(
                fac_id=args.facid,
                split=args.split,
                scenario=args.scenario,
                bootstrap=args.bootstrap,
                period=args.period,
                from_key=args.from_key,
                to_key=args.to_key,
                use_period_count=args.use_period_count,
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
