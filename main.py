"""
main.py – 워크플로우 컨트롤러 (CLI)

사용 예:
    python main.py sample          # 샘플 JSON 생성 (dataset/)
    python main.py fetch           # Oracle SQL → JSON (dataset/)
    python main.py train           # 모델 학습
    python main.py infer           # 추론 및 결과 저장
    python main.py ui              # React UI + API 서버

    python main.py -i FAC001/train/202406191430 infer
    python main.py sample -s single_heavy_wip --fac-id FAC001
    python main.py run -i FAC001/train/20260619070000 --timesteps 5000
    python main.py run --all --fac-id FAC001 --timesteps 200000
    python main.py run --all --bootstrap --scenario default --timesteps 200000
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from typing import List, Optional, Union

from config import (
    CONFIG,
    set_input_folder,
    PERIOD_SPLITS,
    parse_input_folder,
    list_input_folders,
    iter_rule_timekeys,
    validate_path_segment,
)
from data.loader import load_data, validate_data, fetch_from_db, fetch_period_range
from data.generator import (
    generate_sample_data,
    generate_sample_period_range,
    bootstrap_facility_datasets,
    list_sample_scenarios,
    SAMPLE_SCENARIOS,
)
from data.preprocessor import preprocess
from agent.rl_agent import SchedulingAgent
from agent.registry import VALID_ALGORITHMS
from inference.runner import run_inference, save_result


def step_generate_sample(
    scenario: str = "default",
    fac_id: str = "FAC001",
    split: str = "train",
    bootstrap: bool = False,
    from_date: str = None,
    to_date: str = None,
):
    """[1] 샘플 JSON 데이터 생성 → external/dataset/{FAC_ID}/..."""
    print("=" * 60)
    print(f"[STEP 1] 샘플 데이터 생성 (시나리오: {scenario}, FAC: {fac_id})")
    if bootstrap:
        info = bootstrap_facility_datasets(fac_id=fac_id, scenario=scenario)
        snap = info["train_snapshot"]
        set_input_folder(f"{fac_id}/train/{snap}")
        train_entry = info["paths"]["train"]
        train_input = train_entry[-1]["input"] if isinstance(train_entry, list) else train_entry["input"]
        print(f"  train periods: {info.get('train_periods', [snap])}")
        print(f"  test periods:  {info.get('test_periods', [info.get('test_period')])}")
        return Path(train_input)
    if from_date and to_date:
        paths = generate_sample_period_range(
            scenario=scenario,
            fac_id=fac_id,
            from_date=from_date,
            to_date=to_date,
            split=split,
        )
        last = paths[-1]
        set_input_folder(f"{fac_id}/{split}/{last.parent.name}")
        return last
    if from_date or to_date:
        print("[오류] --from-date 와 --to-date 를 함께 지정하세요.")
        sys.exit(1)
    path = generate_sample_data(
        scenario=scenario,
        fac_id=fac_id,
        split=split,
    )
    if split in PERIOD_SPLITS:
        set_input_folder(f"{fac_id}/{split}/{path.parent.name}")
    else:
        set_input_folder(f"{fac_id}/{split}")
    return path


def step_fetch_db(
    fac_id: str = "FAC001",
    split: str = "train",
    snapshot: str = None,
    from_date: str = None,
    to_date: str = None,
):
    """[1b] Oracle SQL → JSON (external/sql → dataset input)"""
    print("=" * 60)
    print(f"[STEP 1b] DB 조회 → JSON (FAC: {fac_id}, split: {split})")
    if from_date and to_date:
        paths = fetch_period_range(
            fac_id=fac_id,
            from_date=from_date,
            to_date=to_date,
            split=split,
        )
        last = paths[-1]
        set_input_folder(f"{fac_id}/{split}/{last.parent.name}")
        print(f"  {len(paths)}개 기간 폴더 생성, 마지막: {last}")
        return last
    if from_date or to_date:
        print("[오류] --from-date 와 --to-date 를 함께 지정하세요.")
        sys.exit(1)
    path = fetch_from_db(fac_id=fac_id, split=split, snapshot=snapshot)
    if split in PERIOD_SPLITS:
        key = f"{fac_id}/{split}/{path.parent.name}"
    else:
        key = f"{fac_id}/{split}"
    set_input_folder(key)
    print(f"  입력 경로: {path}")
    return path


def load_env_data_for_folder(folder: str, *, verbose: bool = True) -> dict:
    set_input_folder(folder)
    if verbose:
        print(f"  input: {CONFIG.path.input_dir}")
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        for e in errors:
            print(f"  [오류] {e}")
        sys.exit(1)
    env_data = preprocess(raw)
    if verbose:
        print(f"  EQP: {len(env_data['eqp_ids'])}대  "
              f"LOT: {len(env_data['lots'])}개  "
              f"제품: {len(env_data['prod_keys'])}종  "
              f"공정: {len(env_data['oper_ids'])}종")
    return env_data


def step_load() -> dict:
    print("=" * 60)
    print("[STEP 2] 데이터 로드 및 전처리")
    print(f"  input: {CONFIG.path.input_dir}")
    return load_env_data_for_folder(CONFIG.path.input_folder_key, verbose=True)


def step_load_many(folders: List[str]) -> List[dict]:
    print("=" * 60)
    print(f"[STEP 2] 데이터 로드 ({len(folders)}개 기간)")
    datasets = []
    for folder in folders:
        print(f"  · {folder}")
        datasets.append(load_env_data_for_folder(folder, verbose=False))
    if datasets:
        d0 = datasets[0]
        print(f"  (대표) EQP: {len(d0['eqp_ids'])}대  "
              f"LOT: {len(d0['lots'])}개  "
              f"제품: {len(d0['prod_keys'])}종  "
              f"공정: {len(d0['oper_ids'])}종")
    return datasets


def split_folders_for_fac(fac_id: str, split: str) -> List[str]:
    prefix = f"{validate_path_segment(fac_id, 'FAC_ID')}/{split}/"
    return sorted(f for f in list_input_folders() if f.startswith(prefix))


def filter_folders_by_period(
    folders: List[str],
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[str]:
    if not from_date or not to_date:
        return folders
    keys = set(iter_rule_timekeys(from_date, to_date))
    return [f for f in folders if f.rsplit("/", 1)[-1] in keys]


def step_train(env_data: Union[dict, List[dict]], timesteps: int = None):
    print("=" * 60)
    print("[STEP 3] 모델 학습")
    datasets = env_data if isinstance(env_data, list) else [env_data]
    if len(datasets) > 1:
        print(f"  학습 기간: {len(datasets)}개 (VecEnv)")
    if timesteps:
        CONFIG.rl.total_timesteps = timesteps
    print(f"  Total Timesteps: {CONFIG.rl.total_timesteps:,}")

    agent = SchedulingAgent()
    agent.train(env_data, verbose=1)
    agent.save()

    print("\n  평가 중 (3 에피소드, 첫 번째 기간)...")
    metrics = agent.evaluate(datasets[0], n_episodes=3)
    print(f"  평균 보상:       {metrics['mean_reward']:.2f}")
    print(f"  공정 전환(평균): {metrics['mean_oper_sw']:.1f}")
    print(f"  제품 전환(평균): {metrics['mean_prod_sw']:.1f}")
    print(f"  Idle 합계(평균): {metrics['mean_idle']:.0f}분")
    return agent


def step_infer(
    env_data: dict,
    agent: SchedulingAgent = None,
    algorithm: str = "rl",
    *,
    output_dir: Path = None,
    result_name: str = "result",
) -> dict:
    print("=" * 60)
    print(f"[STEP 4] 추론 실행 (알고리즘: {algorithm})")
    print(f"  입력: {CONFIG.path.input_dir}")
    result = run_inference(env_data, algorithm=algorithm, agent=agent)
    out_dir = output_dir or CONFIG.path.infer_output_dir
    path = save_result(result, output_dir=out_dir, result_name=result_name)

    stats = result["stats"]
    print(f"  배정 LOT 수:    {len(result['schedule'])}")
    print(f"  공정 전환 횟수: {stats['oper_switches']}")
    print(f"  제품 전환 횟수: {stats['prod_switches']}")
    print(f"  Idle 합계:      {stats['idle_total']}분")
    print(f"  결과 파일:      {path}")
    return result


def _load_single_train(folder: str) -> dict:
    print("=" * 60)
    print("[STEP 2] 데이터 로드 및 전처리")
    return load_env_data_for_folder(folder, verbose=True)


def resolve_test_folder(train_folder: str, test_folder: str | None = None) -> str:
    """train 폴더 키 → 동일 RULE_TIMEKEY의 test 폴더 (또는 명시 --test)."""
    if test_folder:
        return test_folder.strip()
    fac_id, split, period = parse_input_folder(train_folder.strip())
    if split != "train":
        raise ValueError(
            f"train 폴더 형식이 필요합니다: FAC_ID/train/{{RULE_TIMEKEY}} (받은 값: {train_folder!r})"
        )
    if not period:
        raise ValueError(
            "test 폴더를 자동 추론하려면 train/{RULE_TIMEKEY} 형식이거나 --test 를 지정하세요."
        )
    return f"{fac_id}/test/{period}"


def resolve_run_folders(
    *,
    fac_id: str = "FAC001",
    train_folder: Optional[str] = None,
    test_folder: Optional[str] = None,
    all_data: bool = False,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> tuple[List[str], List[str]]:
    if all_data:
        train_folders = filter_folders_by_period(
            split_folders_for_fac(fac_id, "train"), from_date, to_date,
        )
        test_folders = filter_folders_by_period(
            split_folders_for_fac(fac_id, "test"), from_date, to_date,
        )
        if not train_folders:
            raise ValueError(
                f"{fac_id} train 데이터 없음. "
                f"'python main.py sample --bootstrap --fac-id {fac_id}' 또는 fetch로 생성하세요."
            )
        if not test_folders:
            raise ValueError(
                f"{fac_id} test 데이터 없음. sample/fetch로 test 기간을 생성하세요."
            )
        return train_folders, test_folders

    if not train_folder:
        raise ValueError("train 폴더는 -i 또는 --all 로 지정하세요.")

    train_key = train_folder.strip()
    test_key = resolve_test_folder(train_key, test_folder)
    return [train_key], [test_key]


def step_run(
    train_folder: Optional[str] = None,
    test_folder: Optional[str] = None,
    timesteps: int | None = None,
    *,
    compare: bool = False,
    all_data: bool = False,
    fac_id: str = "FAC001",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    bootstrap: bool = False,
    scenario: str = "default",
):
    """train 데이터 학습 → test 데이터 추론 (단일 또는 전체 기간)."""
    fac_id = validate_path_segment(fac_id, "FAC_ID")

    if bootstrap:
        print("=" * 60)
        print(f"[RUN] bootstrap 샘플 생성 (scenario={scenario})")
        bootstrap_facility_datasets(fac_id=fac_id, scenario=scenario)

    train_folders, test_folders = resolve_run_folders(
        fac_id=fac_id,
        train_folder=train_folder,
        test_folder=test_folder,
        all_data=all_data,
        from_date=from_date,
        to_date=to_date,
    )

    print("=" * 60)
    print("[RUN] train 학습 + test 추론")
    print(f"  train ({len(train_folders)}): {', '.join(train_folders)}")
    print(f"  test  ({len(test_folders)}): {', '.join(test_folders)}")

    train_env = (
        step_load_many(train_folders)
        if len(train_folders) > 1
        else _load_single_train(train_folders[0])
    )
    agent = step_train(train_env, timesteps=timesteps)

    algorithms = ["rl"]
    if compare:
        algorithms.append("minprogress")

    results_by_test: dict = {}
    for test_key in test_folders:
        test_input = set_input_folder(test_key)
        if not test_input.is_dir():
            print(f"[오류] test 입력 폴더 없음: {test_input}")
            sys.exit(1)
        print("=" * 60)
        print(f"[STEP 2] test 로드: {test_key}")
        test_env = load_env_data_for_folder(test_key, verbose=True)
        test_output = CONFIG.path.output_dir
        test_output.mkdir(parents=True, exist_ok=True)
        period = test_key.rsplit("/", 1)[-1]
        folder_results = {}
        for algo in algorithms:
            rl_agent = agent if algo == "rl" else None
            if len(algorithms) == 1 and len(test_folders) == 1:
                name = "result"
            elif len(algorithms) == 1:
                name = f"result_{period}"
            else:
                name = f"result_{period}_{algo}"
            folder_results[algo] = step_infer(
                test_env,
                agent=rl_agent,
                algorithm=algo,
                output_dir=test_output,
                result_name=name,
            )
        results_by_test[test_key] = folder_results

    print("\n" + "=" * 60)
    print("[RUN] 완료")
    print(f"  모델: {CONFIG.path.model_dir / CONFIG.rl.model_name}.zip")
    for test_key, folder_results in results_by_test.items():
        print(f"  test {test_key} → {CONFIG.path.output_dir}")
        for algo, res in folder_results.items():
            stats = res["stats"]
            print(
                f"    [{algo}] LOT {len(res['schedule'])} · "
                f"oper_sw={stats['oper_switches']} · prod_sw={stats['prod_switches']} · "
                f"idle={stats['idle_total']}분"
            )
    return agent, results_by_test


def step_launch_ui():
    import os
    import time
    import urllib.error
    import urllib.request
    import webbrowser

    print("=" * 60)
    print("[STEP 5] React UI + API 서버 실행")
    print("  API:      http://127.0.0.1:8000")
    print("  Frontend: http://localhost:5173")
    print("  종료: Ctrl+C")

    frontend_dir = ROOT / "frontend"
    if not (frontend_dir / "node_modules").exists():
        print("\n  [안내] frontend 의존성 설치 중 (npm install)...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True, shell=True)

    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "api.server:app",
            "--host", "127.0.0.1", "--port", "8000", "--reload",
            "--reload-exclude", "models",
            "--reload-exclude", "external/dataset",
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
        print("\n종료 중...")
    finally:
        fe_proc.terminate()
        api_proc.terminate()
        fe_proc.wait(timeout=5)
        api_proc.wait(timeout=5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Scheduling RL 워크플로우",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    input_help = "dataset 경로 키 (예: FAC001/train/20260620070000, FAC001/infer)"
    parser.add_argument(
        "--input", "-i", metavar="PATH", default=None,
        help=input_help + " — train/infer 앞·뒤 모두 가능",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample_p = sub.add_parser("sample", help="샘플 JSON 생성 (generator)")
    sample_p.add_argument("--fac-id", default="FAC001", help="공장 ID")
    sample_p.add_argument(
        "--scenario", "-s",
        choices=sorted(SAMPLE_SCENARIOS),
        default="default",
    )
    sample_p.add_argument(
        "--split", choices=("train", "test", "infer"), default="train",
    )
    sample_p.add_argument(
        "--bootstrap", action="store_true",
        help="train/test/infer 전체 골격 + 샘플 생성",
    )
    sample_p.add_argument(
        "--from-date", "--from-timekey", dest="from_date", metavar="RULE_TIMEKEY",
        help="시작 RULE_TIMEKEY (YYYYMMDDHHmmss, 8자리면 070000 붙음)",
    )
    sample_p.add_argument(
        "--to-date", "--to-timekey", dest="to_date", metavar="RULE_TIMEKEY",
        help="종료 RULE_TIMEKEY (YYYYMMDDHHmmss)",
    )

    fetch_p = sub.add_parser("fetch", help="Oracle SQL → JSON (loader)")
    fetch_p.add_argument("--fac-id", default="FAC001", help="공장 ID")
    fetch_p.add_argument(
        "--split", choices=("train", "test", "infer"), default="train",
    )
    fetch_p.add_argument(
        "--snapshot", default=None,
        help="단일 RULE_TIMEKEY 폴더명 (YYYYMMDDHHmmss)",
    )
    fetch_p.add_argument(
        "--from-date", "--from-timekey", dest="from_date",
        metavar="RULE_TIMEKEY", help="시작 RULE_TIMEKEY",
    )
    fetch_p.add_argument(
        "--to-date", "--to-timekey", dest="to_date",
        metavar="RULE_TIMEKEY", help="종료 RULE_TIMEKEY",
    )

    train_p = sub.add_parser("train", help="모델 학습")
    train_p.add_argument(
        "--input", "-i", metavar="PATH", default=None, help=input_help,
    )
    train_p.add_argument("--timesteps", type=int, default=None)

    infer_p = sub.add_parser("infer", help="추론 실행 및 결과 저장")
    infer_p.add_argument(
        "--input", "-i", metavar="PATH", default=None, help=input_help,
    )
    infer_p.add_argument(
        "--algorithm", choices=sorted(VALID_ALGORITHMS), default="rl",
    )

    run_p = sub.add_parser(
        "run",
        help="train 학습 후 test 추론 (한 번에)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예:\n"
            "  # 단일 기간\n"
            "  python main.py run -i FAC001/train/20260619070000 --timesteps 5000\n"
            "  # FAC 전체 train/test (bootstrap 기본 3 train + 1 test)\n"
            "  python main.py run --all --bootstrap --timesteps 200000\n"
            "  # Oracle fetch 후 전체\n"
            "  python main.py fetch --split train --from-date 20260601 --to-date 20260607\n"
            "  python main.py fetch --split test --from-date 20260608 --to-date 20260608\n"
            "  python main.py run --all --timesteps 200000"
        ),
    )
    run_p.add_argument(
        "--input", "-i", metavar="PATH", default=None,
        help="단일 train dataset 경로 (--all 과 함께 쓰지 않음)",
    )
    run_p.add_argument(
        "--all", action="store_true",
        help="FAC의 train 전 기간 VecEnv 학습 + test 전 기간 추론",
    )
    run_p.add_argument("--fac-id", default="FAC001", help="--all / bootstrap 용 FAC_ID")
    run_p.add_argument(
        "--scenario", "-s",
        choices=sorted(SAMPLE_SCENARIOS),
        default="default",
        help="--bootstrap 시 샘플 시나리오",
    )
    run_p.add_argument(
        "--bootstrap", action="store_true",
        help="run 전 train/test/infer 샘플 골격 생성 (train 3 + test 1 기간)",
    )
    run_p.add_argument(
        "--from-date", "--from-timekey", dest="from_date", metavar="RULE_TIMEKEY",
        help="--all 시 train/test 기간 필터 시작",
    )
    run_p.add_argument(
        "--to-date", "--to-timekey", dest="to_date", metavar="RULE_TIMEKEY",
        help="--all 시 train/test 기간 필터 종료",
    )
    run_p.add_argument(
        "--test", metavar="PATH", default=None,
        help="test dataset 경로 키 (미지정 시 train과 동일 RULE_TIMEKEY)",
    )
    run_p.add_argument("--timesteps", type=int, default=None)
    run_p.add_argument(
        "--compare", action="store_true",
        help="test에서 RL + Min-Progress 둘 다 실행",
    )

    sub.add_parser("all", help="샘플생성 + 학습 + 추론 일괄 실행")
    sub.add_parser("ui", help="React UI + API 서버 실행")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command != "run" and getattr(args, "input", None):
        set_input_folder(args.input)
        print(f"[설정] 입력: {CONFIG.path.input_dir}")

    if args.command == "sample":
        step_generate_sample(
            scenario=args.scenario,
            fac_id=args.fac_id,
            split=args.split,
            bootstrap=args.bootstrap,
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
        )

    elif args.command == "fetch":
        step_fetch_db(
            fac_id=args.fac_id,
            split=args.split,
            snapshot=args.snapshot,
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
        )

    elif args.command == "train":
        env_data = step_load()
        step_train(env_data, timesteps=args.timesteps)

    elif args.command == "infer":
        if not args.input:
            fac = CONFIG.path.fac_id
            set_input_folder(f"{fac}/infer")
            print(f"[설정] 추론 입력: {CONFIG.path.infer_input_dir}")
        env_data = step_load()
        algo = args.algorithm
        agent = None
        if algo == "rl":
            agent = SchedulingAgent()
            if not agent.model_exists():
                print("[오류] 저장된 모델이 없습니다. 먼저 'python main.py train'을 실행하세요.")
                sys.exit(1)
            agent = SchedulingAgent.load()
        step_infer(env_data, agent=agent, algorithm=algo)

    elif args.command == "run":
        if not args.all and not args.input:
            print("[오류] run 은 -i TRAIN_PATH 또는 --all 중 하나가 필요합니다.")
            sys.exit(1)
        if args.all and args.input:
            print("[오류] --all 과 -i 는 함께 사용할 수 없습니다.")
            sys.exit(1)
        if (args.from_date or args.to_date) and not args.all:
            print("[오류] --from-date/--to-date 는 --all 과 함께 사용하세요.")
            sys.exit(1)
        if args.from_date and not args.to_date:
            print("[오류] --from-date 와 --to-date 를 함께 지정하세요.")
            sys.exit(1)
        try:
            step_run(
                train_folder=args.input,
                test_folder=args.test,
                timesteps=args.timesteps,
                compare=args.compare,
                all_data=args.all,
                fac_id=args.fac_id,
                from_date=args.from_date,
                to_date=args.to_date,
                bootstrap=args.bootstrap,
                scenario=args.scenario,
            )
        except ValueError as e:
            print(f"[오류] {e}")
            sys.exit(1)

    elif args.command == "all":
        fac = getattr(args, "fac_id", "FAC001")
        step_generate_sample(bootstrap=True, fac_id=fac)
        env_data = step_load()
        agent = step_train(env_data)
        step_infer(env_data, agent=agent)
        print("\n전체 워크플로우 완료!")

    elif args.command == "ui":
        step_launch_ui()


if __name__ == "__main__":
    main()
