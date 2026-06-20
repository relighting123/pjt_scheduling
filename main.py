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
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder, train_snapshot_now, PERIOD_SPLITS
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


def step_load() -> dict:
    print("=" * 60)
    print("[STEP 2] 데이터 로드 및 전처리")
    print(f"  input: {CONFIG.path.input_dir}")
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        for e in errors:
            print(f"  [오류] {e}")
        sys.exit(1)
    env_data = preprocess(raw)
    print(f"  EQP: {len(env_data['eqp_ids'])}대  "
          f"LOT: {len(env_data['lots'])}개  "
          f"제품: {len(env_data['prod_keys'])}종  "
          f"공정: {len(env_data['oper_ids'])}종")
    return env_data


def step_train(env_data: dict, timesteps: int = None):
    print("=" * 60)
    print("[STEP 3] 모델 학습")
    if timesteps:
        CONFIG.rl.total_timesteps = timesteps
    print(f"  Total Timesteps: {CONFIG.rl.total_timesteps:,}")

    agent = SchedulingAgent()
    agent.train(env_data, verbose=1)
    agent.save()

    print("\n  평가 중 (3 에피소드)...")
    metrics = agent.evaluate(env_data, n_episodes=3)
    print(f"  평균 보상:       {metrics['mean_reward']:.2f}")
    print(f"  공정 전환(평균): {metrics['mean_oper_sw']:.1f}")
    print(f"  제품 전환(평균): {metrics['mean_prod_sw']:.1f}")
    print(f"  Idle 합계(평균): {metrics['mean_idle']:.0f}분")
    return agent


def step_infer(env_data: dict, agent: SchedulingAgent = None, algorithm: str = "rl") -> dict:
    print("=" * 60)
    print(f"[STEP 4] 추론 실행 (알고리즘: {algorithm})")
    print(f"  입력: {CONFIG.path.input_dir}")
    result = run_inference(env_data, algorithm=algorithm, agent=agent)
    out_dir = CONFIG.path.infer_output_dir
    path = save_result(result, output_dir=out_dir)

    stats = result["stats"]
    print(f"  배정 LOT 수:    {len(result['schedule'])}")
    print(f"  공정 전환 횟수: {stats['oper_switches']}")
    print(f"  제품 전환 횟수: {stats['prod_switches']}")
    print(f"  Idle 합계:      {stats['idle_total']}분")
    print(f"  결과 파일:      {path}")
    return result


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
        [sys.executable, "-m", "uvicorn", "api.server:app",
         "--host", "127.0.0.1", "--port", "8000", "--reload"],
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
    parser.add_argument(
        "--input", "-i", metavar="PATH", default=None,
        help="dataset 경로 키 (예: FAC001/train/20260620070000, FAC001/infer)",
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
    train_p.add_argument("--timesteps", type=int, default=None)

    infer_p = sub.add_parser("infer", help="추론 실행 및 결과 저장")
    infer_p.add_argument(
        "--algorithm", choices=sorted(VALID_ALGORITHMS), default="rl",
    )
    sub.add_parser("all", help="샘플생성 + 학습 + 추론 일괄 실행")
    sub.add_parser("ui", help="React UI + API 서버 실행")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.input:
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
