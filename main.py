"""
main.py – 워크플로우 컨트롤러 (CLI)
DB조회 → JSON변환 → 학습/추론 → 결과저장 → (DB 적재) 순서를 명령줄로 제어합니다.

사용 예:
    python main.py sample          # 샘플 데이터 생성
    python main.py train           # 모델 학습
    python main.py infer           # 추론 및 결과 저장
    python main.py all             # 샘플생성 + 학습 + 추론 일괄 실행
    python main.py ui              # React UI + API 서버 실행

    python main.py train --timesteps 50000   # 타임스텝 지정 학습
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder, EXTERNAL_DIR
from data.loader import load_data, validate_data, generate_sample_data, list_sample_scenarios, SAMPLE_SCENARIOS
from data.preprocessor import preprocess
from agent.rl_agent import SchedulingAgent
from agent.registry import VALID_ALGORITHMS
from inference.runner import run_inference, save_result


# ── 개별 워크플로우 단계 ──────────────────────────────────────────────────────

def step_generate_sample(scenario: str = "default", output_dir=None):
    """[1] 샘플 JSON 데이터 생성"""
    print("=" * 60)
    print(f"[STEP 1] 샘플 데이터 생성 (시나리오: {scenario})")
    path = generate_sample_data(output_dir=output_dir, scenario=scenario)
    return path


def step_load() -> dict:
    """[2] JSON 로드 + 검증 + 전처리"""
    print("=" * 60)
    print("[STEP 2] 데이터 로드 및 전처리")
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
    """[3] PPO 모델 학습 및 저장"""
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
    """[4] 추론 실행 + 결과 저장 (external/output/result.json)"""
    print("=" * 60)
    print(f"[STEP 4] 추론 실행 (알고리즘: {algorithm})")
    result = run_inference(env_data, algorithm=algorithm, agent=agent)
    path = save_result(result)

    stats = result["stats"]
    print(f"  배정 LOT 수:    {len(result['schedule'])}")
    print(f"  공정 전환 횟수: {stats['oper_switches']}")
    print(f"  제품 전환 횟수: {stats['prod_switches']}")
    print(f"  Idle 합계:      {stats['idle_total']}분")
    print(f"  결과 파일:      {path}")
    return result


def step_launch_ui():
    """[5] FastAPI 백엔드 + React 프론트엔드 실행"""
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

    # API 준비 대기 (프론트엔드가 먼저 뜨며 /api 호출 실패하는 것 방지)
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
        print("[오류] API 서버가 시작되지 않았습니다. uvicorn 설치 여부를 확인하세요.")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    fe_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir),
        shell=True,
        env={**os.environ, "BROWSER": "none"},
    )

    # Vite 준비 대기
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


# ── CLI 파서 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Scheduling RL 워크플로우",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", metavar="FOLDER", default=None,
        help="입력 데이터 폴더명 (external/<FOLDER>/, 기본: input)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample_p = sub.add_parser("sample", help="샘플 데이터 생성")
    sample_p.add_argument(
        "--scenario", "-s",
        choices=sorted(SAMPLE_SCENARIOS),
        default="default",
        help="샘플 시나리오 (single_heavy_wip: 단일제품 ST½ 재공다량)",
    )

    train_p = sub.add_parser("train", help="모델 학습")
    train_p.add_argument("--timesteps", type=int, default=None,
                         help="학습 타임스텝 수 (기본: config 값)")

    infer_p = sub.add_parser("infer", help="추론 실행 및 결과 저장")
    infer_p.add_argument(
        "--algorithm", choices=sorted(VALID_ALGORITHMS), default="rl",
        help="추론 알고리즘 (기본: rl)",
    )
    sub.add_parser("all",   help="샘플생성 + 학습 + 추론 일괄 실행")
    sub.add_parser("ui",    help="React UI + API 서버 실행")

    return parser.parse_args()


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.input:
        set_input_folder(args.input)
        print(f"[설정] 입력 폴더: {CONFIG.path.input_dir}")

    if args.command == "sample":
        out = EXTERNAL_DIR / args.input if args.input else None
        step_generate_sample(scenario=args.scenario, output_dir=out)

    elif args.command == "train":
        env_data = step_load()
        step_train(env_data, timesteps=args.timesteps)

    elif args.command == "infer":
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
        step_generate_sample()
        env_data = step_load()
        agent = step_train(env_data)
        step_infer(env_data, agent=agent)
        print("\n전체 워크플로우 완료!")

    elif args.command == "ui":
        step_launch_ui()


if __name__ == "__main__":
    main()
