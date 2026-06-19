"""
main.py – 워크플로우 컨트롤러 (CLI)
DB조회 → JSON변환 → 학습/추론 → 결과저장 → (DB 적재) 순서를 명령줄로 제어합니다.

사용 예:
    python main.py sample          # 샘플 데이터 생성
    python main.py train           # 모델 학습
    python main.py infer           # 추론 및 결과 저장
    python main.py all             # 샘플생성 + 학습 + 추론 일괄 실행
    python main.py ui              # Streamlit UI 실행

    python main.py train --timesteps 50000   # 타임스텝 지정 학습
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CONFIG
from data.loader import load_data, validate_data, generate_sample_data
from data.preprocessor import preprocess
from agent.rl_agent import SchedulingAgent
from inference.runner import run_inference, save_result


# ── 개별 워크플로우 단계 ──────────────────────────────────────────────────────

def step_generate_sample():
    """[1] 샘플 JSON 데이터 생성 (external/input/ → DB 조회 결과 대체)"""
    print("=" * 60)
    print("[STEP 1] 샘플 데이터 생성")
    generate_sample_data()


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


def step_infer(env_data: dict, agent: SchedulingAgent = None) -> dict:
    """[4] 추론 실행 + 결과 저장 (external/output/result.json)"""
    print("=" * 60)
    print("[STEP 4] 추론 실행")
    result = run_inference(env_data, agent=agent)
    path = save_result(result)

    stats = result["stats"]
    print(f"  배정 LOT 수:    {len(result['schedule'])}")
    print(f"  공정 전환 횟수: {stats['oper_switches']}")
    print(f"  제품 전환 횟수: {stats['prod_switches']}")
    print(f"  Idle 합계:      {stats['idle_total']}분")
    print(f"  결과 파일:      {path}")
    return result


def step_launch_ui():
    """[5] Streamlit UI 실행"""
    print("=" * 60)
    print("[STEP 5] Streamlit UI 실행")
    ui_path = ROOT / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)],
                   check=True)


# ── CLI 파서 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Scheduling RL 워크플로우",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sample", help="샘플 데이터 생성")

    train_p = sub.add_parser("train", help="모델 학습")
    train_p.add_argument("--timesteps", type=int, default=None,
                         help="학습 타임스텝 수 (기본: config 값)")

    sub.add_parser("infer", help="추론 실행 및 결과 저장")
    sub.add_parser("all",   help="샘플생성 + 학습 + 추론 일괄 실행")
    sub.add_parser("ui",    help="Streamlit UI 실행")

    return parser.parse_args()


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.command == "sample":
        step_generate_sample()

    elif args.command == "train":
        env_data = step_load()
        step_train(env_data, timesteps=args.timesteps)

    elif args.command == "infer":
        env_data = step_load()
        agent = SchedulingAgent()
        if not agent.model_exists():
            print("[오류] 저장된 모델이 없습니다. 먼저 'python main.py train'을 실행하세요.")
            sys.exit(1)
        loaded = SchedulingAgent.load()
        step_infer(env_data, agent=loaded)

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
