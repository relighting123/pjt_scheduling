"""
config.py – 전체 프로젝트 설정
모든 경로·환경·RL·보상 파라미터를 한 곳에서 관리합니다.
"""
from dataclasses import dataclass, field
from pathlib import Path

# ── 기본 경로 ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent          # pjt_scheduling/
EXTERNAL_DIR = BASE_DIR.parent / "external"  # git 외부 폴더 (DB 연동)


@dataclass
class PathConfig:
    input_dir:   Path = field(default_factory=lambda: EXTERNAL_DIR / "input")
    output_dir:  Path = field(default_factory=lambda: EXTERNAL_DIR / "output")
    model_dir:   Path = field(default_factory=lambda: BASE_DIR / "models")

    schedule_file:     str = "schedule.json"      # 초기 스케줄링 결과
    availability_file: str = "availability.json"  # 투입 가능 여부
    plan_file:         str = "plan.json"           # 계획 데이터
    flow_file:         str = "flow.json"           # FLOW 데이터


@dataclass
class EnvConfig:
    # 관측 공간 정의용 최대값 (학습/추론 환경이 동일 크기를 유지해야 함)
    max_eqp_count:    int = 10   # EQP 최대 수
    max_oper_count:   int = 15   # OPER 종류 최대 수
    max_prod_count:   int = 10   # PLAN_PROD_KEY 종류 최대 수
    max_queue_size:   int = 20   # 현재 EQP에 배정 가능한 LOT 최대 수
    sim_time_horizon: int = 1440  # 시뮬레이션 최대 시간(분) – 기본 24h


@dataclass
class RLConfig:
    algorithm:       str   = "PPO"
    learning_rate:   float = 3e-4
    n_steps:         int   = 2048    # PPO 업데이트 주기 (스텝)
    batch_size:      int   = 64
    n_epochs:        int   = 10
    gamma:           float = 0.99
    total_timesteps: int   = 200_000
    model_name:      str   = "scheduling_rl"
    eval_freq:       int   = 10_000  # 모델 평가 주기


@dataclass
class RewardConfig:
    w_same_oper:       float = 2.0   # 동일 OPER 연속 진행 보너스
    w_same_prod:       float = 1.0   # 동일 제품 연속 진행 보너스
    w_idle_per_min:    float = -0.5  # EQP 분당 Idle 패널티
    w_completion:      float = 1.0   # LOT 완료 시 WF_QTY 기반 보상
    w_plan_hit:        float = 5.0   # 계획 달성 시 보너스


# ── UI 색상 팔레트 ─────────────────────────────────────────────────────────────
PROD_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#CCB974",
    "#64B5CD", "#76B7B2",
]

OPER_BORDER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5",
]


@dataclass
class Config:
    path:   PathConfig   = field(default_factory=PathConfig)
    env:    EnvConfig    = field(default_factory=EnvConfig)
    rl:     RLConfig     = field(default_factory=RLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


# 싱글톤 인스턴스 – 전 모듈에서 import하여 사용
CONFIG = Config()
