"""
config.py – 전체 프로젝트 설정
모든 경로·환경·RL·보상 파라미터를 한 곳에서 관리합니다.
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── 기본 경로 ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent          # pjt_scheduling/
EXTERNAL_DIR = BASE_DIR / "external"          # 프로젝트 내 DB 연동 데이터 폴더

_INPUT_FOLDER_RE = re.compile(r"^[\w.-]+$")


def validate_input_folder(name: str) -> str:
    """external/ 하위 입력 폴더명 검증 (경로 탐색 방지)"""
    name = name.strip().strip("/\\")
    if not name or not _INPUT_FOLDER_RE.match(name):
        raise ValueError(
            f"입력 폴더명이 올바르지 않습니다: {name!r} "
            "(영문·숫자·_·.·- 만 사용)"
        )
    return name


def list_input_folders() -> list[str]:
    """external/ 아래 사용 가능한 입력 폴더 목록"""
    if not EXTERNAL_DIR.exists():
        return ["input"]
    found: list[str] = []
    for path in EXTERNAL_DIR.iterdir():
        if not path.is_dir() or path.name == "output":
            continue
        if (path / "plan.json").exists() or (path / "schedule.json").exists():
            found.append(path.name)
    current = CONFIG.path.input_folder
    if current not in found:
        found.append(current)
    return sorted(set(found))


def set_input_folder(name: str) -> Path:
    """입력 데이터셋 폴더 설정 → external/{name}/"""
    CONFIG.path.input_folder = validate_input_folder(name)
    return CONFIG.path.input_dir


@dataclass
class PathConfig:
    input_folder: str = "input"   # external/{input_folder}/
    model_dir:    Path = field(default_factory=lambda: BASE_DIR / "models")

    schedule_file:     str = "schedule.json"
    availability_file: str = "availability.json"
    incoming_wip_file: str = "incoming_wip.json"
    plan_file:         str = "plan.json"
    flow_file:         str = "flow.json"

    @property
    def input_dir(self) -> Path:
        return EXTERNAL_DIR / self.input_folder

    @property
    def output_dir(self) -> Path:
        return EXTERNAL_DIR / "output" / self.input_folder


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

_env_input = os.environ.get("SCHEDULING_INPUT", "").strip()
if _env_input:
    set_input_folder(_env_input)
