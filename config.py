"""
config.py – 전체 프로젝트 설정
모든 경로·환경·RL·보상 파라미터를 한 곳에서 관리합니다.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ── 기본 경로 ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent          # pjt_scheduling/
EXTERNAL_DIR = BASE_DIR / "external"          # 프로젝트 내 DB 연동 데이터 폴더
DATASET_DIR  = EXTERNAL_DIR / "dataset"       # external/dataset/{FAC_ID}/...
SQL_DIR      = EXTERNAL_DIR / "sql"           # external/sql/*.sql → *.json

DATASET_SPLITS = ("train", "test", "infer")
PERIOD_SPLITS = ("train", "test")  # infer 제외 – 기간 하위 폴더 없음

# 테이블 RULE_TIMEKEY 형식: YYYYMMDDHHmmss (예: 20260620070000)
RULE_TIMEKEY_FMT = "%Y%m%d%H%M%S"
RULE_TIMEKEY_DEFAULT_HMS = "070000"  # 일별 기본 시각 07:00:00

_PATH_SEGMENT_RE = re.compile(r"^[\w.-]+$")


def validate_path_segment(name: str, label: str = "경로") -> str:
    name = name.strip().strip("/\\")
    if not name or not _PATH_SEGMENT_RE.match(name):
        raise ValueError(
            f"{label}명이 올바르지 않습니다: {name!r} "
            "(영문·숫자·_·.·- 만 사용)"
        )
    return name


def rule_timekey_now() -> str:
    """현재 시각 RULE_TIMEKEY (YYYYMMDDHHmmss)"""
    return datetime.now().strftime(RULE_TIMEKEY_FMT)


def rule_timekey_today(hour: int = 7, minute: int = 0, second: int = 0) -> str:
    """오늘 지정 시각 RULE_TIMEKEY (기본 07:00:00)"""
    now = datetime.now()
    return now.replace(
        hour=hour, minute=minute, second=second, microsecond=0,
    ).strftime(RULE_TIMEKEY_FMT)


def train_snapshot_now() -> str:
    """단일 스냅샷 폴더명 (= rule_timekey_now)"""
    return rule_timekey_now()


def period_today() -> str:
    """일별 기간 폴더명 (= 오늘 07:00:00 RULE_TIMEKEY)"""
    return rule_timekey_today()


def normalize_rule_timekey(value: str) -> str:
    """
    RULE_TIMEKEY 정규화 → 14자리 YYYYMMDDHHmmss
    8자리(YYYYMMDD) → 뒤에 070000 붙임
    12자리(YYYYMMDDHHmm) → 뒤에 00 붙임
    """
    value = value.strip()
    if len(value) == 14 and value.isdigit():
        return value
    if len(value) == 8 and value.isdigit():
        return value + RULE_TIMEKEY_DEFAULT_HMS
    if len(value) == 12 and value.isdigit():
        return value + "00"
    raise ValueError(
        f"RULE_TIMEKEY 형식은 YYYYMMDDHHmmss (14자리) 이어야 합니다: {value!r}"
    )


normalize_period = normalize_rule_timekey


def iter_rule_timekeys(from_key: str, to_key: str):
    """RULE_TIMEKEY 구간의 일별 키 생성 (시·분·초는 시작 키 기준 유지)"""
    start = datetime.strptime(normalize_rule_timekey(from_key), RULE_TIMEKEY_FMT)
    end = datetime.strptime(normalize_rule_timekey(to_key), RULE_TIMEKEY_FMT)
    if end < start:
        raise ValueError(f"종료 RULE_TIMEKEY가 시작보다 앞섭니다: {from_key} ~ {to_key}")
    h, m, s = start.hour, start.minute, start.second
    cur = start
    while cur <= end:
        yield cur.strftime(RULE_TIMEKEY_FMT)
        nxt = cur + timedelta(days=1)
        cur = nxt.replace(hour=h, minute=m, second=s, microsecond=0)


iter_periods = iter_rule_timekeys


def latest_period(fac_id: str, split: str = "train") -> Optional[str]:
    """train|test/{period} 중 최신 period 폴더명"""
    if split not in PERIOD_SPLITS:
        return None
    root = DATASET_DIR / validate_path_segment(fac_id, "FAC_ID") / split
    if not root.is_dir():
        return None
    snaps = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "input").is_dir()
    )
    return snaps[-1] if snaps else None


def latest_train_snapshot(fac_id: str) -> Optional[str]:
    return latest_period(fac_id, "train")


def resolve_dataset_path(
    fac_id: str,
    split: str,
    period: Optional[str] = None,
    *,
    snapshot: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    dataset 경로 해석 → (input_dir, output_dir)
    train/test: dataset/{FAC_ID}/{split}/{period}/input|output
    infer:      dataset/{FAC_ID}/infer/input|output  (기간 하위 폴더 없음)
    """
    period = period or snapshot
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    split = validate_path_segment(split, "split")
    if split not in DATASET_SPLITS:
        raise ValueError(f"split은 {DATASET_SPLITS} 중 하나여야 합니다: {split!r}")

    base = DATASET_DIR / fac_id / split
    if split in PERIOD_SPLITS:
        per = period or latest_period(fac_id, split)
        if not per:
            per = period_today() if split == "test" else rule_timekey_now()
        per = validate_path_segment(normalize_rule_timekey(per), "RULE_TIMEKEY")
        root = base / per
    else:
        root = base

    return root / "input", root / "output"


def infer_paths(fac_id: str) -> Tuple[Path, Path]:
    """추론 전용 dataset/{FAC_ID}/infer/input|output"""
    return resolve_dataset_path(fac_id, "infer")


def parse_input_folder(folder: str) -> Tuple[str, str, Optional[str]]:
    """
    입력 폴더 키 파싱
      FAC001/train/20260620070000  → train 기간 (RULE_TIMEKEY)
      FAC001/test/20260620070000   → test 기간
      FAC001/infer              → infer (기간 없음)
    """
    parts = [validate_path_segment(p, "경로") for p in folder.strip("/\\").split("/")]
    if len(parts) == 2:
        fac_id, split = parts
        if split in PERIOD_SPLITS:
            per = latest_period(fac_id, split)
            return fac_id, split, per
        return fac_id, split, None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        f"입력 폴더 형식: FAC_ID/split 또는 FAC_ID/{{train|test}}/period (받은 값: {folder!r})"
    )


def list_input_folders() -> List[str]:
    """사용 가능한 dataset 입력 경로 키 목록"""
    found: List[str] = []
    if not DATASET_DIR.is_dir():
        return [CONFIG.path.input_folder_key]

    for fac_path in sorted(DATASET_DIR.iterdir()):
        if not fac_path.is_dir():
            continue
        fac_id = fac_path.name
        for split in DATASET_SPLITS:
            split_path = fac_path / split
            if not split_path.is_dir():
                continue
            if split in PERIOD_SPLITS:
                for per_path in sorted(split_path.iterdir()):
                    if per_path.is_dir() and (per_path / "input").is_dir():
                        found.append(f"{fac_id}/{split}/{per_path.name}")
            elif (split_path / "input").is_dir():
                found.append(f"{fac_id}/{split}")

    current = CONFIG.path.input_folder_key
    if current not in found:
        found.append(current)
    return sorted(set(found))


def set_input_folder(folder: str) -> Path:
    """입력 dataset 경로 설정"""
    fac_id, split, period = parse_input_folder(folder)
    CONFIG.path.fac_id = fac_id
    CONFIG.path.dataset_split = split
    CONFIG.path.train_snapshot = period or ""
    return CONFIG.path.input_dir


@dataclass
class PathConfig:
    fac_id: str = "FAC001"
    dataset_split: str = "train"          # train | test | infer
    train_snapshot: str = ""              # train 전용; 비어 있으면 최신 스냅샷

    model_dir: Path = field(default_factory=lambda: BASE_DIR / "models")

    schedule_file:     str = "schedule.json"
    discrete_arrange_file: str = "discrete_arrange.json"
    abstract_arrange_file: str = "abstract_arrange.json"
    incoming_wip_file: str = "incoming_wip.json"
    plan_file:         str = "plan.json"
    flow_file:         str = "flow.json"
    split_file:        str = "split.json"
    lot_master_file:   str = "lot_master.json"
    tool_capacity_file: str = "tool_capacity.json"
    lot_route_file:    str = "lot_route.json"

    @property
    def input_folder_key(self) -> str:
        if self.dataset_split in PERIOD_SPLITS:
            per = self.train_snapshot or latest_period(self.fac_id, self.dataset_split)
            if not per:
                per = period_today() if self.dataset_split == "test" else rule_timekey_now()
            return f"{self.fac_id}/{self.dataset_split}/{per}"
        return f"{self.fac_id}/{self.dataset_split}"

    @property
    def input_dir(self) -> Path:
        per = self.train_snapshot or None if self.dataset_split in PERIOD_SPLITS else None
        inp, _ = resolve_dataset_path(self.fac_id, self.dataset_split, per)
        return inp

    @property
    def output_dir(self) -> Path:
        per = self.train_snapshot or None if self.dataset_split in PERIOD_SPLITS else None
        _, out = resolve_dataset_path(self.fac_id, self.dataset_split, per)
        return out

    @property
    def infer_input_dir(self) -> Path:
        return infer_paths(self.fac_id)[0]

    @property
    def infer_output_dir(self) -> Path:
        return infer_paths(self.fac_id)[1]

    @property
    def sql_dir(self) -> Path:
        return SQL_DIR


# SQL 파일명 ↔ JSON 파일명 (loader.fetch_from_db)
SQL_JSON_MAP = {
    "discrete_arrange": ("discrete_arrange.sql", "discrete_arrange.json"),
    "abstract_arrange": ("abstract_arrange.sql", "abstract_arrange.json"),
    "plan":             ("plan.sql",             "plan.json"),
    "flow":             ("flow.sql",             "flow.json"),
    "split":            ("split.sql",            "split.json"),
}


@dataclass
class OracleConfig:
    user: str = field(default_factory=lambda: os.environ.get("ORACLE_USER", ""))
    password: str = field(default_factory=lambda: os.environ.get("ORACLE_PASSWORD", ""))
    dsn: str = field(default_factory=lambda: os.environ.get("ORACLE_DSN", ""))
    # 추가 WHERE 바인드 (JSON 문자열 또는 key=value)
    extra_binds: dict = field(default_factory=dict)


@dataclass
class EnvConfig:
    max_eqp_count:    int = 10
    max_oper_count:   int = 15
    max_prod_count:   int = 10
    max_model_count:  int = 4           # bucket = (ppk, model, oper)의 model 축 K
    max_queue_size:   int = 20          # legacy compat
    sim_time_horizon: int = 1440
    hard_horizon_minutes: int = 1440    # 07:00 → 익일 07:00
    soft_cutoff_minutes:  int = 1320    # 익일 05:00
    conversion_minutes:   int = 60      # LOT_CD/TEMP 변경 시 setup


@dataclass
class RLConfig:
    algorithm:       str   = "PPO"
    learning_rate:   float = 3e-4
    n_steps:         int   = 2048
    batch_size:      int   = 64
    n_epochs:        int   = 10
    gamma:           float = 0.99
    total_timesteps: int   = 200_000
    default_n_episodes:   int   = 100       # UI 에피소드 학습 기본값
    model_name:      str   = "scheduling_rl"
    eval_freq:       int   = 10_000


@dataclass
class RewardConfig:
    w_same_oper:       float = 2.0
    w_same_prod:       float = 0.5       # 같은 PPK 재공이 남아 있을 때만 (조건부)
    w_prod_switch:     float = 0.8       # 이전 PPK 재공 고갈 시 전환 보너스
    w_idle_per_min:    float = -0.5
    w_completion:      float = 1.0
    w_plan_hit:        float = 5.0
    w_pacing:          float = 3.0         # 계획 있는 (PPK, OPER)만 적용
    w_conversion:      float = -30.0    # LOT_CD/TEMP 전환 1회 패널티
    w_late_finish:     float = -2.0     # soft cutoff(05:00) 이후 END_TM


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
    oracle: OracleConfig = field(default_factory=OracleConfig)
    env:    EnvConfig    = field(default_factory=EnvConfig)
    rl:     RLConfig     = field(default_factory=RLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


CONFIG = Config()

_env_input = os.environ.get("SCHEDULING_INPUT", "").strip()
if _env_input:
    set_input_folder(_env_input)
