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
DATA_DIR     = BASE_DIR / "data"
DATASET_DIR  = DATA_DIR / "dataset"         # data/dataset/{FAC_ID}/...
SQL_DIR      = DATA_DIR / "sql"              # data/sql/*.sql → *.json
SQL_EXAMPLE_DIR = DATA_DIR / "sql.example"   # Oracle 쿼리 템플릿 (참고용)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass


_load_dotenv()

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


def format_missing_input_file_error(input_dir: Path, filename: str = "discrete_arrange.json") -> str:
    """
    입력 JSON 미발견 시 진단 메시지.
    파일이 input/ 밖(상위 폴더·output·data/sql)에 있는 경우 힌트를 붙입니다.
    """
    input_dir = Path(input_dir)
    try:
        resolved_input = input_dir.resolve()
    except OSError:
        resolved_input = input_dir.absolute()
    target = resolved_input / filename
    lines = [f"입력 파일 없음: {target}"]

    parent_period = resolved_input.parent
    wrong_parent = parent_period / filename
    if wrong_parent.is_file():
        lines.append(
            f"※ {filename}이 input/ 밖 상위 폴더에 있습니다: {wrong_parent}\n"
            f"  → 올바른 위치: {resolved_input / filename}"
        )

    output_dir = parent_period / "output"
    wrong_output = output_dir / filename
    if wrong_output.is_file():
        lines.append(
            f"※ output/ 폴더에만 있습니다: {wrong_output}\n"
            f"  → input/ 으로 복사하거나 fetch/sample을 다시 실행하세요."
        )

    sql_copy = SQL_DIR / filename
    if sql_copy.is_file():
        lines.append(
            f"※ data/sql/ 에는 있으나 dataset input 에는 없습니다: {sql_copy}\n"
            f"  → fetch 결과는 dataset/.../input/ 에 저장되어야 합니다."
        )

    if not resolved_input.is_dir():
        lines.append(f"※ input 폴더 자체가 없습니다: {resolved_input}")
        if parent_period.is_dir():
            siblings = sorted(p.name for p in parent_period.iterdir())
            lines.append(f"  상위({parent_period.name}) 내용: {', '.join(siblings) or '(비어 있음)'}")
    else:
        names = sorted(p.name for p in resolved_input.iterdir())
        lines.append(f"※ input/ 안 파일: {', '.join(names) if names else '(비어 있음)'}")
        for name in names:
            if name.lower() == filename.lower() and name != filename:
                lines.append(f"※ 대소문자 다른 파일명: {name} (필요: {filename})")

    lines.append("python main.py sample 또는 python main.py fetch 로 데이터를 생성하세요.")
    return "\n".join(lines)


def rule_timekey_now() -> str:
    """현재 시각 RULE_TIMEKEY (YYYYMMDDHHmmss)"""
    return datetime.now().strftime(RULE_TIMEKEY_FMT)


def rule_timekey_today(hour: int = 7, minute: int = 0, second: int = 0) -> str:
    """오늘 지정 시각 RULE_TIMEKEY (기본 07:00:00)"""
    now = datetime.now()
    return now.replace(
        hour=hour, minute=minute, second=second, microsecond=0,
    ).strftime(RULE_TIMEKEY_FMT)


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


def resolve_train_period_range(
    *,
    prevcnt: Optional[int] = None,
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    학습용 RULE_TIMEKEY 구간 해석.
    --prevcnt N: 현재 시각 기준 최근 N일(포함)
    --from / --to: 명시 구간
    """
    if prevcnt is not None:
        if from_key or to_key:
            raise ValueError("--prevcnt 와 --from/--to 는 함께 쓸 수 없습니다.")
        if prevcnt < 1:
            raise ValueError("--prevcnt 는 1 이상이어야 합니다.")
        end = rule_timekey_today()
        end_dt = datetime.strptime(end, RULE_TIMEKEY_FMT)
        start_dt = end_dt - timedelta(days=prevcnt - 1)
        return start_dt.strftime(RULE_TIMEKEY_FMT), end

    if from_key and to_key:
        return normalize_rule_timekey(from_key), normalize_rule_timekey(to_key)
    if from_key or to_key:
        raise ValueError("--from 와 --to 를 함께 지정하세요.")
    raise ValueError("--prevcnt 또는 --from/--to 가 필요합니다.")


def resolve_infer_rule_timekey(
    fac_id: str,
    rule_timekey: Optional[str] = None,
    *,
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
    prevcnt: Optional[int] = None,
    nodb: bool = False,
) -> str:
    """
    추론 SQL 조회용 RULE_TIMEKEY (단일값).

    --ruletimekey: 명시 지정
    --from / --to: 구간 내 RULE_TIMEKEY 를 DB BETWEEN 으로 조회 후 최신값 사용
    --prevcnt: 최신 기준 최근 N개 RULE_TIMEKEY 조회 후 최신값 사용
    미지정 시: DB 최신 → 로컬 폴더 최신 → 현재 시각
    """
    if rule_timekey:
        return normalize_rule_timekey(rule_timekey)

    if from_key or to_key or prevcnt is not None:
        if bool(from_key) != bool(to_key):
            raise ValueError("--from 와 --to 를 함께 지정하세요.")
        if (from_key and to_key) and prevcnt is not None:
            raise ValueError("--prevcnt 와 --from/--to 는 함께 쓸 수 없습니다.")
        from data.loader.rule_timekey_query import resolve_collect_periods
        periods, _ = resolve_collect_periods(
            fac_id,
            prevcnt=prevcnt or 1,
            from_key=from_key,
            to_key=to_key,
            require_db=not nodb,
        )
        if not periods:
            raise ValueError(f"해당 구간에 RULE_TIMEKEY 가 없습니다 (FAC_ID={fac_id}).")
        return periods[-1]

    try:
        from data.loader.rule_timekey_query import fetch_latest_rule_timekey
        db_key = fetch_latest_rule_timekey(fac_id)
        if db_key:
            return db_key
    except Exception:
        pass
    return (
        latest_period(fac_id, "test")
        or latest_period(fac_id, "train")
        or rule_timekey_now()
    )


def list_split_folders(fac_id: str, split: str) -> List[str]:
    """dataset/{FAC}/{split}/... 입력 폴더 키 목록 (정렬)."""
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    prefix = f"{fac_id}/{split}/"
    if split not in PERIOD_SPLITS:
        key = f"{fac_id}/{split}"
        inp = DATASET_DIR / fac_id / split / "input"
        return [key] if inp.is_dir() else []
    return sorted(
        f for f in list_input_folders()
        if f.startswith(prefix)
    )


def folders_in_period_range(
    fac_id: str,
    split: str,
    from_key: str,
    to_key: str,
) -> List[str]:
    """폴더 RULE_TIMEKEY가 [from_key, to_key] 시각 구간에 들어가면 포함 (이름 exact match 아님)."""
    start = datetime.strptime(normalize_rule_timekey(from_key), RULE_TIMEKEY_FMT)
    end = datetime.strptime(normalize_rule_timekey(to_key), RULE_TIMEKEY_FMT)
    matched: List[str] = []
    for folder in list_split_folders(fac_id, split):
        period = folder.rsplit("/", 1)[-1]
        try:
            dt = datetime.strptime(normalize_rule_timekey(period), RULE_TIMEKEY_FMT)
        except ValueError:
            continue
        if start <= dt <= end:
            matched.append(folder)
    return matched


def resolve_train_folders(
    fac_id: str,
    from_key: str,
    to_key: str,
    *,
    prevcnt: Optional[int] = None,
    split: str = "train",
) -> List[str]:
    """
    학습/검증용 period 폴더 목록 (이미 수집된 dataset 기준).
    RULE_TIMEKEY 구간과 일치하는 폴더만 반환합니다.
    """
    del prevcnt
    return folders_in_period_range(fac_id, split, from_key, to_key)


def train_folders_for_periods(
    fac_id: str, periods: List[str], split: str = "train",
) -> List[str]:
    """DB 등에서 확정된 RULE_TIMEKEY 목록에 해당하는 period 폴더 키."""
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    available = set(list_split_folders(fac_id, split))
    folders: List[str] = []
    for period in periods:
        key = normalize_rule_timekey(period)
        folder = f"{fac_id}/{split}/{key}"
        if folder in available:
            folders.append(folder)
    return folders


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
    """사용 가능한 dataset 입력 경로 키 목록 (실제 input JSON 있는 경로만)"""
    found: List[str] = []
    if not DATASET_DIR.is_dir():
        return found

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
                    inp = per_path / "input"
                    if inp.is_dir() and (inp / CONFIG.path.discrete_arrange_file).is_file():
                        found.append(f"{fac_id}/{split}/{per_path.name}")
            elif (split_path / "input").is_dir() and (
                (split_path / "input" / CONFIG.path.discrete_arrange_file).is_file()
            ):
                found.append(f"{fac_id}/{split}")

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

    discrete_arrange_file: str = "discrete_arrange.json"
    abstract_arrange_file: str = "abstract_arrange.json"
    plan_file:         str = "plan.json"
    flow_file:         str = "flow.json"
    split_file:        str = "split.json"
    lot_master_file:   str = "lot_master.json"
    tool_capacity_file: str = "tool_capacity.json"
    eqp_initial_state_file: str = "eqp_initial_state.json"
    batch_info_file:   str = "batch_info.json"
    conversion_group_file: str = "conversion_group.json"
    eqp_conv_plan_file: str = "eqp_conv_plan.json"
    eqp_down_file:     str = "eqp_down.json"
    output_file:       str = "output.json"

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
# 각 SQL 첫 줄: -- @db: Prd / Dev / Prd.Plan  (config/databases.yaml 계층)
# 메타 SQL (JSON 변환 없음): rule_timekey_latest.sql, rule_timekey_list.sql, rule_timekey_recent.sql
REQUIRED_SQL_JSON_MAP = {
    "discrete_arrange": ("discrete_arrange.sql", "discrete_arrange.json"),
    "abstract_arrange": ("abstract_arrange.sql", "abstract_arrange.json"),
    "plan":             ("plan.sql",             "plan.json"),
    "flow":             ("flow.sql",             "flow.json"),
    "split":            ("split.sql",            "split.json"),
    "batch_info":       ("batch_info.sql",       "batch_info.json"),
}
OPTIONAL_SQL_JSON_MAP = {
    "lot_master":        ("lot_master.sql",        "lot_master.json"),
    "tool_capacity":     ("tool_capacity.sql",     "tool_capacity.json"),
    "eqp_initial_state": ("eqp_initial_state.sql", "eqp_initial_state.json"),
    "eqp_conv_plan":     ("eqp_conv_plan.sql",     "eqp_conv_plan.json"),
    "eqp_down":          ("eqp_down.sql",          "eqp_down.json"),
}
SQL_JSON_MAP = {**REQUIRED_SQL_JSON_MAP, **OPTIONAL_SQL_JSON_MAP}
SQL_REQUIRED_KEYS = frozenset(REQUIRED_SQL_JSON_MAP)


@dataclass
class OracleConfig:
    user: str = field(default_factory=lambda: os.environ.get("ORACLE_USER", ""))
    password: str = field(default_factory=lambda: os.environ.get("ORACLE_PASSWORD", ""))
    dsn: str = field(default_factory=lambda: os.environ.get("ORACLE_DSN", ""))
    # 추가 WHERE 바인드 (JSON 문자열 또는 key=value)
    extra_binds: dict = field(default_factory=dict)


@dataclass
class EnvConfig:
    max_oper_count:   int = 3          # RL action/obs 고정 축 O
    max_prod_count:   int = 10          # RL action/obs 고정 축 P
    max_model_count:  int = 5           # bucket = (ppk, model, oper)의 model 축 K
    hard_horizon_minutes: int = 1440    # 07:00 → 익일 07:00
    soft_cutoff_minutes:  int = 1320    # 익일 05:00
    conversion_minutes:   int = 60      # LOT_CD/TEMP 변경 시 setup
    max_conversions: Optional[int] = None          # 시뮬 전체 전환 상한 (None=무제한)
    max_conversions_per_eqp: Optional[int] = None  # EQP별 전환 상한 (None=무제한)
    # RTS_EQPCONVPLAN_INF/HIS 테이블 저장 여부 (옵션). False면 두 테이블 모두
    # 비워서 출력한다(간트/API 응답의 conversion_plans는 영향 없이 항상 전체가 보임).
    conv_output_enabled: bool = True
    # RTS_EQPCONVPLAN_INF/HIS 출력에는 RULE_TIMEKEY 기준 이 시간(분) 이내에
    # 시작하는 전환만 포함한다. 그보다 먼 미래의 전환은 재계획 여지가 커
    # 추측성이므로 확정 출력에 싣지 않는다(간트/API 응답의 conversion_plans는 영향 없음).
    conv_output_window_minutes: int = 60


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
    # GPU / 병렬 환경 설정
    device:          str   = "auto"         # "auto" | "cuda" | "cpu"
    n_envs:          int   = 1              # SubprocVecEnv 병렬 환경 수 (1=DummyVecEnv)


@dataclass
class RewardConfig:
    # --- Step A: 스케일 재정규화 (모든 항을 ±5 band로, step reward clip) ---
    # 제품·공정이 '모두' 직전과 동일할 때만 주는 연속 보너스 (전환 회피와 정렬).
    w_same_setup:      float = 1.0
    w_idle_per_min:    float = 0.0       # [제거] idle 분당 (1·2·6·7만 유지)
    w_plan_hit:        float = 0.0       # [제거] 달성 진척 (cover 무시 → 전담 방해 1위라 제거)
    w_pacing:          float = 2.5       # 선형 takt 추종 (achievable 기준; Step C) ↑강화
    # pacing 진척을 'done'이 아니라 'done + 다른 장비(본인 제외)의 잔여 horizon 투영
    # 생산(coverage)'으로 봄 → 이미 다른 장비가 충분히 덮는 제품은 pace 충족으로 간주해
    # 추가 몰림(lockstep)을 억제. 0이면 done-only(기존 동작). 작게 시작 권장.
    pacing_coverage_scale: float = 1.0
    w_conversion:      float = -10.0     # LOT_CD/TEMP 전환 1회 패널티
    # 회피 가능한 conversion(이미 같은 LOT_CD/TEMP로 세팅된 다른 장비가 시간 내
    # 재공을 커버 가능)일 때 w_conversion에 더해지는 추가 패널티. 0이면 비활성.
    w_avoidable_conversion: float = -8.0
    # 변환 후 무변환으로 돌 가동시간이 (이 배수 × conversion_minutes) 이상이면
    # 변환을 정당한 것으로 보고 회피 패널티를 면제. 짧으면 패널티↑.
    conversion_amortize_factor: float = 1.0
    # --- 벌크 점유(Bulk-Fill) 전용 보상항 (SchedulingRLEnv에서만 사용; 0이면 비활성) ---
    # ① 블록 크기 보너스: 같은 제품군을 큰 블록으로 커밋할수록 보상(전담 커밋 유도).
    w_bulk_block_bonus:   float = 0.0
    # ② 전용 오용 페널티(<0): 범용 장비가 더 전용적인 idle 장비도 가능한 버킷을
    #    잡을 때 감점 → 전용 장비가 놀지 않게.
    w_dedication_misuse:  float = 0.0
    # ③ 중복 커버 페널티(<0): 이미 다른 셋업 장비가 horizon 내 충분히 덮는 버킷을
    #    잡으면 감점 → 다른 제품으로 전환할 '용기'.
    w_redundant_cover:    float = 0.0
    # --- Step B: flow-balance shaping (WIP 비중 vs 계획 비중 기준) ---
    w_flow_balance:    float = 0.0       # [제거] cover 무시 → 전담 방해 2위라 제거
    # 후속 ready WIP / 후속 장비 합산 분당 처리량(매/분) ≤ 이 값(분)일 때만 feeding 보너스
    flow_balance_starving_cover_min: float = 120.0
    # --- Step A: step reward clip 범위 (PPO advantage 안정화) ---
    reward_clip:       float = 10.0
    # --- Step C: achievable target 사용 여부 (재공 한계까지만 계획 추종) ---
    use_achievable_target: bool = True


def reward_params_dict(reward: Optional[RewardConfig] = None) -> dict:
    """RewardConfig → API/UI 공유 dict."""
    r = reward or CONFIG.reward
    return {
        "w_same_setup": r.w_same_setup,
        "w_idle_per_min": r.w_idle_per_min,
        "w_plan_hit": r.w_plan_hit,
        "w_pacing": r.w_pacing,
        "pacing_coverage_scale": r.pacing_coverage_scale,
        "w_conversion": r.w_conversion,
        "w_avoidable_conversion": r.w_avoidable_conversion,
        "conversion_amortize_factor": r.conversion_amortize_factor,
        "w_bulk_block_bonus": r.w_bulk_block_bonus,
        "w_dedication_misuse": r.w_dedication_misuse,
        "w_redundant_cover": r.w_redundant_cover,
        "w_flow_balance": r.w_flow_balance,
        "flow_balance_starving_cover_min": r.flow_balance_starving_cover_min,
        "reward_clip": r.reward_clip,
        "use_achievable_target": r.use_achievable_target,
    }


def apply_reward_params(params: dict) -> None:
    """학습 요청 파라미터 → CONFIG.reward 반영."""
    r = CONFIG.reward
    float_keys = (
        "w_same_setup", "w_idle_per_min",
        "w_plan_hit", "w_pacing", "pacing_coverage_scale", "w_conversion",
        "w_avoidable_conversion", "conversion_amortize_factor",
        "w_bulk_block_bonus", "w_dedication_misuse", "w_redundant_cover",
        "w_flow_balance", "flow_balance_starving_cover_min", "reward_clip",
    )
    for key in float_keys:
        if key in params and params[key] is not None:
            setattr(r, key, float(params[key]))
    if "use_achievable_target" in params and params["use_achievable_target"] is not None:
        r.use_achievable_target = bool(params["use_achievable_target"])


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
