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


# ── 전환 그룹(conversion group) ────────────────────────────────────────────────
# FAC_ID별 하드코딩 설정. (LOT_CD, TEMP)를 GROUP_ID로 묶어, 같은 그룹끼리만
# 장비 전환을 허용한다(simulation/simulator.py:_conversion_group_blocks).
# split/period(train/test/infer)와 무관하게 fac 전체에 공통 적용된다.
# 행 포맷: {"GROUP_ID": str, "LOT_CD": str, "TEMP": str}
CONVERSION_GROUPS: dict = {
    # "FAC001": [
    #     {"GROUP_ID": "G1", "LOT_CD": "LOTCODE_A", "TEMP": "T1"},
    #     {"GROUP_ID": "G1", "LOT_CD": "LOTCODE_B", "TEMP": "T1"},
    #     {"GROUP_ID": "G2", "LOT_CD": "LOTCODE_C", "TEMP": ""},
    # ],
}


def conversion_group_rows_for(fac_id: str) -> List[dict]:
    """FAC_ID에 해당하는 전환 그룹 행 목록(CONVERSION_GROUPS). 미설정이면 빈 리스트
    → 전환 그룹 제약 비활성(기존 동작 유지)."""
    return CONVERSION_GROUPS.get(fac_id, [])


def model_dir_for(fac_id: Optional[str] = None) -> Path:
    """FAC_ID별 RL 모델 저장 경로: models/{FAC_ID}/.

    fac_id 미지정(None/빈 문자열)이면 기존 공용 경로(CONFIG.path.model_dir,
    models/ 바로 아래)를 그대로 반환해 이전에 학습된 모델과 호환된다 —
    FAC_ID별 분리는 호출부가 명시적으로 fac_id를 넘길 때만 적용된다.
    """
    if not fac_id:
        return CONFIG.path.model_dir
    return CONFIG.path.model_dir / validate_path_segment(fac_id, "FAC_ID")


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
    # False면 LOT_STAT_CD=WAIT인 discrete_arrange 행의 discrete 정보(특정 EQP 고정,
    # 실측 ST)를 배정 로직에서 쓰지 않고 abstract 매칭 경로(모델 평균 ST, 모델이
    # 맞는 아무 장비)만 태운다. LOT의 수량/제품/공정 정체성(WIP 카운트)은 그대로
    # 유지된다 — PROC/LOAD/RESV/SELE(강제 배정) LOT은 이 옵션과 무관하게 항상
    # discrete 그대로 배정된다.
    discrete_wait_enabled: bool = True
    # SchedulingRLEnv 전용 '결정 컨텍스트' 관측 8채널 추가 여부.
    # simulator.get_observation()은 결정 중인 장비를 모델 인덱스로만 흘려줘서,
    # 같은 모델 장비가 여러 대인 대칭 케이스에서 "내가 어느 장비이고 지금 블록
    # 중인지"를 구분할 수 없었다(= 전담 정책이 관측만으로 표현 불가능).
    # 자세한 채널 정의는 env/scheduling_rl_env.py의 RL_EXTRA_OBS_DIM 주석 참고.
    # False로 두면 이전 obs_dim(936)으로 학습된 모델과 호환된다.
    rl_extra_obs: bool = True


@dataclass
class RLConfig:
    algorithm:       str   = "PPO"
    learning_rate:   float = 3e-4
    # n_steps=2048은 이 문제의 에피소드 길이(수십~수백 스텝)에 비해 지나치게
    # 길다. 데이터셋 8개를 co-train하면 롤아웃 1개가 8×2048=16,384 transition이
    # 되어 total_timesteps=200k에서 PPO 업데이트가 **12번**밖에 일어나지 않는다
    # (학습 곡선이 평평해 보이는 직접적 원인 중 하나). 512로 낮추면 같은
    # 예산에서 업데이트 횟수가 4배가 된다.
    n_steps:         int   = 512
    batch_size:      int   = 64
    gamma:           float = 0.99
    # 엔트로피 보너스: 0.0(기본 SB3 값)이면 대칭 케이스(예: SYM_5x5)에서 정책이
    # 조기에 하나의 순열로 굳어져(collapse) 전환 0회 최적해를 못 찾고 국소 최적에
    # 갇히는 경향이 실험으로 확인됨. ent_coef를 고정값으로만 주면(과거 0.02) 학습
    # 후반(21만 스텝 이후)에도 여전히 전환 2회에서 정체되는 것도 확인됨 — 그래서
    # start(탐색)→final(착취)로 선형 감쇠시킨다(EntropyDecayCallback,
    # agent/train_progress.py). ent_coef_final을 ent_coef와 같은 값으로 두면
    # 감쇠 없이 기존처럼 고정값 그대로 동작한다.
    # BC 워밍스타트를 쓰는 경우(기본) 시작값을 크게 잡으면 안 된다 — PPO 첫
    # 업데이트에서 advantage가 아직 신뢰할 수 없는 동안 엔트로피 항만 유효
    # 그래디언트를 내면서 복제된 정책을 균등분포 쪽으로 되돌려버린다
    # (= "학습할수록 나빠짐"의 주범 중 하나). 0.05 → 0.01로 낮췄다.
    ent_coef:        float = 0.01   # 학습 시작 시점 엔트로피 계수
    ent_coef_final:  float = 0.001  # 학습 종료 시점 엔트로피 계수(선형 감쇠 목표)
    # 감쇠 구간. 과거에는 0.2(초반 20%에 0까지)로 짧게 잡았는데, 그건 "탐색이
    # BC 정책을 흔들어 망가뜨린다"를 엔트로피를 죽여서 막으려던 대증요법이었다.
    # 이제는 expert_anchor가 그 역할을 제대로 하므로(전문가에서 멀어지는 것만
    # 직접 억제), 탐색은 오래 살려둬야 dedication '위'를 찾을 수 있다.
    ent_coef_decay_fraction: float = 0.8
    total_timesteps: int   = 200_000
    default_n_episodes:   int   = 100       # UI 에피소드 학습 기본값
    model_name:      str   = "scheduling_rl"
    eval_freq:       int   = 10_000
    # GPU / 병렬 환경 설정
    device:          str   = "auto"         # "auto" | "cuda" | "cpu"
    n_envs:          int   = 1              # SubprocVecEnv 병렬 환경 수 (1=DummyVecEnv)

    # --- PPO 안정화 하이퍼파라미터 -------------------------------------------
    gae_lambda:      float = 0.95
    clip_range:      float = 0.2
    vf_coef:         float = 0.5
    max_grad_norm:   float = 0.5
    # 업데이트 1회가 정책을 너무 멀리 옮기면 그 epoch를 즉시 중단(SB3 내장).
    # BC로 얻어둔 좋은 정책이 한 번의 큰 업데이트로 무너지는 걸 막는 안전장치.
    # None이면 비활성.
    target_kl:       Optional[float] = 0.03
    # 예전 기본값 10은 같은 롤아웃을 10번 재사용해 off-policy drift가 컸다.
    # 4로 낮추고 target_kl과 함께 쓰면 업데이트당 이동량이 훨씬 안정적이다.
    n_epochs:        int   = 4
    lr_schedule:     str  = "linear"        # "linear"(3e-4→0) | "constant"
    # 관측 936차원(대부분 0)에 [64,64]는 표현력이 부족했다.
    policy_net_arch: tuple = (256, 256)
    seed:            Optional[int] = None
    # 보상 정규화(VecNormalize(norm_reward=True)). step reward가 ±20이고
    # 에피소드가 수백~수천 스텝이라 return이 수백~수천 규모가 되는데, 이걸
    # 그대로 두면 value loss가 폭주해 explained_variance≈0 → advantage가 사실상
    # 잡음이 되고 학습 곡선이 x축과 평행해진다. 관측은 이미 [0,1]로 정규화돼
    # 있어 norm_obs는 끈다(추론 시 통계 없이도 동일 동작 보장).
    normalize_reward: bool = True
    # 종단 보상(w_terminal_*)까지 신호가 역전파되려면 유효 horizon이 에피소드
    # 길이에 가까워야 한다. 0.99(유효 horizon≈100 스텝)로는 에피소드 끝의
    # 생산량 신호가 초반 결정까지 전혀 도달하지 못한다.
    gamma:           float = 0.997

    # --- 모방학습(BC) 워밍스타트 ---------------------------------------------
    # DedicationAgent 시연으로 정책을 지도학습한 뒤 PPO를 이어서 돌린다.
    # 0이면 비활성(무작위 초기화 그대로 PPO 시작).
    #
    # 과거 구현의 문제와 대응:
    #  · 시연이 데이터셋당 결정적 1 에피소드(=수십 스텝)뿐이라 표본이 절대적으로
    #    부족했다 → bc_episodes_per_dataset / bc_noise_eps 로 DAgger식 상태분포
    #    확장(무작위 교란 후 전문가 라벨 재부여)을 한다.
    #  · 정책만 학습하고 크리틱(value head)은 무작위로 남겨둬서, PPO 첫
    #    업데이트의 advantage가 순수 잡음이 되어 복제 정책을 즉시 파괴했다
    #    → bc_value_coef 로 크리틱도 같이 회귀 학습한다(가장 중요한 수정).
    #  · full-batch NLL을 오래 돌리면 엔트로피≈0으로 결정적 붕괴 →
    #    미니배치 + 엔트로피 정규화(bc_entropy_coef) + 검증 조기종료로 막는다.
    # epochs는 상한이고 실제 종료는 검증 NLL 조기종료가 정한다(과적합/결정적
    # 붕괴 직전에서 멈춤) — 넉넉히 주고 조기종료에 맡기는 편이 안전하다.
    bc_pretrain_epochs: int   = 200
    bc_pretrain_lr:     float = 3e-4
    bc_batch_size:      int   = 256
    bc_episodes_per_dataset: int = 16       # 데이터셋당 시연 에피소드 수
    bc_noise_eps:       float = 0.15        # 교란(무작위 feasible 행동) 비율
    bc_val_fraction:    float = 0.15        # 검증 분할 비율(조기종료 판정용)
    bc_entropy_coef:    float = 0.01        # 결정적 붕괴 방지
    bc_value_coef:      float = 0.5         # 크리틱 동시 학습 가중치
    bc_early_stop_patience: int = 8
    # BC 시연자 후보. 2개 이상이면 시나리오마다 실제로 굴려보고 KPI가 가장
    # 좋은 전문가를 그 시나리오의 시연자로 쓴다(agent/experts.py). 전담이 늘
    # 최선은 아니라서 — 처리시간이 이질적인 시나리오에서는 Earliest-ST 계열이
    # 전담의 2배 점수를 냈다. ["dedication"]으로 두면 기존처럼 전담 고정.
    bc_expert_candidates: tuple = ("dedication", "earliest_st")

    # --- 전문가 앵커(PPO 중 BC 보조손실) -------------------------------------
    # PPO 롤아웃마다 전문가 시연에 대해 BC 그래디언트를 몇 스텝 섞어, 정책이
    # RL 잡음으로 전문가 정책에서 파국적으로 이탈(catastrophic forgetting)하는
    # 걸 막는다. 계수는 학습 진행에 따라 start→final로 선형 감쇠 —
    # 초반엔 전담 정책을 붙잡고, 후반엔 RL이 자유롭게 그 위를 개선한다.
    # 0이면 비활성.
    expert_anchor_coef:  float = 1.0
    expert_anchor_final: float = 0.0
    expert_anchor_decay_fraction: float = 0.6
    expert_anchor_steps: int = 4            # 롤아웃당 BC 그래디언트 스텝 수
    expert_anchor_batch: int = 256

    # --- KPI 기반 best 모델 선택 ---------------------------------------------
    # EvalCallback의 '평균 보상'은 shaping 항이 섞인 대리지표라, 보상이 가장
    # 높은 체크포인트가 벤치마크 KPI(생산량·전환수) 최고와 일치하지 않는다.
    # True면 eval 시점마다 실제 KPI로 채점해 best_model을 고른다.
    kpi_eval_enabled: bool = True
    # KPI 점수 = 생산량(sim_end 내 완료 carrier) − kpi_conversion_weight × 전환수
    kpi_conversion_weight: float = 1.0
    # 학습 시작 시 DedicationAgent 베이스라인 KPI를 측정해 로그·차트 기준선으로
    # 남기고, 최종 모델이 베이스라인에 못 미치면 경고한다.
    kpi_baseline_enabled: bool = True

    # --- 학습/추론 환경 일치 --------------------------------------------------
    # 학습 시뮬레이터는 지금까지 simulator 기본값(termination_mode="all_wip",
    # enable_wip_inflow=True)으로 돌았는데, 추론(inference/runner.py)은
    # termination_mode="current_wip_assigned" + enable_wip_inflow=False 로
    # 돈다. 즉 정책이 학습한 MDP와 실제로 평가받는 MDP가 달랐다(train/serve
    # skew) — 학습은 잘 되는데 벤치마크에서 무너지는 전형적 원인.
    # True면 학습용 데이터셋 복사본에 추론과 같은 플래그를 채워 넣는다
    # (데이터셋이 값을 명시했으면 그 값을 존중).
    align_train_env_with_inference: bool = True
    # 추론은 truncate_on_time=False(= sim_end를 지나서도 남은 재공을 계속 배정)로
    # 도는데 학습 env는 True라 sim_end에서 잘렸다. 그래서 정책은 컷오프 이후
    # 구간을 학습한 적이 없는데, 벤치마크는 그 구간에서 일어난 전환까지 전부
    # 점수에 반영한다(실측: 같은 모델의 학습 KPI 전환 63회 ↔ 벤치마크 83회).
    # False(기본)로 두면 학습·BC 시연·KPI 평가가 모두 추론과 같은 조건이 된다.
    train_truncate_on_time: bool = False

    # --- 데이터셋 샘플링 (도메인 랜덤화) ---------------------------------------
    # "per_env": 데이터셋 1개당 env 1개 (기존 동작). 데이터셋이 많으면 롤아웃
    #     크기가 데이터셋 수에 비례해 커져 PPO 업데이트 횟수가 줄고, 정책이
    #     각 시나리오의 정답을 외우기 쉽다.
    # "random_reset": env를 `dataset_pool_envs`개만 만들고 에피소드마다 풀에서
    #     시나리오를 하나 뽑는다. 데이터셋이 아무리 많아도 롤아웃 크기가
    #     일정하고, 같은 시나리오를 연속으로 보지 않아 일반화가 잘 된다.
    #     (실측: per_env로 BENCH_SUITE 8종만 학습하면 그 8종에선 dedication을
    #      상회하지만 학습에 안 쓴 HOLDOUT 6종에선 오히려 뒤졌다.)
    dataset_sampling: str = "random_reset"
    dataset_pool_envs: int = 4          # random_reset일 때 만들 env 수
    # 데이터셋이 이 수 이하이면 random_reset이어도 per_env로 둔다
    # (소수 데이터셋에서는 env를 나눠 쓰는 편이 롤아웃 다양성에 유리).
    dataset_sampling_min_pool: int = 2


@dataclass
class RewardConfig:
    # --- Step A: 스케일 재정규화 (모든 항을 ±5 band로, step reward clip) ---
    # 제품·공정이 '모두' 직전과 동일할 때만 주는 연속 보너스 (전환 회피와 정렬).
    w_same_setup:      float = 1.0
    w_idle_per_min:    float = 0.0       # [제거] idle 분당 (1·2·6·7만 유지)
    w_plan_hit:        float = 0.0       # [제거] 달성 진척 (cover 무시 → 전담 방해 1위라 제거)
    w_pacing:          float = 0.0       # [제거] 선형 takt 추종 (achievable 기준; Step C)
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
    # SYM_5x5(대칭 5x5, 전담 시 전환 0회가 최적) 학습 실험에서 이 세 항을 꺼둔
    # 기본값(0.0)으로는 200k 스텝을 학습해도 전환이 15회 안팎에서 정체됐다.
    # 아래 값(bench_suite.py의 co-train 전용 오버라이드였던 값과 동일)으로 켜면
    # 같은 스텝 수에서 전환이 2회까지 줄어드는 걸 확인해 기본값으로 승격했다.
    # ① 블록 크기 보너스: 같은 제품군을 큰 블록으로 커밋할수록 보상(전담 커밋 유도).
    w_bulk_block_bonus:   float = 3.0
    # ② 전용 오용 페널티(<0): 범용 장비가 더 전용적인 idle 장비도 가능한 버킷을
    #    잡을 때 감점 → 전용 장비가 놀지 않게.
    w_dedication_misuse:  float = -4.0
    # ③ 중복 커버 페널티(<0): 이미 다른 셋업 장비가 horizon 내 충분히 덮는 버킷을
    #    잡으면 감점 → 다른 제품으로 전환할 '용기'.
    w_redundant_cover:    float = -5.0
    # --- Step B: flow-balance shaping (WIP 비중 vs 계획 비중 기준) ---
    w_flow_balance:    float = 0.0       # [제거] cover 무시 → 전담 방해 2위라 제거
    # 후속 ready WIP / 후속 장비 합산 분당 처리량(매/분) ≤ 이 값(분)일 때만 feeding 보너스
    flow_balance_starving_cover_min: float = 120.0
    # --- Step A: step reward clip 범위 (PPO advantage 안정화) ---
    # w_conversion(-10.0) 하나만으로 이미 옛 clip(10.0)을 다 써버려서, 그 위에
    # 붙는 w_avoidable_conversion(-8.0)이 항상 clip에 눌려 아무 효과가 없었다
    # (회피 가능한 전환과 불가피한 전환이 똑같이 -10.0으로만 보임). |w_conversion|
    # + |w_avoidable_conversion| = 18.0 을 clip 안에 넣어 그 차이가 실제로
    # 보상에 반영되게 한다.
    reward_clip:       float = 20.0
    # --- Step C: achievable target 사용 여부 (재공 한계까지만 계획 추종) ---
    use_achievable_target: bool = True

    # --- 종단(terminal) KPI 정렬 보상 (SchedulingRLEnv 전용) -------------------
    # 지금까지의 step reward는 전부 대리지표(전환 패널티, 같은 셋업 보너스,
    # 블록 크기 보너스…)라서 "보상이 가장 높은 정책"과 "벤치마크 KPI가 가장
    # 좋은 정책"이 일치한다는 보장이 없었다. 벤치마크가 실제로 재는 값
    # (sim_end 안에 끝난 carrier 수, 전환 총횟수)을 에피소드 종료 시 직접
    # 보상으로 준다 → 최적화 대상과 평가 대상을 일치시킨다.
    #
    # 이 항은 step reward clip 적용 이후에 더해진다(clip에 눌리면 의미 없음).
    # 신호가 초반 결정까지 역전파되려면 RLConfig.gamma가 충분히 커야 한다
    # (기본 0.997).
    w_terminal_throughput: float = 30.0   # × (완료 carrier / 생산가능 carrier)
    w_terminal_conversion: float = -1.0   # × 전환 총횟수
    # 종단 보상 전체를 이 범위로 clip (한 스텝에 과도한 보상이 들어가 PPO
    # advantage 분포를 망가뜨리는 걸 방지). 0이면 clip 없음.
    terminal_reward_clip: float = 60.0


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
        "w_terminal_throughput": r.w_terminal_throughput,
        "w_terminal_conversion": r.w_terminal_conversion,
        "terminal_reward_clip": r.terminal_reward_clip,
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
        "w_terminal_throughput", "w_terminal_conversion", "terminal_reward_clip",
    )
    for key in float_keys:
        if key in params and params[key] is not None:
            setattr(r, key, float(params[key]))
    if "use_achievable_target" in params and params["use_achievable_target"] is not None:
        r.use_achievable_target = bool(params["use_achievable_target"])


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class SchedulerConfig:
    """API 서버(main.py ui / uvicorn) 기동 시 주기적으로 infer 를 자동 실행하는 옵션.

    기본은 비활성(enabled=False) — 명시적으로 켜야 동작한다. fac_id 없이
    enabled=True 만 켜면 시작하지 않고 경고만 남긴다(운영 DB에 의도치 않게
    자동 적재되는 사고를 막기 위한 opt-in 설계).
    """
    enabled:          bool  = field(default_factory=lambda: _env_bool("SCHEDULER_ENABLED"))
    interval_seconds: int   = field(default_factory=lambda: int(os.environ.get("SCHEDULER_INTERVAL_SECONDS", "3600") or 3600))
    fac_id:           str   = field(default_factory=lambda: os.environ.get("SCHEDULER_FAC_ID", "").strip())
    algorithm:        str   = field(default_factory=lambda: os.environ.get("SCHEDULER_ALGORITHM", "scheduling_rl").strip())
    db_alias:         Optional[str] = field(default_factory=lambda: os.environ.get("SCHEDULER_DB_ALIAS", "").strip() or None)
    no_history:       bool  = field(default_factory=lambda: _env_bool("SCHEDULER_NO_HISTORY"))


@dataclass
class Config:
    path:      PathConfig      = field(default_factory=PathConfig)
    oracle:    OracleConfig    = field(default_factory=OracleConfig)
    env:       EnvConfig       = field(default_factory=EnvConfig)
    rl:        RLConfig        = field(default_factory=RLConfig)
    reward:    RewardConfig    = field(default_factory=RewardConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


CONFIG = Config()

_env_input = os.environ.get("SCHEDULING_INPUT", "").strip()
if _env_input:
    set_input_folder(_env_input)
