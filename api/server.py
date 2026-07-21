"""
api/server.py – FastAPI 백엔드
React UI가 호출하는 REST API를 제공합니다.

실행: uvicorn api.server:app --reload --port 8001
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder, list_input_folders, PERIOD_SPLITS, validate_path_segment, parse_input_folder, latest_period, folders_in_period_range, format_missing_input_file_error, reward_params_dict, apply_reward_params, resolve_infer_rule_timekey, resolve_train_folders, normalize_rule_timekey
from data.loader import load_data, validate_data, fetch_from_db, preprocess
from data.loader.sql_binds import resolve_lot_cd
from data.writer.db_load import load_output_sql_files
from agent.rl_agent import SchedulingAgent
from agent.registry import ALGORITHMS, validate_algorithm
from inference.runner import run_inference, run_inference_compare, save_result
from validation.output_checks import validate_schedule_output
from api.serializers import (
    env_data_summary, empty_data_summary, serialize_inference_result, serialize_compare_response,
)
from api.test_benchmark_store import (
    load_benchmark,
    save_benchmark,
    clear_benchmark,
    init_benchmark,
    append_dataset,
    empty_state,
)
from api.train_service import train_progress, start_training, stop_training, is_training
from benchmark.optimal.runner import run_optimal_benchmark
from benchmark.tool_change_bench import run_detailed_benchmark

app = FastAPI(title="Scheduling RL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 캐시 – 마지막 추론 결과 (히스토리 포함)
_last_inference: Optional[dict] = None
_env_data_cache: Optional[dict] = None
_benchmark_rl_agent: Optional[SchedulingAgent] = None
_data_warnings: list[str] = []


def _is_hard_error(msg: str) -> bool:
    """구조적 결함(빈 데이터, 파일 없음, OPER_ID/SEQ 누락)은 hard. 그 외 필드 누락은 soft warning."""
    if "비어 있습니다" in msg or "FileNotFound" in msg or "OPER_ID 또는 SEQ" in msg:
        return True
    if "필드 누락" in msg and "OPER_ID" in msg:
        return True
    return False


def _split_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    hard = [e for e in errors if _is_hard_error(e)]
    soft = [e for e in errors if not _is_hard_error(e)]
    return hard, soft


def _require_input_files(input_dir: Path) -> None:
    """dataset input 폴더에 필수 JSON이 있는지 확인."""
    required = CONFIG.path.discrete_arrange_file
    path = input_dir / required
    if not path.is_file():
        raise FileNotFoundError(format_missing_input_file_error(input_dir, required))


def _load_env_data(period_key: Optional[str] = None) -> dict:
    global _env_data_cache, _data_warnings
    if _env_data_cache is not None:
        return _env_data_cache
    _require_input_files(CONFIG.path.input_dir)
    try:
        raw = load_data()
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "errors": [f"JSON 파싱 오류: {exc}"],
                "hint": (
                    f"input 폴더의 JSON 파일 형식이 올바르지 않습니다. "
                    f"확인: {CONFIG.path.input_dir}"
                ),
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "errors": [f"입력 파일 읽기 실패: {exc}"],
                "hint": f"input 폴더 권한·경로를 확인하세요: {CONFIG.path.input_dir}",
            },
        ) from exc
    errors = validate_data(raw)
    hard, soft = _split_errors(errors)
    if hard:
        raise HTTPException(status_code=400, detail={"errors": hard})
    _data_warnings = soft  # 소프트 경고는 저장하고 계속 진행
    try:
        _env_data_cache = preprocess(raw, period_key=period_key)
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "errors": [str(exc)],
                "hint": "입력 JSON 필드가 전처리 규칙과 맞지 않습니다. dataset input 폴더를 확인하세요.",
            },
        ) from exc
    return _env_data_cache


def _apply_input_folder(folder: Optional[str]) -> None:
    global _env_data_cache, _data_warnings
    if folder:
        set_input_folder(folder)
        _env_data_cache = None
        _data_warnings = []


def _resolve_infer_fac_id(fac_id: Optional[str], input_folder: Optional[str]) -> str:
    if fac_id:
        return validate_path_segment(fac_id, "FAC_ID")
    if input_folder:
        parsed, _, _ = parse_input_folder(input_folder)
        return parsed
    if CONFIG.path.fac_id:
        return CONFIG.path.fac_id
    return "FAC001"


def _prepare_infer_input(
    *,
    fac_id: Optional[str] = None,
    input_folder: Optional[str] = None,
    rule_timekey: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    prevcnt: Optional[int] = None,
    lot_cd: Optional[str] = None,
) -> dict:
    """CLI cmd_inference 와 동일한 input 준비 (Oracle fetch → infer 폴더 설정)."""
    global _env_data_cache

    resolved_fac = _resolve_infer_fac_id(fac_id, input_folder)
    if rule_timekey and (prevcnt is not None or from_date or to_date):
        raise ValueError("rule_timekey는 prevcnt, from_date/to_date와 함께 쓸 수 없습니다.")
    if prevcnt is not None and (from_date or to_date):
        raise ValueError("prevcnt와 from_date/to_date를 함께 쓸 수 없습니다.")
    rtk = resolve_infer_rule_timekey(
        resolved_fac, rule_timekey,
        from_key=from_date, to_key=to_date, prevcnt=prevcnt,
    )
    lcd = resolve_lot_cd(lot_cd)

    fetch_from_db(fac_id=resolved_fac, split="infer", period=rtk, lot_cd=lcd)
    _env_data_cache = None

    infer_folder = f"{resolved_fac}/infer"
    set_input_folder(infer_folder)
    _env_data_cache = None

    print(f"[inference] FAC={resolved_fac}  RULE_TIMEKEY={rtk}")

    return {
        "fac_id": resolved_fac,
        "rule_timekey": rtk,
        "lot_cd": lcd,
        "input_folder": infer_folder,
        "fetched_from_db": True,
    }


def _load_env_data_for_folder(folder: str) -> dict:
    """지정 train 스냅샷 1개 로드 (전역 입력 경로는 복원)."""
    original = CONFIG.path.input_folder_key
    try:
        set_input_folder(folder)
        _require_input_files(CONFIG.path.input_dir)
        raw = load_data()
        errors = validate_data(raw)
        hard, _ = _split_errors(errors)
        if hard:
            raise HTTPException(
                status_code=400,
                detail={"errors": hard, "folder": folder},
            )
        try:
            return preprocess(raw)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"errors": [str(exc)], "folder": folder},
            ) from exc
    finally:
        set_input_folder(original)


def _normalize_input_folder_key(folder: str) -> str:
    """dataset 경로 키 검증 (FAC001/train/YYYYMMDDHHmmss 형식)."""
    try:
        fac_id, split, period = parse_input_folder(folder.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if split in PERIOD_SPLITS:
        per = period or latest_period(fac_id, split)
        if not per:
            raise HTTPException(status_code=404, detail=f"기간 폴더 없음: {folder.strip()}")
        return f"{fac_id}/{split}/{per}"
    return f"{fac_id}/{split}"


def _resolve_train_folders(req: "TrainRequest") -> list[str]:
    """학습에 사용할 train 스냅샷 경로 목록."""
    fac_id = req.fac_id or CONFIG.path.fac_id
    available = set(list_input_folders())

    if req.input_folders:
        folders = [
            _normalize_input_folder_key(f)
            for f in req.input_folders
            if f and f.strip()
        ]
        if not folders:
            raise HTTPException(status_code=400, detail="input_folders가 비어 있습니다.")
        missing = [f for f in folders if f not in available]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"데이터셋 없음: {', '.join(missing)}",
            )
        return folders

    if req.from_date and req.to_date:
        if req.prevcnt is not None:
            raise HTTPException(
                status_code=400,
                detail="prevcnt와 from_date/to_date를 함께 쓸 수 없습니다.",
            )
        start_key = normalize_rule_timekey(req.from_date)
        end_key = normalize_rule_timekey(req.to_date)
        folders = resolve_train_folders(fac_id, start_key, end_key)
        if not folders:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"기간 {req.from_date}~{req.to_date}에 해당하는 "
                    f"train JSON 데이터가 없습니다."
                ),
            )
        return folders

    if req.prevcnt is not None:
        # 오늘 기준 날짜 구간이 아니라, 이미 존재하는 train 폴더 중 최근 N개
        prefix = f"{fac_id}/train/"
        folders = sorted(f for f in list_input_folders() if f.startswith(prefix))
        folders = folders[-req.prevcnt:] if req.prevcnt else folders
        if not folders:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"최근 {req.prevcnt}개에 해당하는 "
                    f"train JSON 데이터가 없습니다."
                ),
            )
        return folders

    if req.input_folder:
        folder = _normalize_input_folder_key(req.input_folder)
        if folder not in available:
            raise HTTPException(status_code=404, detail=f"데이터셋 없음: {folder}")
        return [folder]

    return [CONFIG.path.input_folder_key]


def _prepare_train_env_data(req: "TrainRequest") -> tuple[list[dict], list[str]]:
    folders = _resolve_train_folders(req)
    return [_load_env_data_for_folder(f) for f in folders], folders


# ── 요청 스키마 ───────────────────────────────────────────────────────────────

class RewardParams(BaseModel):
    w_same_setup: float = Field(default=CONFIG.reward.w_same_setup)
    w_idle_per_min: float = Field(default=CONFIG.reward.w_idle_per_min)
    w_plan_hit: float = Field(default=CONFIG.reward.w_plan_hit)
    w_pacing: float = Field(default=CONFIG.reward.w_pacing)
    pacing_coverage_scale: float = Field(default=CONFIG.reward.pacing_coverage_scale)
    w_conversion: float = Field(default=CONFIG.reward.w_conversion)
    w_avoidable_conversion: float = Field(default=CONFIG.reward.w_avoidable_conversion)
    conversion_amortize_factor: float = Field(default=CONFIG.reward.conversion_amortize_factor)
    # 벌크 점유(Bulk-Fill) 전용 보상항
    w_bulk_block_bonus: float = Field(default=CONFIG.reward.w_bulk_block_bonus)
    w_dedication_misuse: float = Field(default=CONFIG.reward.w_dedication_misuse)
    w_redundant_cover: float = Field(default=CONFIG.reward.w_redundant_cover)
    w_flow_balance: float = Field(default=CONFIG.reward.w_flow_balance)
    flow_balance_starving_cover_min: float = Field(default=CONFIG.reward.flow_balance_starving_cover_min)
    reward_clip: float = Field(default=CONFIG.reward.reward_clip, ge=0.1)
    use_achievable_target: bool = Field(default=CONFIG.reward.use_achievable_target)


class TrainRequest(RewardParams):
    algorithm: str = Field(
        default="scheduling_rl",
        description="학습 환경 유형: scheduling_rl (SchedulingRLEnv)",
    )
    total_timesteps: int = Field(default=CONFIG.rl.total_timesteps, ge=1000)
    learning_rate: float = Field(default=CONFIG.rl.learning_rate, gt=0)
    train_budget_mode: Literal["timesteps", "episodes"] = Field(
        default="timesteps",
        description="학습량 기준: timesteps | episodes",
    )
    n_episodes: Optional[int] = Field(
        default=None,
        ge=1,
        le=100_000,
        description="train_budget_mode=episodes 일 때 목표 에피소드 수",
    )
    input_folder: Optional[str] = Field(
        default=None,
        description="단일 train 스냅샷 (미지정 시 사이드바 현재 선택)",
    )
    input_folders: Optional[list[str]] = Field(
        default=None,
        description="복수 train 스냅샷 – VecEnv로 병렬 학습",
    )
    from_date: Optional[str] = Field(
        default=None,
        description="train 기간 시작 RULE_TIMEKEY (YYYYMMDDHHmmss)",
    )
    to_date: Optional[str] = Field(
        default=None,
        description="train 기간 종료 RULE_TIMEKEY (YYYYMMDDHHmmss)",
    )
    prevcnt: Optional[int] = Field(
        default=None,
        ge=1,
        description="현재 기준 최근 N개 RULE_TIMEKEY의 train 폴더 사용 (from_date/to_date 와 함께 쓸 수 없음)",
    )
    fac_id: Optional[str] = Field(
        default=None,
        description="from/to·prevcnt 기간 검색용 FAC_ID (기본: 현재 설정)",
    )


class InferFetchOptions(BaseModel):
    fac_id: Optional[str] = Field(
        default=None,
        description="공장 ID (미지정 시 input_folder 또는 현재 설정)",
    )
    rule_timekey: Optional[str] = Field(
        default=None,
        description="추론 RULE_TIMEKEY (미지정 시 최신)",
    )
    from_date: Optional[str] = Field(
        default=None,
        description="구간 시작 RULE_TIMEKEY (BETWEEN 조회 후 최신값 사용, rule_timekey와 함께 쓸 수 없음)",
    )
    to_date: Optional[str] = Field(
        default=None,
        description="구간 종료 RULE_TIMEKEY (BETWEEN 조회 후 최신값 사용, rule_timekey와 함께 쓸 수 없음)",
    )
    prevcnt: Optional[int] = Field(
        default=None,
        ge=1,
        description="최신 기준 최근 N개 RULE_TIMEKEY 조회 후 최신값 사용 (rule_timekey와 함께 쓸 수 없음)",
    )
    lot_cd: Optional[str] = Field(
        default=None,
        description="SQL :LOT_CD 바인드 (discrete_arrange 제외)",
    )
    db_alias: Optional[str] = Field(
        default=None,
        description="추론 후 Oracle RTS 테이블 적재 시 사용할 DB alias (미지정 시 databases.yaml default)",
    )
    no_history: bool = Field(
        default=False,
        description="추론 후 DB 적재 시 HIS 테이블 적재 생략",
    )
    max_conversions: Optional[int] = Field(
        default=None,
        ge=0,
        description="시뮬 전체 전환(컨버전) 상한",
    )
    max_conversions_per_eqp: Optional[int] = Field(
        default=None,
        ge=0,
        description="EQP별 전환(컨버전) 상한",
    )
    conversion_minutes: Optional[int] = Field(
        default=None,
        ge=0,
        description="LOT_CD/TEMP 전환 1회 소요 시간(분)",
    )
    discrete_wait_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "WAIT LOT의 전환 불필요 배정에 discrete(EQP×carrier 정밀 조합) 자격 검증을 "
            "요구할지 여부 (기본 True). False면 discrete 조합이 없어도 abstract로 배정 가능."
        ),
    )


class InferenceRequest(InferFetchOptions):
    algorithm: str = Field(default="scheduling_rl", description="scheduling_rl | minprogress | earliest_st")
    input_folder: Optional[str] = Field(
        default=None,
        description="FAC_ID 추론용 (미지정 시 현재 선택). fetch 후에는 {FAC_ID}/infer 로 고정",
    )
    decision_log: bool = Field(
        default=False,
        description="step별 EQP/PPK/OPER 결정 및 미할당 사유 로그 포함",
    )
    enable_wip_inflow: bool = Field(
        default=False,
        description="공정 완료 시 다음 공정 flow 재공 유입 이벤트 사용",
    )
    include_history: bool = Field(
        default=False,
        description="시뮬레이션 재생용 history/event payload 포함",
    )


class InputFolderRequest(BaseModel):
    input_folder: str = Field(description="사용할 입력 폴더명")


class CompareRequest(InferFetchOptions):
    algorithms: list[str] = Field(
        min_length=1,
        description="비교할 알고리즘 ID 목록",
    )
    input_folder: Optional[str] = Field(
        default=None,
        description="FAC_ID 추론용 (미지정 시 현재 선택). fetch 후에는 {FAC_ID}/infer 로 고정",
    )
    decision_log: bool = Field(
        default=False,
        description="step별 EQP/PPK/OPER 결정 및 미할당 사유 로그 포함",
    )
    enable_wip_inflow: bool = Field(
        default=False,
        description="공정 완료 시 다음 공정 flow 재공 유입 이벤트 사용",
    )
    include_history: bool = Field(
        default=False,
        description="비교 응답에 history/event payload 포함",
    )


class TestPeriodFilter(BaseModel):
    from_date: Optional[str] = Field(default=None, description="검증 시작 RULE_TIMEKEY")
    to_date: Optional[str] = Field(default=None, description="검증 종료 RULE_TIMEKEY")
    prevcnt: Optional[int] = Field(
        default=None, ge=1,
        description="최신 기준 최근 N개 test 폴더만 사용 (from_date/to_date 와 함께 쓸 수 없음)",
    )


class TestBenchmarkRequest(TestPeriodFilter):
    algorithms: list[str] = Field(min_length=1, description="비교할 알고리즘 ID 목록")
    fac_id: Optional[str] = Field(default=None, description="FAC_ID (기본: 현재 설정)")
    input_folders: Optional[list[str]] = Field(
        default=None,
        description="test 데이터셋 경로 목록 (미지정 시 fac_id 하위 test 전체)",
    )


class TestBenchmarkInitRequest(TestPeriodFilter):
    algorithms: list[str] = Field(min_length=1)
    fac_id: Optional[str] = None


class TestBenchmarkRunOneRequest(BaseModel):
    algorithms: list[str] = Field(min_length=1)
    input_folder: str = Field(min_length=1)
    fac_id: Optional[str] = None
    progress_current: int = Field(ge=0)
    progress_total: int = Field(ge=1)
    done: bool = False


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """헬스 체크: 시스템 상태 진단"""
    from datetime import datetime, timezone

    status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {}
    }

    # 1. API 서버 상태
    status["components"]["api"] = {"status": "healthy"}

    # 2. 데이터베이스 연결 상태
    try:
        from data.db_registry import test_db_connection
        test_db_connection()
        status["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        status["status"] = "degraded"

    # 3. 모델 파일 상태
    try:
        agent = SchedulingAgent()
        model_exists = agent.model_exists()
        status["components"]["model"] = {
            "status": "healthy" if model_exists else "not_found",
            "exists": model_exists
        }
    except Exception as e:
        status["components"]["model"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # 4. 입력 데이터 폴더 접근성
    try:
        input_dir = CONFIG.path.input_dir
        if input_dir.exists() and input_dir.is_dir():
            status["components"]["input_folder"] = {
                "status": "healthy",
                "path": str(input_dir)
            }
        else:
            status["components"]["input_folder"] = {
                "status": "unhealthy",
                "path": str(input_dir),
                "error": "폴더를 찾을 수 없거나 접근할 수 없음"
            }
            status["status"] = "degraded"
    except Exception as e:
        status["components"]["input_folder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        status["status"] = "degraded"

    # 5. 출력 폴더 접근성
    try:
        output_dir = CONFIG.path.output_dir
        if output_dir.exists() and output_dir.is_dir():
            status["components"]["output_folder"] = {
                "status": "healthy",
                "path": str(output_dir)
            }
        else:
            status["components"]["output_folder"] = {
                "status": "unhealthy",
                "path": str(output_dir),
                "error": "폴더를 찾을 수 없거나 접근할 수 없음"
            }
            status["status"] = "degraded"
    except Exception as e:
        status["components"]["output_folder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        status["status"] = "degraded"

    # 6. 현재 입력 폴더 상태
    try:
        status["components"]["current_input_folder"] = {
            "status": "healthy",
            "input_folder": CONFIG.path.input_folder_key,
            "fac_id": CONFIG.path.fac_id
        }
    except Exception as e:
        status["components"]["current_input_folder"] = {
            "status": "unknown",
            "error": str(e)
        }

    return status


@app.get("/api/health/detailed")
def health_detailed():
    """상세 헬스 체크: 모든 구성 요소의 상세 정보"""
    from datetime import datetime, timezone
    import psutil

    status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
        "system": {}
    }

    # 1. API 서버
    status["components"]["api"] = {"status": "healthy"}

    # 2. 데이터베이스
    try:
        from data.db_registry import test_db_connection
        test_db_connection()
        status["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)[:200]
        }
        status["status"] = "degraded"

    # 3. 모델
    try:
        agent = SchedulingAgent()
        model_exists = agent.model_exists()
        status["components"]["model"] = {
            "status": "healthy" if model_exists else "not_found",
            "exists": model_exists,
            "model_dir": str(CONFIG.path.model_dir)
        }
    except Exception as e:
        status["components"]["model"] = {
            "status": "unhealthy",
            "error": str(e)[:200]
        }

    # 4. 입력/출력 폴더
    try:
        input_dir = CONFIG.path.input_dir
        status["components"]["input_folder"] = {
            "status": "healthy" if (input_dir.exists() and input_dir.is_dir()) else "unhealthy",
            "path": str(input_dir),
            "exists": input_dir.exists()
        }
    except Exception as e:
        status["components"]["input_folder"] = {
            "status": "unhealthy",
            "error": str(e)[:200]
        }

    try:
        output_dir = CONFIG.path.output_dir
        status["components"]["output_folder"] = {
            "status": "healthy" if (output_dir.exists() and output_dir.is_dir()) else "unhealthy",
            "path": str(output_dir),
            "exists": output_dir.exists()
        }
    except Exception as e:
        status["components"]["output_folder"] = {
            "status": "unhealthy",
            "error": str(e)[:200]
        }

    # 5. 시스템 정보
    try:
        status["system"]["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        status["system"]["memory_percent"] = psutil.virtual_memory().percent
        status["system"]["disk_percent"] = psutil.disk_usage("/").percent
    except Exception:
        pass

    # 6. 입력 폴더 설정
    try:
        status["components"]["current_input_folder"] = {
            "status": "healthy",
            "input_folder": CONFIG.path.input_folder_key,
            "fac_id": CONFIG.path.fac_id
        }
    except Exception as e:
        status["components"]["current_input_folder"] = {
            "status": "unknown",
            "error": str(e)[:200]
        }

    return status


@app.get("/api/config")
def get_config():
    return {
        "model_dir": str(CONFIG.path.model_dir),
        "input_folder": CONFIG.path.input_folder_key,
        "fac_id": CONFIG.path.fac_id,
        "dataset_split": CONFIG.path.dataset_split,
        "train_snapshot": CONFIG.path.train_snapshot,
        "sql_dir": str(CONFIG.path.sql_dir),
        "input_dir": str(CONFIG.path.input_dir),
        "output_dir": str(CONFIG.path.output_dir),
        "infer_input_dir": str(CONFIG.path.infer_input_dir),
        "infer_output_dir": str(CONFIG.path.infer_output_dir),
        "input_folders": list_input_folders(),
        "default_timesteps": CONFIG.rl.total_timesteps,
        "default_n_episodes": CONFIG.rl.default_n_episodes,
        "default_learning_rate": CONFIG.rl.learning_rate,
        "default_reward": reward_params_dict(),
        "default_env": {
            "conversion_minutes": CONFIG.env.conversion_minutes,
            "max_conversions": CONFIG.env.max_conversions,
            "max_conversions_per_eqp": CONFIG.env.max_conversions_per_eqp,
            "discrete_wait_enabled": CONFIG.env.discrete_wait_enabled,
        },
    }


@app.get("/api/data/summary")
def data_summary():
    try:
        env_data = _load_env_data()
    except FileNotFoundError:
        return empty_data_summary()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "errors": [str(e)],
                "hint": f"데이터 요약 실패. input 폴더를 확인하세요: {CONFIG.path.input_dir}",
            },
        ) from e
    result = env_data_summary(env_data)
    result["warnings"] = _data_warnings  # 소프트 경고 포함
    return result


@app.post("/api/config/input")
def select_input_folder(req: InputFolderRequest):
    try:
        path = set_input_folder(req.input_folder)
        _require_input_files(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    global _env_data_cache
    _env_data_cache = None
    return {
        "message": f"입력 폴더가 '{req.input_folder}'(으)로 설정되었습니다.",
        "input_folder": CONFIG.path.input_folder_key,
        "input_dir": str(path),
        "output_dir": str(CONFIG.path.output_dir),
    }


@app.get("/api/model/status")
def model_status():
    agent = SchedulingAgent()
    return {"exists": agent.model_exists()}


@app.post("/api/train/start")
def train_start(req: TrainRequest):
    try:
        env_list, folders = _prepare_train_env_data(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if is_training():
        raise HTTPException(status_code=409, detail="이미 학습이 진행 중입니다.")
    params = req.model_dump()
    params["input_folders"] = folders
    payload = env_list if len(env_list) > 1 else env_list[0]
    if not start_training(payload, params):
        raise HTTPException(status_code=409, detail="학습을 시작할 수 없습니다.")
    return {"message": "학습 시작", "input_folders": folders}


@app.get("/api/train/status")
def train_status():
    return train_progress.snapshot()


@app.post("/api/train/stop")
def train_stop():
    if not stop_training():
        raise HTTPException(status_code=409, detail="진행 중인 학습이 없습니다.")
    return {"message": "학습 중지 요청"}


@app.post("/api/train")
def train(req: TrainRequest):
    try:
        env_list, folders = _prepare_train_env_data(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    CONFIG.rl.total_timesteps = req.total_timesteps
    CONFIG.rl.learning_rate = req.learning_rate
    apply_reward_params(req.model_dump())

    from env.scheduling_rl_env import SchedulingRLEnv
    env_cls = SchedulingRLEnv

    agent = SchedulingAgent()
    payload = env_list if len(env_list) > 1 else env_list[0]
    train_kwargs: dict = {"verbose": 0, "env_cls": env_cls}
    if req.train_budget_mode == "episodes" and req.n_episodes:
        train_kwargs["n_episodes"] = req.n_episodes
    agent.train(payload, **train_kwargs)
    agent.save()
    eval_eps = req.n_episodes if req.train_budget_mode == "episodes" and req.n_episodes else 3
    metrics = agent.evaluate(env_list[0], n_episodes=eval_eps)
    return {"message": "학습 완료", "metrics": metrics, "input_folders": folders}


@app.get("/api/algorithms")
def list_algorithms():
    return {"algorithms": ALGORITHMS}


@app.post("/api/inference")
def inference(req: InferenceRequest):
    global _last_inference
    try:
        infer_meta = _prepare_infer_input(
            fac_id=req.fac_id,
            input_folder=req.input_folder,
            rule_timekey=req.rule_timekey,
            from_date=req.from_date,
            to_date=req.to_date,
            prevcnt=req.prevcnt,
            lot_cd=req.lot_cd,
        )
    except (ValueError, FileNotFoundError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {e}")

    original_discrete_wait_enabled = CONFIG.env.discrete_wait_enabled
    if req.discrete_wait_enabled is not None:
        CONFIG.env.discrete_wait_enabled = req.discrete_wait_enabled
    try:
        try:
            env_data = _load_env_data(period_key=infer_meta["rule_timekey"])
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        try:
            validate_algorithm(req.algorithm)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        agent = None
        if req.algorithm == "scheduling_rl":
            try:
                agent = SchedulingAgent.load(env_data=env_data)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

        result = run_inference(
            env_data,
            algorithm=req.algorithm,
            agent=agent,
            record_history=req.include_history,
            record_decision_log=req.decision_log,
            enable_wip_inflow=req.enable_wip_inflow,
            max_conversions=req.max_conversions,
            max_conversions_per_eqp=req.max_conversions_per_eqp,
            conversion_minutes=req.conversion_minutes,
        )
        result["prod_keys"] = env_data["prod_keys"]
        result["oper_ids"] = env_data["oper_ids"]
        result["eqp_ids"] = env_data["eqp_ids"]
        result["sim_end_minutes"] = env_data["sim_end_minutes"]
        result["validation"] = validate_schedule_output(result, env_data)
        save_result(
            result, output_dir=CONFIG.path.output_dir, env_data=env_data,
            fac_id=infer_meta["fac_id"], rule_timekey=infer_meta["rule_timekey"],
        )
    finally:
        CONFIG.env.discrete_wait_enabled = original_discrete_wait_enabled
    try:
        load_output_sql_files(
            CONFIG.path.output_dir,
            db_alias=req.db_alias,
            include_history=not req.no_history,
        )
        infer_meta["db_loaded"] = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 적재 실패: {e}") from e
    _last_inference = result
    payload = serialize_inference_result(
        result,
        include_history=req.include_history,
        include_event_log=req.include_history,
        include_decision_log=req.decision_log,
    )
    payload["infer_meta"] = infer_meta
    return payload


@app.post("/api/inference/compare")
def inference_compare(req: CompareRequest):
    try:
        infer_meta = _prepare_infer_input(
            fac_id=req.fac_id,
            input_folder=req.input_folder,
            rule_timekey=req.rule_timekey,
            from_date=req.from_date,
            to_date=req.to_date,
            prevcnt=req.prevcnt,
            lot_cd=req.lot_cd,
        )
    except (ValueError, FileNotFoundError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {e}")

    original_discrete_wait_enabled = CONFIG.env.discrete_wait_enabled
    if req.discrete_wait_enabled is not None:
        CONFIG.env.discrete_wait_enabled = req.discrete_wait_enabled
    try:
        try:
            env_data = _load_env_data(period_key=infer_meta["rule_timekey"])
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        for algo in req.algorithms:
            try:
                validate_algorithm(algo)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        payload = run_inference_compare(
            env_data,
            req.algorithms,
            record_history=req.include_history,
            record_decision_log=req.decision_log,
            enable_wip_inflow=req.enable_wip_inflow,
            max_conversions=req.max_conversions,
            max_conversions_per_eqp=req.max_conversions_per_eqp,
            conversion_minutes=req.conversion_minutes,
        )
    finally:
        CONFIG.env.discrete_wait_enabled = original_discrete_wait_enabled
    if not payload["results"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "실행 가능한 알고리즘이 없습니다.",
                "errors": payload["errors"],
            },
        )
    response = serialize_compare_response(payload, include_history=req.include_history)
    response["infer_meta"] = infer_meta
    return response


def _minutes_from_timekey(value: str, base: datetime) -> int:
    try:
        fmt = "%Y%m%d%H%M%S" if len(value) == 14 else "%Y%m%d%H%M"
        return int((datetime.strptime(value, fmt) - base).total_seconds() // 60)
    except Exception:
        return 0


def _result_from_rts_output(payload: dict, env_data: dict) -> dict:
    """RTS output.json만 있을 때 UI 간트용 result 구조로 복원."""
    base = env_data["sim_base_time"]
    schedule = []
    for row in payload.get("RTS_RSLT_MAS", []):
        start_tm = _minutes_from_timekey(str(row.get("START_TIME", "")), base)
        end_tm = _minutes_from_timekey(str(row.get("END_TIME", "")), base)
        schedule.append({
            "EQP_ID":        row.get("EQP_ID", ""),
            "LOT_ID":        row.get("LOT_ID", ""),
            "CARRIER_ID":    row.get("CARRIER_ID", ""),
            "PLAN_PROD_ATTR_VAL": row.get("PLAN_PROD_ATTR_VAL", ""),
            "OPER_ID":       row.get("OPER_ID", ""),
            "EQP_MODEL":     row.get("EQP_MODEL_CD", ""),
            "SEQ":           int(row.get("SEQ_NO") or 0),
            "START_TM":      start_tm,
            "END_TM":        end_tm,
            "PROC_TIME":     max(end_tm - start_tm, 0),
            "WF_QTY":        int(row.get("PRODUCE_QTY") or 0),
            "LOT_CD":        row.get("LOT_CD", ""),
            "TEMP":          row.get("TEMPER_VAL", ""),
            "START_TM_STR":  row.get("START_TIME", ""),
            "END_TM_STR":    row.get("END_TIME", ""),
        })
    schedule.sort(key=lambda r: (r["START_TM"], r["EQP_ID"], r["SEQ"], r["LOT_ID"]))

    conversion_plans = []
    for row in payload.get("RTS_EQPCONVPLAN_INF", []):
        conv_start = _minutes_from_timekey(str(row.get("CONV_START_TM", "")), base)
        conv_end = _minutes_from_timekey(str(row.get("CONV_END_TM", "")), base)
        conversion_plans.append({
            "eqp_id":         row.get("EQP_ID", ""),
            "eqp_model_cd":   row.get("EQP_MODEL_CD", ""),
            "oper_id":        row.get("OPER_ID", ""),
            "PLAN_PROD_ATTR_VAL":  row.get("PLAN_PROD_ATTR_VAL", ""),
            "from_lot_cd":    row.get("LOT_CD", ""),
            "from_temp":      row.get("TEMPER_VAL", ""),
            "to_lot_cd":      row.get("TO_LOT_CD", ""),
            "to_temp":        row.get("TO_TEMPER_VAL", ""),
            "conv_start_min": conv_start,
            "conv_end_min":   conv_end,
            "conv_time":      int(row.get("CONV_TIME") or max(conv_end - conv_start, 0)),
        })

    completed: dict[str, int] = {}
    for rec in schedule:
        key = f"{rec['PLAN_PROD_ATTR_VAL']}|{rec.get('OPER_ID', '')}"
        completed[key] = completed.get(key, 0) + int(rec.get("WF_QTY") or 0)

    meta = payload.get("meta", {})
    return {
        "schedule":         schedule,
        "history":          [],
        "event_log":        [],
        "decision_log":     [],
        "conversion_plans": conversion_plans,
        "down_windows":     [],
        "stats": {
            "idle_total": 0,
            "oper_switches": 0,
            "prod_switches": 0,
            "completed_qty": completed,
            "source_file": "output.json",
        },
        "plan":             env_data["plan"],
        "prod_keys":        env_data["prod_keys"],
        "oper_ids":         env_data["oper_ids"],
        "eqp_ids":          env_data["eqp_ids"],
        "sim_end_minutes":  env_data["sim_end_minutes"],
        "algorithm":        meta.get("ALGORITHM", "saved"),
    }


@app.get("/api/inference/result")
def get_inference_result(input_folder: Optional[str] = None):
    global _last_inference
    if input_folder is None and _last_inference is not None:
        return serialize_inference_result(
            _last_inference,
            include_history=False,
            include_event_log=True,
            include_decision_log=True,
        )

    _apply_input_folder(input_folder)

    # 캐시 없으면 env_data만 로드해 prod/oper 키 복원
    try:
        env_data = _load_env_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    import json

    output_dir = CONFIG.path.output_dir
    full_path = output_dir / "result_full.json"
    if full_path.exists():
        with open(full_path, encoding="utf-8") as f:
            saved = json.load(f)
        result = {
            "schedule": saved.get("schedule", []),
            "history": saved.get("history", []),
            "event_log": saved.get("event_log", []),
            "decision_log": saved.get("decision_log", []),
            "conversion_plans": saved.get("conversion_plans", []),
            "down_windows": saved.get("down_windows", []),
            "stats": {**saved.get("stats", {}), "source_file": "result_full.json"},
            "plan": saved.get("plan", env_data["plan"]),
            "prod_keys": env_data["prod_keys"],
            "oper_ids": env_data["oper_ids"],
            "eqp_ids": env_data["eqp_ids"],
            "sim_end_minutes": env_data["sim_end_minutes"],
            "algorithm": saved.get("algorithm", "scheduling_rl"),
            "validation": saved.get("validation"),
        }
        return serialize_inference_result(
            result,
            include_history=False,
            include_event_log=True,
            include_decision_log=True,
        )

    output_path = output_dir / CONFIG.path.output_file
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"저장된 추론 결과가 없습니다: {output_dir}")
    with open(output_path, encoding="utf-8") as f:
        result = _result_from_rts_output(json.load(f), env_data)
    return serialize_inference_result(
        result,
        include_history=False,
        include_event_log=True,
        include_decision_log=True,
    )


def _test_folders_for_fac(
    fac_id: str,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    prevcnt: Optional[int] = None,
) -> list[str]:
    if from_date and to_date:
        return folders_in_period_range(fac_id, "test", from_date, to_date)
    prefix = f"{fac_id}/test/"
    folders = sorted(f for f in list_input_folders() if f.startswith(prefix))
    if prevcnt is not None:
        folders = folders[-prevcnt:]
    return folders


def _validate_algorithms(algorithms: list[str]) -> None:
    for algo in algorithms:
        try:
            validate_algorithm(algo)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


def _get_benchmark_rl_agent():
    """test 벤치마크용 PPO 모델 – 세션 동안 1회만 로드"""
    global _benchmark_rl_agent
    if _benchmark_rl_agent is not None:
        return _benchmark_rl_agent
    probe = SchedulingAgent()
    if not probe.model_exists():
        return None
    _benchmark_rl_agent = SchedulingAgent.load()
    return _benchmark_rl_agent


def _benchmark_single_folder(folder: str, algorithms: list[str]) -> dict:
    """단일 test 데이터셋에 대해 알고리즘 비교 실행."""
    label = folder.rsplit("/", 1)[-1]
    original_folder = CONFIG.path.input_folder_key
    global _env_data_cache
    rl_agent = _get_benchmark_rl_agent() if "scheduling_rl" in algorithms else None
    try:
        _apply_input_folder(folder)
        env_data = _load_env_data()
        payload = run_inference_compare(
            env_data,
            algorithms,
            rl_agent=rl_agent,
            record_history=False,
        )
        return {
            "input_folder": folder,
            "label": label,
            "results": [serialize_inference_result(r, include_history=False) for r in payload["results"]],
            "errors": payload.get("errors", []),
            "plan": payload.get("plan", []),
            "prod_keys": payload.get("prod_keys", []),
            "oper_ids": payload.get("oper_ids", []),
            "eqp_ids": payload.get("eqp_ids", []),
            "sim_end_minutes": payload.get("sim_end_minutes", 0),
        }
    except FileNotFoundError as e:
        return {
            "input_folder": folder,
            "label": label,
            "error": str(e),
            "results": [],
            "errors": [],
            "plan": [],
            "prod_keys": [],
            "oper_ids": [],
            "eqp_ids": [],
            "sim_end_minutes": 0,
        }
    except HTTPException as e:
        detail = e.detail
        msg = detail if isinstance(detail, str) else str(detail)
        return {
            "input_folder": folder,
            "label": label,
            "error": msg,
            "results": [],
            "errors": [],
            "plan": [],
            "prod_keys": [],
            "oper_ids": [],
            "eqp_ids": [],
            "sim_end_minutes": 0,
        }
    finally:
        set_input_folder(original_folder)
        _env_data_cache = None


def _benchmark_response(state: dict) -> dict:
    return {
        "fac_id": state["fac_id"],
        "algorithms": state.get("algorithms", []),
        "status": state.get("status", "idle"),
        "progress": state.get("progress", {"current": 0, "total": 0, "label": ""}),
        "updated_at": state.get("updated_at"),
        "datasets": state.get("datasets", []),
    }


@app.get("/api/test/datasets")
def list_test_datasets(
    fac_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    prevcnt: Optional[int] = None,
):
    fac = validate_path_segment(fac_id or CONFIG.path.fac_id, "FAC_ID")
    if from_date and to_date and prevcnt is not None:
        raise HTTPException(status_code=400, detail="prevcnt와 from_date/to_date를 함께 쓸 수 없습니다.")
    folders = _test_folders_for_fac(fac, from_date=from_date, to_date=to_date, prevcnt=prevcnt)
    return {
        "fac_id": fac,
        "datasets": [
            {"input_folder": f, "label": f.rsplit("/", 1)[-1]}
            for f in folders
        ],
    }


@app.get("/api/test/benchmark/saved")
def get_saved_test_benchmark(fac_id: Optional[str] = None):
    fac = validate_path_segment(fac_id or CONFIG.path.fac_id, "FAC_ID")
    return _benchmark_response(load_benchmark(fac))


@app.delete("/api/test/benchmark/saved")
def delete_saved_test_benchmark(fac_id: Optional[str] = None):
    fac = validate_path_segment(fac_id or CONFIG.path.fac_id, "FAC_ID")
    clear_benchmark(fac)
    return _benchmark_response(empty_state(fac))


@app.post("/api/test/benchmark/init")
def init_test_benchmark(req: TestBenchmarkInitRequest):
    fac_id = validate_path_segment(req.fac_id or CONFIG.path.fac_id, "FAC_ID")
    _validate_algorithms(req.algorithms)
    global _benchmark_rl_agent
    _benchmark_rl_agent = None
    if req.from_date and req.to_date and req.prevcnt is not None:
        raise HTTPException(status_code=400, detail="prevcnt와 from_date/to_date를 함께 쓸 수 없습니다.")
    folders = _test_folders_for_fac(
        fac_id, from_date=req.from_date, to_date=req.to_date, prevcnt=req.prevcnt,
    )
    if not folders:
        raise HTTPException(
            status_code=404,
            detail=f"test 데이터셋이 없습니다 (FAC_ID={fac_id}).",
        )
    state = init_benchmark(fac_id, req.algorithms, len(folders))
    return _benchmark_response(state)


@app.post("/api/test/benchmark/run-one")
def run_test_benchmark_one(req: TestBenchmarkRunOneRequest):
    fac_id = validate_path_segment(req.fac_id or CONFIG.path.fac_id, "FAC_ID")
    _validate_algorithms(req.algorithms)
    entry = _benchmark_single_folder(req.input_folder, req.algorithms)
    state = append_dataset(
        fac_id,
        entry,
        req.progress_current,
        req.progress_total,
        entry["label"],
        req.done,
    )
    state["algorithms"] = req.algorithms
    save_benchmark(state)
    return _benchmark_response(load_benchmark(fac_id))


@app.get("/api/test/optimal-bench")
def get_optimal_bench(algorithms: Optional[str] = None):
    """증명된 최적해 벤치마크(benchmark/optimal) 실행 — 실제 test 데이터셋과 무관,
    코드 안에 미리 정의된 소규모 검증 케이스만 사용한다."""
    algo_list = [a for a in algorithms.split(",") if a] if algorithms else None
    if algo_list:
        _validate_algorithms(algo_list)
    return run_optimal_benchmark(algorithms=algo_list)


@app.get("/api/benchmark/tool-change")
def get_tool_change_bench(algorithms: Optional[str] = None):
    """전환(conversion) 벤치마크(benchmark/tool_change_bench) 실행 — 케이스별
    정답지(오라클) 스케줄 + 알고리즘별 스케줄을 함께 반환해 프런트에서
    간트/KPI 비교에 바로 쓸 수 있게 한다. 실제 test 데이터셋과 무관하게
    코드 내장 케이스 10종만 사용한다."""
    algo_list = [a for a in algorithms.split(",") if a] if algorithms else None
    if algo_list:
        _validate_algorithms(algo_list)
    rl_agent = _get_benchmark_rl_agent() if (algo_list and "scheduling_rl" in algo_list) else None
    report = run_detailed_benchmark(algorithms=algo_list, rl_agent=rl_agent)
    return {
        "algorithms": report["algorithms"],
        "summary": report["summary"],
        "cases": [
            {
                "id": c["id"],
                "category": c["category"],
                "desc": c["desc"],
                "test_focus": c["test_focus"],
                "sim_end_minutes": c["sim_end_minutes"],
                "optimal": c["optimal"],
                "reference": serialize_inference_result(c["reference"], include_history=False),
                "reference_kpi": c["reference_kpi"],
                "results": [serialize_inference_result(r, include_history=False) for r in c["results"]],
                "errors": c["errors"],
                "kpi": c["kpi"],
            }
            for c in report["cases"]
        ],
    }


@app.post("/api/test/benchmark")
def test_benchmark(req: TestBenchmarkRequest):
    fac_id = validate_path_segment(req.fac_id or CONFIG.path.fac_id, "FAC_ID")
    _validate_algorithms(req.algorithms)

    if req.input_folders:
        folders = req.input_folders
    else:
        if req.from_date and req.to_date and req.prevcnt is not None:
            raise HTTPException(status_code=400, detail="prevcnt와 from_date/to_date를 함께 쓸 수 없습니다.")
        folders = _test_folders_for_fac(
            fac_id, from_date=req.from_date, to_date=req.to_date, prevcnt=req.prevcnt,
        )

    if not folders:
        raise HTTPException(
            status_code=404,
            detail=f"test 데이터셋이 없습니다 (FAC_ID={fac_id}). 데이터셋 페이지에서 test 기간을 생성하세요.",
        )

    init_benchmark(fac_id, req.algorithms, len(folders))
    datasets_out = []
    total = len(folders)

    for i, folder in enumerate(folders, start=1):
        entry = _benchmark_single_folder(folder, req.algorithms)
        datasets_out.append(entry)
        append_dataset(
            fac_id,
            entry,
            i,
            total,
            entry["label"],
            done=(i >= total),
        )

    ok_count = sum(1 for d in datasets_out if d["results"])
    if ok_count == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "모든 test 데이터셋에서 추론에 실패했습니다.",
                "datasets": datasets_out,
            },
        )

    return _benchmark_response(load_benchmark(fac_id))
