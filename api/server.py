"""
api/server.py – FastAPI 백엔드
React UI가 호출하는 REST API를 제공합니다.

실행: uvicorn api.server:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder, list_input_folders, train_snapshot_now, PERIOD_SPLITS, validate_path_segment, iter_rule_timekeys, parse_input_folder, latest_period
from data.loader import load_data, validate_data, fetch_from_db, fetch_period_range, preprocess
from agent.rl_agent import SchedulingAgent
from agent.registry import ALGORITHMS, validate_algorithm
from inference.runner import run_inference, run_inference_compare, save_result
from api.serializers import env_data_summary, serialize_inference_result, serialize_compare_response
from api.test_benchmark_store import (
    load_benchmark,
    save_benchmark,
    clear_benchmark,
    init_benchmark,
    append_dataset,
    empty_state,
)
from api.train_service import train_progress, start_training, is_training

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


def _apply_input_folder(folder: Optional[str]) -> None:
    global _env_data_cache
    if folder:
        set_input_folder(folder)
        _env_data_cache = None


def _load_env_data() -> dict:
    global _env_data_cache
    if _env_data_cache is not None:
        return _env_data_cache
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    _env_data_cache = preprocess(raw)
    return _env_data_cache


def _load_env_data_for_folder(folder: str) -> dict:
    """지정 train 스냅샷 1개 로드 (전역 입력 경로는 복원)."""
    original = CONFIG.path.input_folder_key
    try:
        set_input_folder(folder)
        raw = load_data()
        errors = validate_data(raw)
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"errors": errors, "folder": folder},
            )
        return preprocess(raw)
    finally:
        set_input_folder(original)


def _train_folders_for_fac(fac_id: str) -> list[str]:
    prefix = f"{fac_id}/train/"
    return sorted(
        f for f in list_input_folders()
        if f.startswith(prefix)
    )


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
        folders = [
            f"{fac_id}/train/{key}"
            for key in iter_rule_timekeys(req.from_date, req.to_date)
            if f"{fac_id}/train/{key}" in available
        ]
        if not folders:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"기간 {req.from_date}~{req.to_date}에 해당하는 "
                    f"train 데이터가 없습니다."
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

class TrainRequest(BaseModel):
    total_timesteps: int = Field(default=CONFIG.rl.total_timesteps, ge=1000)
    learning_rate: float = Field(default=CONFIG.rl.learning_rate, gt=0)
    w_same_oper: float = Field(default=CONFIG.reward.w_same_oper)
    w_idle_per_min: float = Field(default=CONFIG.reward.w_idle_per_min)
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
    fac_id: Optional[str] = Field(
        default=None,
        description="from/to 기간 검색용 FAC_ID (기본: 현재 설정)",
    )


class InferenceRequest(BaseModel):
    algorithm: str = Field(default="rl", description="rl | minprogress | earliest_st")
    input_folder: Optional[str] = Field(
        default=None,
        description="입력 데이터 폴더명 (external/<name>/)",
    )


class GeneratorConfigModel(BaseModel):
    n_products: int = Field(default=3, ge=1, le=20)
    n_eqps: int = Field(default=3, ge=1, le=20)
    n_opers: int = Field(default=2, ge=1, le=10)
    lots_per_oper: int = Field(default=3, ge=1, le=30)
    wf_qty: int = Field(default=25, ge=1, le=500)
    st_min: float = Field(default=3.0, ge=1, description="장당 ST 하한(분/장)")
    st_max: float = Field(default=8.0, ge=1, description="장당 ST 상한(분/장)")
    st_std: float = Field(default=20.0, ge=0)
    eligibility: float = Field(default=0.7, ge=0, le=1)
    plan_qty_min: int = Field(default=25, ge=0)
    plan_qty_max: int = Field(default=150, ge=1)
    plan_priority: int = Field(default=1, ge=1, le=9)
    train_period_count: int = Field(default=3, ge=1, le=365)
    test_period_count: int = Field(default=1, ge=1, le=365)
    split_qty: int = Field(default=3, ge=1, le=100)
    seed: Optional[int] = Field(default=None)

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, value):
        if value is None or value == "":
            return None
        return value


class SampleRequest(BaseModel):
    fac_id: str = Field(default="FAC001", description="공장 ID")
    split: str = Field(default="train", description="train | test | infer")
    scenario: str = Field(default="random", description="default | single_heavy_wip | random")
    bootstrap: bool = Field(default=False, description="train/test/infer 전체 생성")
    from_date: Optional[str] = Field(default=None, description="시작 RULE_TIMEKEY (YYYYMMDDHHmmss)")
    to_date: Optional[str] = Field(default=None, description="종료 RULE_TIMEKEY (YYYYMMDDHHmmss)")
    generator_config: Optional[GeneratorConfigModel] = Field(default=None)
    use_period_count: bool = Field(
        default=False,
        description="True면 train/test_period_count로 폴더 일괄 생성 (from/to 무시)",
    )


class FetchRequest(BaseModel):
    fac_id: str = Field(default="FAC001")
    split: str = Field(default="train")
    snapshot: Optional[str] = Field(default=None, description="단일 RULE_TIMEKEY (YYYYMMDDHHmmss)")
    from_date: Optional[str] = Field(default=None, description="시작 RULE_TIMEKEY (YYYYMMDDHHmmss)")
    to_date: Optional[str] = Field(default=None, description="종료 RULE_TIMEKEY (YYYYMMDDHHmmss)")
    lot_cd: Optional[str] = Field(
        default=None,
        description="SQL :LOT_CD 바인드 (discrete_arrange 제외, 미지정 시 전체)",
    )


class InputFolderRequest(BaseModel):
    input_folder: str = Field(description="사용할 입력 폴더명")


class CompareRequest(BaseModel):
    algorithms: list[str] = Field(
        min_length=1,
        description="비교할 알고리즘 ID 목록",
    )
    input_folder: Optional[str] = Field(
        default=None,
        description="입력 데이터 폴더명 (external/<name>/)",
    )


class TestBenchmarkRequest(BaseModel):
    algorithms: list[str] = Field(min_length=1, description="비교할 알고리즘 ID 목록")
    fac_id: Optional[str] = Field(default=None, description="FAC_ID (기본: 현재 설정)")
    input_folders: Optional[list[str]] = Field(
        default=None,
        description="test 데이터셋 경로 목록 (미지정 시 fac_id 하위 test 전체)",
    )


class TestBenchmarkInitRequest(BaseModel):
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
    return {"status": "ok"}


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
        "default_w_same_oper": CONFIG.reward.w_same_oper,
        "default_w_idle_per_min": CONFIG.reward.w_idle_per_min,
    }


@app.post("/api/config/input")
def select_input_folder(req: InputFolderRequest):
    try:
        path = set_input_folder(req.input_folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    global _env_data_cache
    _env_data_cache = None
    return {
        "message": f"입력 폴더가 '{req.input_folder}'(으)로 설정되었습니다.",
        "input_folder": CONFIG.path.input_folder_key,
        "input_dir": str(path),
        "output_dir": str(CONFIG.path.output_dir),
    }


@app.get("/api/sample/scenarios")
def sample_scenarios():
    return {"scenarios": list_sample_scenarios()}


@app.get("/api/sample/generator-config")
def get_generator_config_defaults():
    return {"defaults": generator_config_to_dict(DEFAULT_GENERATOR_CONFIG)}


@app.post("/api/sample")
def create_sample(req: SampleRequest = Body(default_factory=SampleRequest)):
    try:
        gen_cfg = generator_config_from_dict(
            req.generator_config.model_dump() if req.generator_config else None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    global _env_data_cache
    _env_data_cache = None

    try:
        result = generate_sample(
            scenario=req.scenario,
            fac_id=req.fac_id,
            split=req.split,
            bootstrap=req.bootstrap,
            from_date=req.from_date,
            to_date=req.to_date,
            use_period_count=req.use_period_count,
            gen_config=gen_cfg,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    path = result["path"]
    return {
        "message": f"샘플 데이터가 생성되었습니다. ({path})",
        "scenario": req.scenario,
        "input_folder": CONFIG.path.input_folder_key,
        "input_dir": str(path),
        "generator_config": result["generator_config"],
    }


@app.post("/api/fetch")
def fetch_dataset(req: FetchRequest):
    try:
        if req.from_date and req.to_date:
            paths = fetch_period_range(
                fac_id=req.fac_id,
                from_date=req.from_date,
                to_date=req.to_date,
                split=req.split,
                lot_cd=req.lot_cd,
            )
            path = paths[-1]
        elif req.from_date or req.to_date:
            raise ValueError("from_date와 to_date를 함께 지정하세요.")
        else:
            snap = req.snapshot or (train_snapshot_now() if req.split == "train" else None)
            path = fetch_from_db(
                fac_id=req.fac_id,
                split=req.split,
                snapshot=snap,
                lot_cd=req.lot_cd,
            )
    except (ValueError, FileNotFoundError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {e}")

    if req.split in PERIOD_SPLITS:
        set_input_folder(f"{req.fac_id}/{req.split}/{path.parent.name}")
    else:
        set_input_folder(f"{req.fac_id}/{req.split}")

    global _env_data_cache
    _env_data_cache = None
    return {
        "message": f"DB 데이터가 JSON으로 저장되었습니다. ({path})",
        "input_folder": CONFIG.path.input_folder_key,
        "input_dir": str(path),
    }


@app.get("/api/data/summary")
def data_summary():
    try:
        env_data = _load_env_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return env_data_summary(env_data)


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
    params = {
        "total_timesteps": req.total_timesteps,
        "learning_rate": req.learning_rate,
        "w_same_oper": req.w_same_oper,
        "w_idle_per_min": req.w_idle_per_min,
        "train_budget_mode": req.train_budget_mode,
        "n_episodes": req.n_episodes,
        "input_folders": folders,
    }
    payload = env_list if len(env_list) > 1 else env_list[0]
    if not start_training(payload, params):
        raise HTTPException(status_code=409, detail="학습을 시작할 수 없습니다.")
    return {"message": "학습 시작", "input_folders": folders}


@app.get("/api/train/status")
def train_status():
    return train_progress.snapshot()


@app.post("/api/train")
def train(req: TrainRequest):
    try:
        env_list, folders = _prepare_train_env_data(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    CONFIG.rl.total_timesteps = req.total_timesteps
    CONFIG.rl.learning_rate = req.learning_rate
    CONFIG.reward.w_same_oper = req.w_same_oper
    CONFIG.reward.w_idle_per_min = req.w_idle_per_min

    agent = SchedulingAgent()
    payload = env_list if len(env_list) > 1 else env_list[0]
    train_kwargs: dict = {"verbose": 0}
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
    _apply_input_folder(req.input_folder)
    try:
        env_data = _load_env_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        validate_algorithm(req.algorithm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent = None
    if req.algorithm == "rl":
        agent = SchedulingAgent()
        if not agent.model_exists():
            raise HTTPException(
                status_code=400,
                detail="학습된 모델이 없습니다. 먼저 학습을 실행하세요.",
            )
        agent = SchedulingAgent.load()

    result = run_inference(env_data, algorithm=req.algorithm, agent=agent)
    result["prod_keys"] = env_data["prod_keys"]
    result["oper_ids"] = env_data["oper_ids"]
    result["eqp_ids"] = env_data["eqp_ids"]
    result["sim_end_minutes"] = env_data["sim_end_minutes"]
    save_result(result, output_dir=CONFIG.path.infer_output_dir, env_data=env_data)
    _last_inference = result
    return serialize_inference_result(result)


@app.post("/api/inference/compare")
def inference_compare(req: CompareRequest):
    _apply_input_folder(req.input_folder)
    try:
        env_data = _load_env_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    for algo in req.algorithms:
        try:
            validate_algorithm(algo)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    payload = run_inference_compare(env_data, req.algorithms)
    if not payload["results"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "실행 가능한 알고리즘이 없습니다.",
                "errors": payload["errors"],
            },
        )
    return serialize_compare_response(payload)


@app.get("/api/inference/result")
def get_inference_result():
    global _last_inference
    if _last_inference is not None:
        return serialize_inference_result(_last_inference)

    # 캐시 없으면 env_data만 로드해 prod/oper 키 복원
    try:
        env_data = _load_env_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="추론 결과가 없습니다.")

    from pathlib import Path
    import json

    full_path = CONFIG.path.infer_output_dir / "result_full.json"
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="추론 결과가 없습니다.")

    with open(full_path, encoding="utf-8") as f:
        saved = json.load(f)

    # history는 파일에 없을 수 있음
    result = {
        "schedule": saved.get("schedule", []),
        "history": saved.get("history", []),
        "event_log": saved.get("event_log", []),
        "conversion_plans": saved.get("conversion_plans", []),
        "stats": saved.get("stats", {}),
        "plan": saved.get("plan", []),
        "prod_keys": env_data["prod_keys"],
        "oper_ids": env_data["oper_ids"],
        "eqp_ids": env_data["eqp_ids"],
        "sim_end_minutes": env_data["sim_end_minutes"],
        "algorithm": saved.get("algorithm", "rl"),
    }
    return serialize_inference_result(result)


def _test_folders_for_fac(fac_id: str) -> list[str]:
    prefix = f"{fac_id}/test/"
    return sorted(
        f for f in list_input_folders()
        if f.startswith(prefix)
    )


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
    rl_agent = _get_benchmark_rl_agent() if "rl" in algorithms else None
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
def list_test_datasets(fac_id: Optional[str] = None):
    fac = validate_path_segment(fac_id or CONFIG.path.fac_id, "FAC_ID")
    folders = _test_folders_for_fac(fac)
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
    folders = _test_folders_for_fac(fac_id)
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


@app.post("/api/test/benchmark")
def test_benchmark(req: TestBenchmarkRequest):
    fac_id = validate_path_segment(req.fac_id or CONFIG.path.fac_id, "FAC_ID")
    _validate_algorithms(req.algorithms)

    if req.input_folders:
        folders = req.input_folders
    else:
        folders = _test_folders_for_fac(fac_id)

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
