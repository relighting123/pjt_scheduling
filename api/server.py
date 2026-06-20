"""
api/server.py – FastAPI 백엔드
React UI가 호출하는 REST API를 제공합니다.

실행: uvicorn api.server:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder, list_input_folders, train_snapshot_now, PERIOD_SPLITS
from data.loader import load_data, validate_data, fetch_from_db, fetch_period_range
from data.generator import (
    generate_sample_data,
    generate_sample_period_range,
    list_sample_scenarios,
    bootstrap_facility_datasets,
    generator_config_from_dict,
    generator_config_to_dict,
    DEFAULT_GENERATOR_CONFIG,
)
from data.preprocessor import preprocess
from agent.rl_agent import SchedulingAgent
from agent.registry import ALGORITHMS, validate_algorithm
from inference.runner import run_inference, run_inference_compare, save_result
from api.serializers import env_data_summary, serialize_inference_result, serialize_compare_response

app = FastAPI(title="Post-Scheduling RL API", version="1.0.0")

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


def _apply_input_folder(folder: Optional[str]) -> None:
    global _env_data_cache
    if folder:
        set_input_folder(folder)
        _env_data_cache = None


def _load_env_data() -> dict:
    global _env_data_cache
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    _env_data_cache = preprocess(raw)
    return _env_data_cache


# ── 요청 스키마 ───────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    total_timesteps: int = Field(default=CONFIG.rl.total_timesteps, ge=1000)
    learning_rate: float = Field(default=CONFIG.rl.learning_rate, gt=0)
    w_same_oper: float = Field(default=CONFIG.reward.w_same_oper)
    w_idle_per_min: float = Field(default=CONFIG.reward.w_idle_per_min)


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
    st_min: float = Field(default=60.0, ge=1)
    st_max: float = Field(default=180.0, ge=1)
    st_std: float = Field(default=20.0, ge=0)
    eligibility: float = Field(default=0.7, ge=0, le=1)
    plan_qty_min: int = Field(default=25, ge=0)
    plan_qty_max: int = Field(default=150, ge=1)
    plan_priority: int = Field(default=1, ge=1, le=9)
    train_period_count: int = Field(default=3, ge=1, le=365)
    test_period_count: int = Field(default=1, ge=1, le=365)
    split_qty: int = Field(default=3, ge=1, le=100)
    seed: Optional[int] = Field(default=None)


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


def _split_input_path(paths: dict, split: str) -> str:
    entry = paths[split]
    if isinstance(entry, list):
        return entry[-1]["input"]
    return entry["input"]


@app.post("/api/sample")
def create_sample(req: SampleRequest = Body(default_factory=SampleRequest)):
    from data.generator import SAMPLE_SCENARIOS

    if req.scenario not in SAMPLE_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"알 수 없는 시나리오: {req.scenario}")

    try:
        gen_cfg = generator_config_from_dict(
            req.generator_config.model_dump() if req.generator_config else None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    global _env_data_cache
    _env_data_cache = None

    if req.bootstrap:
        info = bootstrap_facility_datasets(
            fac_id=req.fac_id, scenario=req.scenario, gen_config=gen_cfg,
        )
        snap = info["train_snapshot"]
        set_input_folder(f"{req.fac_id}/train/{snap}")
        path = _split_input_path(info["paths"], "train")
    elif req.use_period_count:
        count = (
            gen_cfg.train_period_count if req.split == "train"
            else gen_cfg.test_period_count if req.split == "test"
            else 1
        )
        paths = generate_sample_period_range(
            scenario=req.scenario,
            fac_id=req.fac_id,
            split=req.split,
            gen_config=gen_cfg,
            period_count=count,
        )
        last = paths[-1]
        if req.split in PERIOD_SPLITS:
            set_input_folder(f"{req.fac_id}/{req.split}/{last.parent.name}")
        else:
            set_input_folder(f"{req.fac_id}/{req.split}")
        path = last
    elif req.from_date and req.to_date:
        paths = generate_sample_period_range(
            scenario=req.scenario,
            fac_id=req.fac_id,
            from_date=req.from_date,
            to_date=req.to_date,
            split=req.split,
            gen_config=gen_cfg,
        )
        last = paths[-1]
        set_input_folder(f"{req.fac_id}/{req.split}/{last.parent.name}")
        path = last
    elif req.from_date or req.to_date:
        raise HTTPException(status_code=400, detail="from_date와 to_date를 함께 지정하세요.")
    else:
        path = generate_sample_data(
            scenario=req.scenario,
            fac_id=req.fac_id,
            split=req.split,
            gen_config=gen_cfg,
        )
        if req.split in PERIOD_SPLITS:
            set_input_folder(f"{req.fac_id}/{req.split}/{path.parent.name}")
        else:
            set_input_folder(f"{req.fac_id}/{req.split}")

    return {
        "message": f"샘플 데이터가 생성되었습니다. ({path})",
        "scenario": req.scenario,
        "input_folder": CONFIG.path.input_folder_key,
        "input_dir": str(path),
        "generator_config": generator_config_to_dict(gen_cfg),
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
            )
            path = paths[-1]
        elif req.from_date or req.to_date:
            raise ValueError("from_date와 to_date를 함께 지정하세요.")
        else:
            snap = req.snapshot or (train_snapshot_now() if req.split == "train" else None)
            path = fetch_from_db(fac_id=req.fac_id, split=req.split, snapshot=snap)
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


@app.post("/api/train")
def train(req: TrainRequest):
    try:
        env_data = _load_env_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    CONFIG.rl.total_timesteps = req.total_timesteps
    CONFIG.rl.learning_rate = req.learning_rate
    CONFIG.reward.w_same_oper = req.w_same_oper
    CONFIG.reward.w_idle_per_min = req.w_idle_per_min

    agent = SchedulingAgent()
    agent.train(env_data, verbose=0)
    agent.save()
    metrics = agent.evaluate(env_data, n_episodes=3)
    return {"message": "학습 완료", "metrics": metrics}


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
    save_result(result, output_dir=CONFIG.path.infer_output_dir)
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
        "initial_schedule": saved.get("initial_schedule", []),
        "history": saved.get("history", []),
        "stats": saved.get("stats", {}),
        "plan": saved.get("plan", []),
        "prod_keys": env_data["prod_keys"],
        "oper_ids": env_data["oper_ids"],
        "eqp_ids": env_data["eqp_ids"],
        "sim_end_minutes": env_data["sim_end_minutes"],
        "algorithm": saved.get("algorithm", "rl"),
    }
    return serialize_inference_result(result)
