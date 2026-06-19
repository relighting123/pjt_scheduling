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

from config import CONFIG, set_input_folder, validate_input_folder, list_input_folders
from data.loader import load_data, validate_data, generate_sample_data, list_sample_scenarios
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


class SampleRequest(BaseModel):
    input_folder: Optional[str] = Field(
        default=None,
        description="샘플을 생성할 폴더명 (없으면 시나리오 기본 폴더)",
    )
    scenario: str = Field(
        default="default",
        description="default | single_heavy_wip",
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


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return {
        "model_dir": str(CONFIG.path.model_dir),
        "input_folder": CONFIG.path.input_folder,
        "input_dir": str(CONFIG.path.input_dir),
        "output_dir": str(CONFIG.path.output_dir),
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
        "input_folder": CONFIG.path.input_folder,
        "input_dir": str(path),
        "output_dir": str(CONFIG.path.output_dir),
    }


@app.get("/api/sample/scenarios")
def sample_scenarios():
    return {"scenarios": list_sample_scenarios()}


@app.post("/api/sample")
def create_sample(req: SampleRequest = Body(default_factory=SampleRequest)):
    from data.loader import SAMPLE_SCENARIOS

    if req.scenario not in SAMPLE_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"알 수 없는 시나리오: {req.scenario}")

    folder = req.input_folder or SAMPLE_SCENARIOS[req.scenario]["default_folder"]
    set_input_folder(folder)
    path = generate_sample_data(output_dir=CONFIG.path.input_dir, scenario=req.scenario)

    global _env_data_cache
    _env_data_cache = None
    return {
        "message": f"샘플 데이터가 생성되었습니다. ({path})",
        "scenario": req.scenario,
        "input_folder": CONFIG.path.input_folder,
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
    save_result(result)
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

    full_path = CONFIG.path.output_dir / "result_full.json"
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
