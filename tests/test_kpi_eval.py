"""agent/kpi_eval.py – 벤치마크 KPI 기준 평가 · best 모델 선택 검증.

`EvalCallback`(평균 보상 기준)이 아니라 실제 KPI(생산량−전환수)로 best를
고르는 게 핵심이다. shaping 보상이 최고인 체크포인트가 벤치마크 KPI 최고와
일치한다는 보장이 없기 때문.
"""
import json
import numpy as np

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from agent.kpi_eval import (
    KPI,
    KPIEvalCallback,
    KPIResult,
    dedication_baseline_kpi,
    rollout_policy_kpi,
)

RULE_TIMEKEY = "20260712070000"


def _sym_2x2_raw():
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate([("EQP001", "PPK001"), ("EQP002", "PPK002")]):
        for c in range(3):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": 30,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                     "D0_PLAN_QTY": 3, "D1_PLAN_QTY": 3, "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                         "EQP_MODEL_CD": "A", "ST": 30})
    return {"discrete_arrange": discrete, "plan": plan, "flow": flow,
            "abstract_arrange": abstract}


def _load_ed():
    ed = preprocess(_sym_2x2_raw(), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 180
    ed["conversion_minutes"] = 30
    return ed


# ── 점수 정의 ────────────────────────────────────────────────────────────────

def test_score_penalizes_conversions():
    kpi = KPI(produced=20, producible=24, conversions=3)
    assert kpi.score(conv_weight=1.0) == 17.0
    assert kpi.score(conv_weight=2.0) == 14.0


def test_kpi_result_aggregates_across_datasets():
    result = KPIResult([KPI(produced=10, producible=12, conversions=1),
                        KPI(produced=8, producible=10, conversions=2)])
    assert result.produced == 18
    assert result.producible == 22
    assert result.conversions == 3
    assert result.score(1.0) == 15.0
    assert result.as_dict(1.0)["prod_pct"] == 81.8


# ── 기준선 ───────────────────────────────────────────────────────────────────

def test_dedication_baseline_is_optimal_on_symmetric_case():
    """대칭 2x2에서 전담은 전환 0회 · 전량 생산이 최적."""
    ed = _load_ed()
    base = dedication_baseline_kpi(SchedulingRLEnv, [ed])
    assert base.conversions == 0
    assert base.produced == base.producible == 6


def test_rollout_policy_kpi_runs_arbitrary_policy():
    ed = _load_ed()

    def always_first_feasible(obs, mask):
        n_bucket = SchedulingRLEnv(ed, record_history=False,
                                   record_event_log=False)._n_bucket
        feasible = np.flatnonzero(mask[:n_bucket])
        return np.array([int(feasible[0]) if feasible.size else 0, 0], dtype=np.int64)

    result = rollout_policy_kpi(SchedulingRLEnv, [ed], always_first_feasible)
    assert len(result.per_dataset) == 1
    assert result.producible == 6
    assert 0 <= result.produced <= 6


# ── 콜백 ────────────────────────────────────────────────────────────────────

class _StubModel:
    """KPIEvalCallback이 필요로 하는 최소 인터페이스."""

    def __init__(self, bucket_seq, tmp_path):
        self._bucket_seq = bucket_seq
        self._i = 0
        self.saved_to = None
        self.tmp_path = tmp_path

    def predict(self, obs, deterministic=True, action_masks=None):
        n = len(action_masks) - 4
        feasible = np.flatnonzero(action_masks[:n])
        pick = int(feasible[0]) if feasible.size else 0
        return np.array([pick, 0], dtype=np.int64), None

    def save(self, path):
        self.saved_to = path
        (self.tmp_path / "best").mkdir(parents=True, exist_ok=True)
        (self.tmp_path / "best" / "best_model.zip").write_bytes(b"stub")


def _make_callback(tmp_path, eval_freq=1000, baseline=None):
    ed = _load_ed()
    cb = KPIEvalCallback(
        SchedulingRLEnv, [ed],
        eval_freq=eval_freq,
        best_model_save_path=str(tmp_path / "best"),
        history_path=str(tmp_path / "logs" / "kpi_evaluations.json"),
        baseline=baseline,
    )
    cb.model = _StubModel([], tmp_path)
    return cb


def test_callback_saves_best_model_and_history(tmp_path):
    cb = _make_callback(tmp_path)
    cb.num_timesteps = 0
    cb._on_training_start()

    assert cb.best_kpi is not None
    assert cb.best_score > -float("inf")
    assert (tmp_path / "best" / "best_model.zip").exists()

    history_file = tmp_path / "logs" / "kpi_evaluations.json"
    assert history_file.exists()
    payload = json.loads(history_file.read_text(encoding="utf-8"))
    assert len(payload["history"]) == 1
    assert "score" in payload["history"][0]


def test_callback_only_overwrites_best_on_improvement(tmp_path):
    """정책이 그대로면(점수 동일) best_model을 다시 쓰지 않아야 한다 —
    'best'가 사실상 '마지막'이 되어버리는 걸 막는다."""
    cb = _make_callback(tmp_path)
    cb.num_timesteps = 0
    cb._on_training_start()
    first_score = cb.best_score

    cb.model.saved_to = None
    cb.num_timesteps = 5000
    cb.evaluate_now()

    assert cb.best_score == first_score
    assert cb.model.saved_to is None, "점수 개선이 없는데 best_model을 덮어썼다"


def test_callback_records_baseline_delta(tmp_path):
    baseline = KPIResult([KPI(produced=6, producible=6, conversions=0)])
    cb = _make_callback(tmp_path, baseline=baseline)
    cb.num_timesteps = 0
    cb._on_training_start()

    entry = cb.history[0]
    assert entry["baseline_score"] == 6.0
    assert entry["score_delta"] == entry["score"] - 6.0


def test_callback_evaluates_on_eval_freq(tmp_path):
    cb = _make_callback(tmp_path, eval_freq=100)
    cb.num_timesteps = 0
    cb._on_training_start()
    assert len(cb.history) == 1

    cb.num_timesteps = 50
    cb._on_step()
    assert len(cb.history) == 1, "eval_freq 이전에 평가했다"

    cb.num_timesteps = 150
    cb._on_step()
    assert len(cb.history) == 2
