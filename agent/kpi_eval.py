"""
agent/kpi_eval.py – 벤치마크 KPI 기준 평가 · best 모델 선택

왜 필요한가
-----------
`EvalCallback`은 **평균 보상**이 가장 높은 체크포인트를 best_model로 남긴다.
그런데 이 프로젝트의 step reward는 전부 대리지표(same_setup, bulk_block_bonus,
redundant_cover…)라서, 보상 최고점이 벤치마크 KPI(생산량·전환수) 최고점과
일치한다는 보장이 없다. 실제로 "학습 곡선상 보상은 올라가는데 벤치마크는
나빠지는" 현상이 나온다.

이 모듈은 eval 시점마다 **실제 KPI로 채점**해서
  score = 생산량(sim_end 안에 완료된 carrier) − w × 전환수
가 최고인 체크포인트를 best_model.zip으로 남긴다. 학습 시작 시
DedicationAgent 기준선 KPI도 한 번 측정해두고, 매 eval마다 기준선 대비
격차를 로그·차트 계열로 남긴다 — "RL이 전담 휴리스틱을 넘었는가"를 학습
도중에 바로 볼 수 있게.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agent.dedication_agent import DedicationAgent, HOLD_ACTION
from env.scheduling_rl_env import build_env

LogFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


@dataclass
class KPI:
    produced: int = 0
    producible: int = 0
    conversions: int = 0
    reward: float = 0.0
    steps: int = 0

    def score(self, conv_weight: float = 1.0) -> float:
        return float(self.produced - conv_weight * self.conversions)

    def as_dict(self, conv_weight: float = 1.0) -> dict:
        return {
            "produced": self.produced,
            "producible": self.producible,
            "conversions": self.conversions,
            "reward": round(float(self.reward), 3),
            "score": round(self.score(conv_weight), 3),
        }


@dataclass
class KPIResult:
    """여러 데이터셋 합산 KPI."""

    per_dataset: List[KPI] = field(default_factory=list)

    @property
    def produced(self) -> int:
        return sum(k.produced for k in self.per_dataset)

    @property
    def producible(self) -> int:
        return sum(k.producible for k in self.per_dataset)

    @property
    def conversions(self) -> int:
        return sum(k.conversions for k in self.per_dataset)

    @property
    def reward(self) -> float:
        return float(sum(k.reward for k in self.per_dataset))

    def score(self, conv_weight: float = 1.0) -> float:
        return float(self.produced - conv_weight * self.conversions)

    def as_dict(self, conv_weight: float = 1.0) -> dict:
        return {
            "produced": self.produced,
            "producible": self.producible,
            "conversions": self.conversions,
            "reward": round(self.reward, 3),
            "score": round(self.score(conv_weight), 3),
            "prod_pct": round(100 * self.produced / max(self.producible, 1), 1),
        }


def _episode_kpi(env, produced_from_schedule: bool = True) -> KPI:
    sim = env.sim
    sim_end = sim.sim_end
    if produced_from_schedule:
        produced = sum(1 for r in sim.schedule if r.get("END_TM", 0) <= sim_end)
    else:
        produced = int(sum(sim.stats["completed_qty"].values()))
    return KPI(
        produced=produced,
        producible=env.producible_carriers(),
        conversions=int(sim.stats.get("conversions", 0)),
    )


def rollout_policy_kpi(
    env_cls: type,
    datasets: List[dict],
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    max_steps_pad: int = 500,
) -> KPIResult:
    """정책(predict_fn)으로 각 데이터셋 1 에피소드를 굴려 KPI 집계.

    predict_fn(obs, action_mask) -> np.ndarray([bucket, size_level])
    """
    result = KPIResult()
    for data in datasets:
        env = build_env(env_cls, data)
        obs, _ = env.reset()
        max_steps = int(data.get("sim_end_minutes", 1440)) + max_steps_pad
        total_r = 0.0
        steps = 0
        while steps < max_steps:
            mask = env.action_masks()
            action = predict_fn(obs, mask)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_r += float(reward)
            steps += 1
            if terminated or truncated:
                break
        kpi = _episode_kpi(env)
        kpi.reward = total_r
        kpi.steps = steps
        result.per_dataset.append(kpi)
    return result


def dedication_baseline_kpi(env_cls: type, datasets: List[dict]) -> KPIResult:
    """DedicationAgent를 SchedulingRLEnv에서 굴려 기준선 KPI를 측정.

    주의: 이 값은 `inference.run_inference(algorithm="dedication")`(SchedulingEnv
    경로, HOLD 지원)와 완전히 동일하지 않다 — RL env에는 진짜 HOLD가 없어
    보류가 '직전 커밋 버킷 재선택'으로 대체되기 때문이다. **RL 환경 안에서**
    전담 정책이 낼 수 있는 성적을 재는 것이 목적이므로 이 정의가 맞다
    (학습 중 정책과 같은 조건에서의 공정 비교).
    """
    result = KPIResult()
    for data in datasets:
        env = build_env(env_cls, data)
        expert = DedicationAgent(data)
        obs, _ = env.reset()
        n_bucket = env._n_bucket
        max_steps = int(data.get("sim_end_minutes", 1440)) + 500
        total_r = 0.0
        steps = 0
        while steps < max_steps:
            mask = env.action_masks()
            eqp_id = env.sim.current_idle_eqp()
            choice = int(expert.predict(env.sim)[0]) if eqp_id is not None else 0
            if choice == HOLD_ACTION:
                committed = expert._committed.get(eqp_id) if eqp_id is not None else None
                if committed is not None:
                    choice = committed
                else:
                    feasible = np.flatnonzero(mask[:n_bucket])
                    choice = int(feasible[0]) if feasible.size else 0
            allowed = np.flatnonzero(mask[n_bucket:])
            level = int(allowed[-1]) if allowed.size else 0
            obs, reward, terminated, truncated, _info = env.step(
                np.array([choice, level], dtype=np.int64)
            )
            total_r += float(reward)
            steps += 1
            if terminated or truncated:
                break
        kpi = _episode_kpi(env)
        kpi.reward = total_r
        kpi.steps = steps
        result.per_dataset.append(kpi)
    return result


class KPIEvalCallback(BaseCallback):
    """eval_freq마다 KPI로 채점하고 최고 점수 모델을 best_model.zip에 저장.

    `EvalCallback`(보상 기준)을 대체한다. 저장 경로/파일명이 같아서
    `SchedulingAgent.train(restore_best=True)`가 그대로 이 best를 복원한다.
    """

    def __init__(
        self,
        env_cls: type,
        datasets: List[dict],
        *,
        eval_freq: int,
        best_model_save_path: Optional[str] = None,
        history_path: Optional[str] = None,
        conv_weight: float = 1.0,
        baseline: Optional[KPIResult] = None,
        state=None,
        log: LogFn = _noop,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._env_cls = env_cls
        self._datasets = datasets
        self._eval_freq = max(int(eval_freq), 1)
        self._save_path = Path(best_model_save_path) if best_model_save_path else None
        self._history_path = Path(history_path) if history_path else None
        self._conv_weight = conv_weight
        self._baseline = baseline
        self._state = state
        self._log = log
        self.best_score: float = -float("inf")
        self.best_kpi: Optional[KPIResult] = None
        self.best_timestep: int = 0
        self.history: List[dict] = []
        self._last_eval_step = 0

    # ── 내부 ────────────────────────────────────────────────────────────────

    def _predict(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
        return np.asarray(action, dtype=np.int64)

    def _record(self, kpi: KPIResult) -> None:
        entry = {"timestep": int(self.num_timesteps), **kpi.as_dict(self._conv_weight)}
        if self._baseline is not None:
            entry["baseline_score"] = round(self._baseline.score(self._conv_weight), 3)
            entry["score_delta"] = round(entry["score"] - entry["baseline_score"], 3)
        self.history.append(entry)

        if self._state is not None:
            self._state.record_kpi_point(
                entry["timestep"],
                score=entry["score"],
                produced=entry["produced"],
                conversions=entry["conversions"],
                baseline_score=entry.get("baseline_score") if self._baseline is not None else None,
            )

        # 수렴 리포트(agent/training_report.py)가 읽어갈 원자료.
        # EvalCallback을 대체했기 때문에 evaluations.npz가 생기지 않으므로,
        # KPI 곡선을 이 파일로 남겨 차트/판정의 입력이 되게 한다.
        if self._history_path is not None:
            try:
                self._history_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "conv_weight": self._conv_weight,
                    "baseline": (
                        self._baseline.as_dict(self._conv_weight)
                        if self._baseline is not None else None
                    ),
                    "history": self.history,
                }
                self._history_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
                )
            except OSError:
                pass

    def evaluate_now(self) -> KPIResult:
        kpi = rollout_policy_kpi(self._env_cls, self._datasets, self._predict)
        self._record(kpi)
        score = kpi.score(self._conv_weight)

        msg = (
            f"[kpi] step {self.num_timesteps:,} · 생산 {kpi.produced}/{kpi.producible} "
            f"· 전환 {kpi.conversions} · 점수 {score:.1f}"
        )
        if self._baseline is not None:
            delta = score - self._baseline.score(self._conv_weight)
            msg += f" (Dedication 대비 {delta:+.1f})"
        self._log(msg)

        if score > self.best_score:
            self.best_score = score
            self.best_kpi = kpi
            self.best_timestep = int(self.num_timesteps)
            if self._save_path is not None:
                self._save_path.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self._save_path / "best_model"))
                self._log(f"[kpi] 신기록 — best_model 갱신 (점수 {score:.1f})")
        return kpi

    # ── 콜백 훅 ──────────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        # 워밍스타트(BC) 직후 성적을 반드시 후보로 남긴다. RL이 그보다
        # 나아지지 못하면 restore_best가 이 시점 모델을 되돌린다.
        self.evaluate_now()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self._eval_freq:
            self._last_eval_step = self.num_timesteps
            self.evaluate_now()
        return True

    def _on_training_end(self) -> None:
        self.evaluate_now()
