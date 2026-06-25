"""학습 진행 상태 – UI 폴링용 공유 상태 및 SB3 콜백."""
from __future__ import annotations

import math
import threading
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Literal, Optional

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

TRAIN_BUDGET_TIMESTEPS = "timesteps"
TRAIN_BUDGET_EPISODES = "episodes"
TrainBudgetMode = Literal["timesteps", "episodes"]

# 에피소드 예산 학습 시 상한 (실제 종료는 EpisodeBudgetCallback)
EPISODE_TRAIN_TIMESTEP_CEILING = 50_000_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_float(value: Any, default: float = 0.0) -> float:
    """JSON 직렬화 가능한 float (inf/nan → default)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _sanitize_json(value: Any) -> Any:
    """dict/list 내 inf·nan·numpy 스칼라를 JSON-safe 값으로 변환."""
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return _json_float(value)
    return value


class TrainProgressState:
    """스레드 세이프 학습 진행 상태."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.status: str = "idle"
            self.stop_requested: bool = False
            self.progress: float = 0.0
            self.timesteps: int = 0
            self.total_timesteps: int = 0
            self.episodes: int = 0
            self.total_episodes: int = 0
            self.train_budget_mode: TrainBudgetMode = TRAIN_BUDGET_TIMESTEPS
            self.logs: list[dict[str, str]] = []
            self.series: dict[str, list] = {
                "timesteps": [],
                "ep_rew_mean": [],
                "eval_timesteps": [],
                "eval_reward": [],
                "policy_loss": [],
                "value_loss": [],
                "explained_variance": [],
            }
            self.metrics: Optional[dict] = None
            self.error: Optional[str] = None

    def add_log(self, message: str, level: str = "info") -> None:
        with self._lock:
            self.logs.append({
                "time": _now_iso(),
                "level": level,
                "message": message,
            })
            if len(self.logs) > 400:
                self.logs = self.logs[-400:]

    def set_running(
        self,
        total_timesteps: int = 0,
        *,
        total_episodes: int = 0,
        budget_mode: TrainBudgetMode = TRAIN_BUDGET_TIMESTEPS,
    ) -> None:
        with self._lock:
            self.status = "running"
            self.progress = 0.0
            self.timesteps = 0
            self.episodes = 0
            self.train_budget_mode = budget_mode
            self.total_timesteps = total_timesteps
            self.total_episodes = total_episodes
            self.metrics = None
            self.error = None

    def update_episode_progress(self, episodes: int, total: int) -> None:
        with self._lock:
            self.episodes = episodes
            self.total_episodes = total
            self.progress = min(1.0, episodes / max(total, 1))

    def set_completed(self, metrics: dict) -> None:
        with self._lock:
            self.status = "completed"
            self.progress = 1.0
            self.metrics = _sanitize_json(metrics)

    def set_failed(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def is_stop_requested(self) -> bool:
        with self._lock:
            return self.stop_requested

    def set_stopped(self) -> None:
        with self._lock:
            self.status = "stopped"
            self.stop_requested = False

    def update_progress(self, timesteps: int, total: int) -> None:
        with self._lock:
            self.timesteps = timesteps
            if self.train_budget_mode != TRAIN_BUDGET_TIMESTEPS:
                return
            self.total_timesteps = total
            self.progress = min(1.0, timesteps / max(total, 1))

    def record_rollout_metrics(self, timestep: int, logger_values: dict[str, float]) -> None:
        with self._lock:
            self.timesteps = timestep
            if self.train_budget_mode == TRAIN_BUDGET_TIMESTEPS and self.total_timesteps:
                self.progress = min(1.0, timestep / self.total_timesteps)
            self.series["timesteps"].append(timestep)
            self.series["ep_rew_mean"].append(
                _json_float(logger_values.get("rollout/ep_rew_mean", 0))
            )
            self.series["policy_loss"].append(
                _json_float(logger_values.get("train/policy_gradient_loss", 0))
            )
            self.series["value_loss"].append(
                _json_float(logger_values.get("train/value_loss", 0))
            )
            self.series["explained_variance"].append(
                _json_float(logger_values.get("train/explained_variance", 0))
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = {
                "status": self.status,
                "progress": _json_float(self.progress),
                "timesteps": int(self.timesteps),
                "total_timesteps": int(self.total_timesteps),
                "episodes": int(self.episodes),
                "total_episodes": int(self.total_episodes),
                "train_budget_mode": self.train_budget_mode,
                "logs": list(self.logs),
                "series": deepcopy(self.series),
                "metrics": deepcopy(self.metrics) if self.metrics else None,
                "error": self.error,
            }
        return _sanitize_json(data)


class StopTrainingCallback(BaseCallback):
    """UI 중지 요청 시 PPO learn 조기 종료."""

    def __init__(self, state: TrainProgressState, verbose: int = 0):
        super().__init__(verbose)
        self._state = state

    def _on_step(self) -> bool:
        if self._state.is_stop_requested():
            self._state.add_log("사용자 요청으로 학습 중지")
            return False
        return True


class ProgressCallback(BaseCallback):
    """rollout마다 보상·loss 기록."""

    def __init__(self, state: TrainProgressState, verbose: int = 0):
        super().__init__(verbose)
        self._state = state
        self._last_log_step = 0

    def _on_training_start(self) -> None:
        if self._state.train_budget_mode == TRAIN_BUDGET_EPISODES:
            self._state.add_log(
                f"PPO 학습 시작 (목표 에피소드={self._state.total_episodes:,})"
            )
            return
        total = self.locals.get("total_timesteps", 0)
        self._state.total_timesteps = int(total)
        self._state.add_log(f"PPO 학습 시작 (total_timesteps={total:,})")

    def _on_step(self) -> bool:
        # rollout 종료 시 record_rollout_metrics에서 progress 갱신 – step마다 lock 방지
        if self.num_timesteps % 512 == 0 and self._state.total_timesteps:
            self._state.update_progress(self.num_timesteps, self._state.total_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return
        if hasattr(logger, "dump"):
            logger.dump(self.num_timesteps)
        values = dict(getattr(logger, "name_to_value", {}))
        if not values:
            return
        self._state.record_rollout_metrics(self.num_timesteps, values)
        ep_rew = values.get("rollout/ep_rew_mean")
        if ep_rew is not None and self.num_timesteps - self._last_log_step >= 2048:
            self._last_log_step = self.num_timesteps
            self._state.add_log(
                f"step {self.num_timesteps:,} · ep_rew_mean={ep_rew:.2f}"
            )


class EvalProgressCallback(EvalCallback):
    """EvalCallback + eval 보상을 progress state에 기록."""

    def __init__(self, state: TrainProgressState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = state
        self._last_logged_reward: Optional[float] = None

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        mean_rew = getattr(self, "last_mean_reward", None)
        if mean_rew is not None and mean_rew != self._last_logged_reward:
            self._last_logged_reward = float(mean_rew)
            self._state.series["eval_timesteps"].append(self.num_timesteps)
            self._state.series["eval_reward"].append(_json_float(mean_rew))
            self._state.add_log(
                f"Eval @ {self.num_timesteps:,} · mean_reward={mean_rew:.2f}"
            )
        return continue_training


class EpisodeBudgetCallback(BaseCallback):
    """목표 에피소드 수 도달 시 학습 종료."""

    def __init__(self, state: TrainProgressState, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self._state = state
        self._max_episodes = max_episodes
        self._episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info and info.get("episode"):
                self._episode_count += 1
                self._state.update_episode_progress(self._episode_count, self._max_episodes)
                if self._episode_count >= self._max_episodes:
                    self._state.add_log(f"목표 에피소드 {self._max_episodes:,}회 도달 – 학습 종료")
                    return False
        return True
