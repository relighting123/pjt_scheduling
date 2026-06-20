"""학습 진행 상태 – UI 폴링용 공유 상태 및 SB3 콜백."""
from __future__ import annotations

import math
import threading
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Optional

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


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
            self.progress: float = 0.0
            self.timesteps: int = 0
            self.total_timesteps: int = 0
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

    def set_running(self, total_timesteps: int) -> None:
        with self._lock:
            self.status = "running"
            self.progress = 0.0
            self.timesteps = 0
            self.total_timesteps = total_timesteps
            self.metrics = None
            self.error = None

    def set_completed(self, metrics: dict) -> None:
        with self._lock:
            self.status = "completed"
            self.progress = 1.0
            self.metrics = _sanitize_json(metrics)

    def set_failed(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error

    def update_progress(self, timesteps: int, total: int) -> None:
        with self._lock:
            self.timesteps = timesteps
            self.total_timesteps = total
            self.progress = min(1.0, timesteps / max(total, 1))

    def record_rollout_metrics(self, timestep: int, logger_values: dict[str, float]) -> None:
        with self._lock:
            self.timesteps = timestep
            if self.total_timesteps:
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
                "logs": list(self.logs),
                "series": deepcopy(self.series),
                "metrics": deepcopy(self.metrics) if self.metrics else None,
                "error": self.error,
            }
        return _sanitize_json(data)


class ProgressCallback(BaseCallback):
    """rollout마다 보상·loss 기록."""

    def __init__(self, state: TrainProgressState, verbose: int = 0):
        super().__init__(verbose)
        self._state = state
        self._last_log_step = 0

    def _on_training_start(self) -> None:
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
