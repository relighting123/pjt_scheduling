"""백그라운드 RL 학습 – UI 폴링용."""
from __future__ import annotations

import threading
from typing import Optional, Union

from config import CONFIG
from agent.rl_agent import SchedulingAgent
from agent.train_progress import TrainProgressState, TRAIN_BUDGET_EPISODES

train_progress = TrainProgressState()
_train_lock = threading.Lock()
_train_thread: Optional[threading.Thread] = None


def _run_train(env_data: Union[dict, list], params: dict) -> None:
    global _train_thread
    try:
        env_list = env_data if isinstance(env_data, list) else [env_data]
        folders: list[str] = params.get("input_folders") or []
        budget_mode = params.get("train_budget_mode", "timesteps")
        n_episodes = params.get("n_episodes")
        train_progress.add_log(f"학습 데이터 {len(folders) or len(env_list)}개 기간")
        for folder in folders[:8]:
            train_progress.add_log(f"  · {folder}")
        if len(folders) > 8:
            train_progress.add_log(f"  · … 외 {len(folders) - 8}개")

        CONFIG.rl.total_timesteps = params["total_timesteps"]
        CONFIG.rl.learning_rate = params["learning_rate"]
        CONFIG.reward.w_same_oper = params["w_same_oper"]
        CONFIG.reward.w_idle_per_min = params["w_idle_per_min"]

        agent = SchedulingAgent()
        payload = env_list if len(env_list) > 1 else env_list[0]
        train_kwargs = {"verbose": 0, "progress_state": train_progress}
        if budget_mode == TRAIN_BUDGET_EPISODES and n_episodes:
            train_kwargs["n_episodes"] = int(n_episodes)
        agent.train(payload, **train_kwargs)
        if train_progress.is_stop_requested():
            train_progress.add_log("학습 중지됨 – 부분 모델 저장 중…")
            agent.save()
            train_progress.set_stopped()
            train_progress.add_log("학습 중지 완료 (부분 모델 저장됨)")
            return
        train_progress.add_log("모델 저장 중…")
        agent.save()
        eval_eps = int(n_episodes) if budget_mode == TRAIN_BUDGET_EPISODES and n_episodes else 1
        train_progress.add_log(f"학습 후 평가 ({eval_eps} 에피소드)…")
        metrics = agent.evaluate(env_list[0], n_episodes=eval_eps)
        train_progress.set_completed(metrics)
        train_progress.add_log("학습 완료")
    except Exception as exc:
        train_progress.set_failed(str(exc))
        train_progress.add_log(str(exc), level="error")
    finally:
        with _train_lock:
            _train_thread = None


def is_training() -> bool:
    with _train_lock:
        return _train_thread is not None and _train_thread.is_alive()


def stop_training() -> bool:
    """진행 중인 학습에 중지 요청."""
    if not is_training():
        return False
    train_progress.request_stop()
    train_progress.add_log("학습 중지 요청됨…")
    return True


def start_training(env_data: Union[dict, list], params: dict) -> bool:
    global _train_thread
    with _train_lock:
        if _train_thread is not None and _train_thread.is_alive():
            return False
        train_progress.reset()
        _train_thread = threading.Thread(
            target=_run_train,
            args=(env_data, params),
            daemon=True,
        )
        _train_thread.start()
    return True
