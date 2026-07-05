"""백그라운드 RL 학습 – UI 폴링용."""
from __future__ import annotations

import threading
from typing import Optional, Union

from config import CONFIG, apply_reward_params
from agent.rl_agent import SchedulingAgent
from agent.train_progress import TrainProgressState, TRAIN_BUDGET_EPISODES, TRAIN_BUDGET_TIMESTEPS
from agent.training_report import save_training_convergence_report

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
        apply_reward_params(params)

        from env.scheduling_env import SchedulingEnv
        env_cls = SchedulingEnv
        if params.get("algorithm") == "bulkfill":
            from env.bulkfill_env import BulkFillEnv
            env_cls = BulkFillEnv
            train_progress.add_log("BulkFillEnv 모드로 학습")

        agent = SchedulingAgent()
        payload = env_list if len(env_list) > 1 else env_list[0]
        train_kwargs = {"verbose": 0, "progress_state": train_progress, "env_cls": env_cls}
        if budget_mode == TRAIN_BUDGET_EPISODES and n_episodes:
            train_kwargs["n_episodes"] = int(n_episodes)
        algorithm = params.get("algorithm", "bulkfill")
        agent.train(payload, **train_kwargs)
        if train_progress.is_stop_requested():
            train_progress.add_log("학습 중지됨 – 부분 모델 저장 중…")
            agent.save(algorithm=algorithm)
            report = save_training_convergence_report(
                CONFIG.path.model_dir, algorithm=algorithm,
                progress_series=train_progress.snapshot()["series"],
            )
            train_progress.add_log(f"수렴 리포트 저장: {report['json_path']}")
            train_progress.set_stopped()
            train_progress.add_log("학습 중지 완료 (부분 모델 저장됨)")
            return
        train_progress.add_log("모델 저장 중…")
        agent.save(algorithm=algorithm)
        eval_eps = int(n_episodes) if budget_mode == TRAIN_BUDGET_EPISODES and n_episodes else 1
        train_progress.add_log(f"학습 후 평가 ({eval_eps} 에피소드)…")
        metrics = agent.evaluate(env_list[0], n_episodes=eval_eps)
        report = save_training_convergence_report(
            CONFIG.path.model_dir, algorithm=algorithm,
            progress_series=train_progress.snapshot()["series"],
            eval_metrics=metrics,
        )
        train_progress.add_log(
            f"수렴 리포트 저장: {report['json_path']} "
            f"({'PNG 포함' if report['png_path'] else 'PNG 없음(matplotlib 미설치)'}) "
            f"— 판정: {report['verdict']}"
        )
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
        budget_mode = params.get("train_budget_mode", "timesteps")
        n_episodes = params.get("n_episodes")
        if budget_mode == TRAIN_BUDGET_EPISODES and n_episodes:
            train_progress.set_running(
                total_episodes=int(n_episodes),
                budget_mode=TRAIN_BUDGET_EPISODES,
            )
        else:
            train_progress.set_running(
                total_timesteps=int(params["total_timesteps"]),
                budget_mode=TRAIN_BUDGET_TIMESTEPS,
            )
        train_progress.add_log("학습 스레드 시작…")
        _train_thread = threading.Thread(
            target=_run_train,
            args=(env_data, params),
            daemon=True,
        )
        _train_thread.start()
    return True
