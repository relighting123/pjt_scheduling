"""
agent/rl_agent.py – StableBaselines3 MaskablePPO 에이전트 래퍼
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CONFIG
from env.scheduling_env import SchedulingEnv
from agent.train_progress import (
    TrainProgressState,
    ProgressCallback,
    EvalProgressCallback,
    EpisodeBudgetCallback,
    EPISODE_TRAIN_TIMESTEP_CEILING,
    TRAIN_BUDGET_EPISODES,
    TRAIN_BUDGET_TIMESTEPS,
)


def _mask_fn(env: SchedulingEnv) -> np.ndarray:
    return env.action_masks()


class SchedulingAgent:
    """RL 에이전트 – MaskablePPO (SB3 Contrib)"""

    def __init__(self, model: Optional[MaskablePPO] = None):
        self.model: Optional[MaskablePPO] = model

    # ── 학습 ─────────────────────────────────────────────────────────────────

    def train(
        self,
        env_data: Union[dict, List[dict]],
        verbose: int = 1,
        progress_state: Optional[TrainProgressState] = None,
        n_episodes: Optional[int] = None,
    ) -> "SchedulingAgent":
        """
        목적: 주어진 환경 데이터로 PPO 에이전트 학습
        Input:
            env_data (dict | list[dict]): preprocess() 결과. list면 기간별 VecEnv
            verbose  (int):  0=조용히, 1=진행상황 출력
        Output:
            self (체이닝 가능)
        """
        cfg = CONFIG.rl
        model_dir = CONFIG.path.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        datasets: List[dict] = env_data if isinstance(env_data, list) else [env_data]

        def make_env(data: dict):
            def _init():
                env = ActionMasker(SchedulingEnv(data), _mask_fn)
                return Monitor(env)
            return _init

        train_env = DummyVecEnv([make_env(d) for d in datasets])
        eval_env = DummyVecEnv([make_env(datasets[0])])

        callbacks = []
        use_episode_budget = n_episodes is not None and n_episodes > 0
        learn_timesteps = (
            EPISODE_TRAIN_TIMESTEP_CEILING if use_episode_budget else cfg.total_timesteps
        )

        if progress_state is not None:
            if use_episode_budget:
                progress_state.set_running(
                    total_episodes=n_episodes,
                    budget_mode=TRAIN_BUDGET_EPISODES,
                )
            else:
                progress_state.set_running(
                    total_timesteps=cfg.total_timesteps,
                    budget_mode=TRAIN_BUDGET_TIMESTEPS,
                )
            callbacks.append(ProgressCallback(progress_state))
            if use_episode_budget:
                callbacks.append(EpisodeBudgetCallback(progress_state, n_episodes))
            callbacks.append(
                EvalProgressCallback(
                    progress_state,
                    eval_env,
                    best_model_save_path=str(model_dir / "best"),
                    log_path=str(model_dir / "logs"),
                    eval_freq=cfg.eval_freq,
                    deterministic=True,
                    verbose=0,
                )
            )
        else:
            if use_episode_budget:
                from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
                callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=n_episodes))
            callbacks.extend([
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(model_dir / "best"),
                    log_path=str(model_dir / "logs"),
                    eval_freq=cfg.eval_freq,
                    deterministic=True,
                    verbose=0,
                ),
                CheckpointCallback(
                    save_freq=cfg.eval_freq,
                    save_path=str(model_dir / "checkpoints"),
                    name_prefix=cfg.model_name,
                    verbose=0,
                ),
            ])

        self.model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            verbose=verbose,
        )
        if progress_state is not None:
            if len(datasets) > 1:
                progress_state.add_log(f"VecEnv {len(datasets)}개 기간 병렬 학습")
            budget_label = (
                f"n_episodes={n_episodes:,}"
                if use_episode_budget
                else f"total_timesteps={cfg.total_timesteps:,}"
            )
            progress_state.add_log(
                f"하이퍼파라미터: {budget_label}, lr={cfg.learning_rate}, "
                f"n_steps={cfg.n_steps}, batch={cfg.batch_size}, eval_freq={cfg.eval_freq}"
            )
        self.model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            progress_bar=(verbose > 0 and progress_state is None),
        )
        return self

    # ── 저장 / 로드 ──────────────────────────────────────────────────────────

    def save(self, path: str = None):
        """
        목적: 학습된 모델을 파일로 저장
        Input:  path (str) – 저장 경로 (확장자 없이). None이면 config 기본값 사용
        Output: 없음
        """
        if self.model is None:
            raise RuntimeError("학습된 모델이 없습니다. train()을 먼저 실행하세요.")
        save_path = path or str(CONFIG.path.model_dir / CONFIG.rl.model_name)
        self.model.save(save_path)
        print(f"[agent] 모델 저장 → {save_path}.zip")

    @classmethod
    def load(cls, path: str = None) -> "SchedulingAgent":
        """
        목적: 저장된 모델 파일을 로드하여 에이전트 반환
        Input:  path (str) – 모델 파일 경로 (.zip 포함 또는 미포함)
        Output: SchedulingAgent 인스턴스
        """
        load_path = path or str(CONFIG.path.model_dir / CONFIG.rl.model_name)
        model = MaskablePPO.load(load_path)
        print(f"[agent] 모델 로드 ← {load_path}")
        return cls(model=model)

    def model_exists(self, path: str = None) -> bool:
        """
        목적: 저장된 모델 파일 존재 여부 확인
        Input:  path (str)
        Output: bool
        """
        p = Path(path or str(CONFIG.path.model_dir / CONFIG.rl.model_name))
        return p.with_suffix(".zip").exists() or p.exists()

    # ── 예측 ─────────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        kwargs = {"deterministic": deterministic}
        if action_masks is not None:
            kwargs["action_masks"] = action_masks
        action, _ = self.model.predict(obs, **kwargs)
        return np.asarray(action, dtype=np.int64)

    def evaluate(self, env_data: dict, n_episodes: int = 5) -> dict:
        rewards, oper_sws, prod_sws, idles, completions, conversions = [], [], [], [], [], []
        max_steps = int(env_data.get("sim_end_minutes", 1440)) + 500

        for _ in range(n_episodes):
            env = ActionMasker(
                SchedulingEnv(env_data, record_history=False),
                _mask_fn,
            )
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                mask = env.action_masks()
                action = self.predict(obs, action_masks=mask)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                done = terminated or truncated
                steps += 1
                if steps >= max_steps:
                    break

            rewards.append(ep_reward)
            oper_sws.append(info["oper_switches"])
            prod_sws.append(info["prod_switches"])
            idles.append(info["idle_total"])
            conversions.append(info.get("conversions", 0))
            total_done = sum(info["completed_qty"].values())
            completions.append(total_done)

        return {
            "mean_reward":      float(np.mean(rewards)),
            "mean_oper_sw":     float(np.mean(oper_sws)),
            "mean_prod_sw":     float(np.mean(prod_sws)),
            "mean_idle":        float(np.mean(idles)),
            "mean_completion":  float(np.mean(completions)),
            "mean_conversions": float(np.mean(conversions)),
        }
