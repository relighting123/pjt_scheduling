"""
agent/rl_agent.py – StableBaselines3 PPO 에이전트 래퍼
학습(train), 모델 저장/로드, 예측(predict) 기능을 제공합니다.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CONFIG
from env.scheduling_env import SchedulingEnv


class SchedulingAgent:
    """
    RL 에이전트 – PPO 기반 (SB3)

    사용 예:
        agent = SchedulingAgent()
        agent.train(env_data)
        agent.save("models/scheduling_rl")

        agent2 = SchedulingAgent.load("models/scheduling_rl")
        action = agent2.predict(obs)
    """

    def __init__(self, model: Optional[PPO] = None):
        self.model: Optional[PPO] = model

    # ── 학습 ─────────────────────────────────────────────────────────────────

    def train(self, env_data: dict, verbose: int = 1) -> "SchedulingAgent":
        """
        목적: 주어진 환경 데이터로 PPO 에이전트 학습
        Input:
            env_data (dict): preprocessor.preprocess() 반환값
            verbose  (int):  0=조용히, 1=진행상황 출력
        Output:
            self (체이닝 가능)
        """
        cfg = CONFIG.rl
        model_dir = CONFIG.path.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        def make_env():
            env = SchedulingEnv(env_data)
            return Monitor(env)

        train_env = DummyVecEnv([make_env])
        eval_env  = DummyVecEnv([make_env])

        callbacks = [
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
        ]

        self.model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            verbose=verbose,
        )
        self.model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            progress_bar=(verbose > 0),
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
        model = PPO.load(load_path)
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

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        목적: 관측 벡터로부터 행동(LOT 인덱스) 예측
        Input:
            obs (np.ndarray): shape=(obs_dim,)
            deterministic (bool): True=greedy, False=확률적
        Output:
            action (int): 0 ~ max_queue_size-1
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def evaluate(self, env_data: dict, n_episodes: int = 5) -> dict:
        """
        목적: 여러 에피소드를 실행하여 평균 성능 지표 반환
        Input:
            env_data   (dict): preprocessor 반환 데이터
            n_episodes (int):  평가 에피소드 수
        Output:
            {
              "mean_reward":     float,
              "mean_oper_sw":    float,
              "mean_prod_sw":    float,
              "mean_idle":       float,
              "mean_completion": float,
            }
        """
        rewards, oper_sws, prod_sws, idles, completions = [], [], [], [], []

        for _ in range(n_episodes):
            env = SchedulingEnv(env_data)
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action  = self.predict(obs)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                done = terminated or truncated

            rewards.append(ep_reward)
            oper_sws.append(info["oper_switches"])
            prod_sws.append(info["prod_switches"])
            idles.append(info["idle_total"])
            total_done = sum(info["completed_qty"].values())
            completions.append(total_done)

        return {
            "mean_reward":     float(np.mean(rewards)),
            "mean_oper_sw":    float(np.mean(oper_sws)),
            "mean_prod_sw":    float(np.mean(prod_sws)),
            "mean_idle":       float(np.mean(idles)),
            "mean_completion": float(np.mean(completions)),
        }
