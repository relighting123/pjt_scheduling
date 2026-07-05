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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import CONFIG
from env.scheduling_env import (
    SchedulingEnv,
    compute_obs_dim,
    format_obs_dim_mismatch,
    validate_obs_shape,
)
from agent.train_progress import (
    TrainProgressState,
    ProgressCallback,
    EvalProgressCallback,
    EpisodeBudgetCallback,
    StopTrainingCallback,
    EPISODE_TRAIN_TIMESTEP_CEILING,
    TRAIN_BUDGET_EPISODES,
    TRAIN_BUDGET_TIMESTEPS,
)


BULKFILL_MODEL_NAME = "bulkfill_model"


def _mask_fn(env: SchedulingEnv) -> np.ndarray:
    return env.action_masks()


def _model_name_for_algorithm(algorithm: str) -> str:
    """알고리즘별 모델 파일명 반환 (확장자 없음)."""
    if algorithm == "bulkfill":
        return BULKFILL_MODEL_NAME
    return CONFIG.rl.model_name


def _model_obs_dim(model: MaskablePPO) -> int:
    return int(model.observation_space.shape[0])


def _model_zip_candidates(explicit: Optional[str] = None) -> List[Path]:
    if explicit:
        p = Path(explicit)
        return [p.with_suffix(".zip") if p.suffix != ".zip" else p]

    model_dir = CONFIG.path.model_dir
    name = CONFIG.rl.model_name
    candidates: List[Path] = [
        model_dir / f"{name}.zip",
        model_dir / "best" / "best_model.zip",
    ]
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts = sorted(
            ckpt_dir.glob(f"{name}_*_steps.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(ckpts)
    return candidates


def _load_compatible_model(
    explicit: Optional[str] = None,
    env_data: Optional[dict] = None,
) -> tuple[MaskablePPO, Path]:
    """현재 env obs 차원과 맞는 모델 로드 (없으면 예외)."""
    expected = compute_obs_dim()
    mismatches: List[tuple[str, int]] = []

    for candidate in _model_zip_candidates(explicit):
        if not candidate.exists():
            continue
        model = MaskablePPO.load(str(candidate))
        dim = _model_obs_dim(model)
        if dim == expected:
            return model, candidate
        mismatches.append((candidate.name, dim))

    if mismatches:
        model_files = [f"{name} (obs_dim={dim})" for name, dim in mismatches]
        msg = format_obs_dim_mismatch(
            expected,
            mismatches[0][1],
            env_data=env_data,
            source="모델 로드",
            model_files=model_files,
        )
        raise ValueError(msg)
    raise FileNotFoundError(
        "학습된 모델이 없습니다. python main.py train 을 먼저 실행하세요."
    )


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
        env_cls: type = SchedulingEnv,
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
        n_envs = max(cfg.n_envs, 1)

        def make_env(data: dict):
            def _init():
                env = ActionMasker(
                    env_cls(data, record_history=False, record_event_log=False),
                    _mask_fn,
                )
                return Monitor(env)
            return _init

        # n_envs > 1 이면 같은 데이터를 n_envs 개 프로세스에서 병렬 롤아웃
        # 기간이 여러 개면 기간 × n_envs 조합으로 확장
        train_fns = [make_env(d) for d in datasets for _ in range(n_envs)]
        if n_envs > 1:
            train_env = SubprocVecEnv(train_fns, start_method="fork")
        else:
            train_env = DummyVecEnv(train_fns)
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
            callbacks.append(StopTrainingCallback(progress_state))
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

        # n_envs > 1이면 롤아웃 버퍼(n_steps × total_envs)가 커지므로 batch_size도 비례 확장
        total_envs = len(datasets) * n_envs
        effective_batch = cfg.batch_size * max(total_envs, 1)
        # batch_size는 rollout buffer(n_steps × total_envs)의 약수여야 함
        rollout_size = cfg.n_steps * total_envs
        while rollout_size % effective_batch != 0:
            effective_batch -= 1

        self.model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=effective_batch,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            verbose=verbose,
            device=cfg.device,
        )
        if progress_state is not None:
            n_total_envs = len(datasets) * n_envs
            if n_total_envs > 1:
                progress_state.add_log(
                    f"VecEnv {n_total_envs}개 "
                    f"({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'}, "
                    f"기간 {len(datasets)}개 × n_envs {n_envs})"
                )
            budget_label = (
                f"n_episodes={n_episodes:,}"
                if use_episode_budget
                else f"total_timesteps={cfg.total_timesteps:,}"
            )
            progress_state.add_log(
                f"하이퍼파라미터: {budget_label}, lr={cfg.learning_rate}, "
                f"n_steps={cfg.n_steps}, batch={effective_batch}(base={cfg.batch_size}×{total_envs}envs), "
                f"eval_freq={cfg.eval_freq}, device={cfg.device}, n_envs={n_envs}"
            )
        self.model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            progress_bar=(verbose > 0 and progress_state is None),
        )
        return self

    # ── 저장 / 로드 ──────────────────────────────────────────────────────────

    def save(self, path: str = None, algorithm: str = "rl"):
        """
        목적: 학습된 모델을 파일로 저장
        Input:  path (str) – 저장 경로 (확장자 없이). None이면 알고리즘별 기본값 사용
                algorithm   – "rl" | "bulkfill" (파일명 결정)
        Output: 없음
        """
        if self.model is None:
            raise RuntimeError("학습된 모델이 없습니다. train()을 먼저 실행하세요.")
        save_path = path or str(CONFIG.path.model_dir / _model_name_for_algorithm(algorithm))
        self.model.save(save_path)
        print(f"[agent] 모델 저장 → {save_path}.zip")

    @classmethod
    def load(cls, path: str = None, env_data: Optional[dict] = None,
             algorithm: str = "bulkfill") -> "SchedulingAgent":
        """
        목적: 저장된 모델 파일을 로드하여 에이전트 반환
        Input:  path (str)       – 명시적 경로. None이면 알고리즘별 기본값 탐색
                env_data (dict)  – obs_dim 진단용
                algorithm (str)  – "rl" | "bulkfill"
        Output: SchedulingAgent 인스턴스
        """
        # algorithm별 기본 파일명 우선, 없으면 기존 후보군 탐색
        if path is None and algorithm != "rl":
            algo_path = CONFIG.path.model_dir / _model_name_for_algorithm(algorithm)
            if (algo_path.with_suffix(".zip")).exists():
                path = str(algo_path)
        model, load_path = _load_compatible_model(path, env_data=env_data)
        print(f"[agent] 모델 로드 ← {load_path} (obs_dim={_model_obs_dim(model)})")
        return cls(model=model)

    def model_exists(self, path: str = None, algorithm: str = "bulkfill") -> bool:
        """
        목적: 저장된 모델 파일 존재 여부 확인
        Input:  path (str)
                algorithm (str) – "rl" | "bulkfill" (path 미지정 시 파일명 결정에 사용)
        Output: bool
        """
        if path is None and algorithm != "rl":
            algo_path = CONFIG.path.model_dir / _model_name_for_algorithm(algorithm)
            if (algo_path.with_suffix(".zip")).exists():
                path = str(algo_path)
        try:
            _load_compatible_model(path)
            return True
        except (FileNotFoundError, ValueError):
            return False

    # ── 예측 ─────────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
        env_data: Optional[dict] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        validate_obs_shape(
            obs,
            expected_dim=_model_obs_dim(self.model),
            env_data=env_data,
            source="RL predict",
        )
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
