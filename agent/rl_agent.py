"""
agent/rl_agent.py – StableBaselines3 MaskablePPO 에이전트 래퍼
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import CONFIG, model_dir_for
from env.scheduling_env import (
    SchedulingEnv,
    compute_obs_dim,
    format_obs_dim_mismatch,
    validate_obs_shape,
)
from env.scheduling_rl_env import SchedulingRLEnv
from agent.dedication_agent import DedicationAgent, HOLD_ACTION
from agent.train_progress import (
    TrainProgressState,
    ProgressCallback,
    EntropyDecayCallback,
    EvalProgressCallback,
    EpisodeBudgetCallback,
    StopTrainingCallback,
    EPISODE_TRAIN_TIMESTEP_CEILING,
    TRAIN_BUDGET_EPISODES,
    TRAIN_BUDGET_TIMESTEPS,
)


def _mask_fn(env: SchedulingEnv) -> np.ndarray:
    return env.action_masks()


def _collect_expert_transitions(
    data: dict,
    env_cls: type,
    max_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DedicationAgent(전담 배분 휴리스틱)로 1 에피소드를 굴려 (obs, action,
    action_mask) 전문가 시연 데이터를 모은다 — 모방학습(BC) 워밍스타트용.

    DedicationAgent는 (PPK,OPER) 버킷 1개(또는 HOLD_ACTION)만 반환하는
    SchedulingEnv 대상 휴리스틱이라, SchedulingRLEnv의 [bucket, size_level]
    액션에 맞춰 두 가지를 보정한다:
      - HOLD(-1): 이 env는 진짜 보류가 없으므로, 방금까지 커밋 중이던
        버킷(agent._committed)이 있으면 그걸, 없으면 feasible 버킷 중
        아무거나(0번)를 대신 쓴다.
      - size_level: DedicationAgent는 "가능한 한 오래 전담 유지"가 목표라
        큰 블록을 선호하지만, 블록이 이미 진행 중인 스텝은 env가 size_mask를
        레벨 0 하나로 강제한다(action_masks() 참고) — 이때 라벨을 최대
        레벨로 주면 실제로는 마스크 밖의 선택이 되어(로그확률이 마스킹으로
        거의 -inf) BC 학습이 폭주한다. 그래서 매 스텝 그 시점의 size_mask에서
        "허용된 가장 큰 레벨"을 라벨로 쓴다 — 새 블록 시작 시점엔 최대 레벨,
        진행 중이면 자동으로 0이 된다.
    """
    env = env_cls(data, record_history=False, record_event_log=False)
    expert = DedicationAgent(data)
    obs_list: List[np.ndarray] = []
    action_list: List[Tuple[int, int]] = []
    mask_list: List[np.ndarray] = []

    obs, _ = env.reset()
    n_bucket = env._n_bucket
    for _ in range(max_steps):
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

        allowed_levels = np.flatnonzero(mask[n_bucket:])
        size_level = int(allowed_levels[-1]) if allowed_levels.size else 0

        obs_list.append(obs)
        action_list.append((choice, size_level))
        mask_list.append(mask)

        obs, _reward, terminated, truncated, _info = env.step(
            np.array([choice, size_level], dtype=np.int64)
        )
        if terminated or truncated:
            break

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.int64),
        np.asarray(mask_list, dtype=bool),
    )


def _behavior_clone_pretrain(
    model: MaskablePPO,
    datasets: List[dict],
    env_cls: type,
    epochs: int,
    lr: float,
    verbose: int = 1,
) -> None:
    """DedicationAgent 시연으로 PPO 정책을 지도학습 워밍스타트 (RL 학습 전).

    무작위 초기화 대신 이미 알려진 좋은 정책(전담 유지) 근처에서 RL을
    시작시켜 수렴을 앞당기는 목적 — SYM_5x5류 대칭 벤치마크에서 순수 PPO는
    ~20만 스텝을 학습해도 전환 2회 부근에서 정체되는 경향이 있었다.
    """
    obs_parts, action_parts, mask_parts = [], [], []
    for data in datasets:
        obs, action, mask = _collect_expert_transitions(data, env_cls)
        if obs.size:
            obs_parts.append(obs)
            action_parts.append(action)
            mask_parts.append(mask)
    if not obs_parts:
        if verbose:
            print("[bc] 시연 데이터 없음 — 워밍스타트 생략")
        return

    obs_t = torch.as_tensor(np.concatenate(obs_parts), dtype=torch.float32, device=model.device)
    action_t = torch.as_tensor(np.concatenate(action_parts), dtype=torch.long, device=model.device)
    mask_t = torch.as_tensor(np.concatenate(mask_parts), dtype=torch.bool, device=model.device)

    policy = model.policy
    policy.set_training_mode(True)
    n = obs_t.shape[0]
    if verbose:
        print(f"[bc] 워밍스타트 시작 — 시연 {n}건, epochs={epochs}, lr={lr}")
    for group in policy.optimizer.param_groups:
        group["lr"] = lr
    for epoch in range(epochs):
        _values, log_prob, _entropy = policy.evaluate_actions(obs_t, action_t, mask_t)
        loss = -log_prob.mean()
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
        if verbose and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(epochs // 5, 1) == 0):
            print(f"[bc] epoch {epoch + 1}/{epochs} loss={loss.item():.4f}")
    policy.set_training_mode(False)


def _model_obs_dim(model: MaskablePPO) -> int:
    return int(model.observation_space.shape[0])


def _model_zip_candidates(
    explicit: Optional[str] = None, fac_id: Optional[str] = None,
) -> List[Path]:
    if explicit:
        p = Path(explicit)
        return [p.with_suffix(".zip") if p.suffix != ".zip" else p]

    model_dir = model_dir_for(fac_id)
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
    fac_id: Optional[str] = None,
) -> tuple[MaskablePPO, Path]:
    """현재 env obs 차원과 맞는 모델 로드 (없으면 예외)."""
    expected = compute_obs_dim()
    mismatches: List[tuple[str, int]] = []

    for candidate in _model_zip_candidates(explicit, fac_id=fac_id):
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
    fac_note = f" (FAC_ID={fac_id})" if fac_id else ""
    raise FileNotFoundError(
        f"학습된 모델이 없습니다{fac_note}. python main.py train 을 먼저 실행하세요."
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
        env_cls: type = SchedulingRLEnv,
        restore_best: bool = True,
        fac_id: Optional[str] = None,
    ) -> "SchedulingAgent":
        """
        목적: 주어진 환경 데이터로 PPO 에이전트 학습
        Input:
            env_data (dict | list[dict]): preprocess() 결과. list면 기간별 VecEnv
            verbose  (int):  0=조용히, 1=진행상황 출력
            restore_best (bool): 학습 종료 후 EvalCallback이 저장해둔
                best_model.zip이 있으면 그걸로 self.model을 되돌린다(기본
                True). PPO가 이미 좋은 정책을 찾은 뒤 계속 학습하며 오히려
                더 나빠지는 경우가 있어, 학습이 끝난 시점의 모델을 무조건
                쓰지 않고 학습 중 가장 평가가 좋았던 체크포인트를 채택한다.
            fac_id (str, optional): 지정하면 models/{FAC_ID}/ 아래에 체크포인트·
                best·logs를 분리해서 저장한다(FAC_ID별 모델 관리). 미지정 시
                기존처럼 공용 models/ 그대로 사용.
        Output:
            self (체이닝 가능)
        """
        cfg = CONFIG.rl
        model_dir = model_dir_for(fac_id)
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
        if cfg.ent_coef_final != cfg.ent_coef:
            callbacks.append(EntropyDecayCallback(
                cfg.ent_coef, cfg.ent_coef_final,
                decay_fraction=cfg.ent_coef_decay_fraction,
            ))
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
            ent_coef=cfg.ent_coef,
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

        if cfg.bc_pretrain_epochs > 0:
            if progress_state is not None:
                progress_state.add_log(
                    f"모방학습 워밍스타트: epochs={cfg.bc_pretrain_epochs}, lr={cfg.bc_pretrain_lr}"
                )
            _behavior_clone_pretrain(
                self.model, datasets, env_cls,
                epochs=cfg.bc_pretrain_epochs, lr=cfg.bc_pretrain_lr,
                verbose=verbose,
            )

        self.model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            progress_bar=(verbose > 0 and progress_state is None),
        )

        if restore_best:
            best_path = model_dir / "best" / "best_model.zip"
            if best_path.exists():
                # EvalCallback/EvalProgressCallback가 eval_freq마다 평가해 가장
                # 좋았던 체크포인트를 best_model.zip에 저장해둔다. PPO는 이미
                # 좋은 정책을 찾은 뒤에도 계속 학습하면서 오히려 더 나빠지는
                # 경우가 실험으로 확인돼(SYM_5x5: 1만 스텝에 전환 0회 도달 후
                # 유지하다가 후반에 다시 나빠짐), 학습이 끝난 시점의 모델을
                # 그대로 쓰는 대신 best_model로 되돌린다.
                if verbose:
                    print(f"[agent] 학습 종료 시점보다 나은 체크포인트로 복원 ← {best_path}")
                self.model = MaskablePPO.load(str(best_path), device=cfg.device)
        return self

    # ── 저장 / 로드 ──────────────────────────────────────────────────────────

    def save(self, path: str = None, fac_id: Optional[str] = None):
        """
        목적: 학습된 모델을 파일로 저장
        Input:  path (str) – 저장 경로 (확장자 없이). None이면 기본값 사용
                fac_id (str, optional) – models/{FAC_ID}/ 아래에 저장 (path 미지정 시만 적용)
        Output: 없음
        """
        if self.model is None:
            raise RuntimeError("학습된 모델이 없습니다. train()을 먼저 실행하세요.")
        save_path = path or str(model_dir_for(fac_id) / CONFIG.rl.model_name)
        self.model.save(save_path)
        print(f"[agent] 모델 저장 → {save_path}.zip")

    @classmethod
    def load(
        cls,
        path: str = None,
        env_data: Optional[dict] = None,
        fac_id: Optional[str] = None,
    ) -> "SchedulingAgent":
        """
        목적: 저장된 모델 파일을 로드하여 에이전트 반환
        Input:  path (str)       – 명시적 경로. None이면 기본 후보군 탐색
                env_data (dict)  – obs_dim 진단용
                fac_id (str, optional) – models/{FAC_ID}/ 아래에서 탐색(path 미지정 시만 적용)
        Output: SchedulingAgent 인스턴스
        """
        model, load_path = _load_compatible_model(path, env_data=env_data, fac_id=fac_id)
        print(f"[agent] 모델 로드 ← {load_path} (obs_dim={_model_obs_dim(model)})")
        return cls(model=model)

    def model_exists(self, path: str = None, fac_id: Optional[str] = None) -> bool:
        """
        목적: 저장된 모델 파일 존재 여부 확인
        Input:  path (str)
                fac_id (str, optional)
        Output: bool
        """
        try:
            _load_compatible_model(path, fac_id=fac_id)
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
                SchedulingRLEnv(env_data, record_history=False, record_event_log=False),
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
