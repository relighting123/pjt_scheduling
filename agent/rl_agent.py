"""
agent/rl_agent.py – StableBaselines3 MaskablePPO 에이전트 래퍼
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from config import CONFIG, model_dir_for
from env.scheduling_env import (
    SchedulingEnv,
    compute_obs_dim,
    format_obs_dim_mismatch,
    validate_obs_shape,
)
from env.scheduling_rl_env import (
    RandomizedSchedulingRLEnv,
    SchedulingRLEnv,
    build_env,
    rl_obs_dim,
)
from agent.dedication_agent import DedicationAgent, HOLD_ACTION
from agent.bc import (
    ExpertAnchorCallback,
    ExpertDataset,
    behavior_clone,
    collect_expert_dataset,
)
from agent.kpi_eval import KPIEvalCallback, dedication_baseline_kpi
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


def _linear_schedule(initial: float) -> Callable[[float], float]:
    """progress_remaining(1→0)에 비례해 initial→0으로 선형 감쇠."""
    def func(progress_remaining: float) -> float:
        return float(progress_remaining) * initial
    return func


def align_datasets_with_inference(datasets: List[dict]) -> List[dict]:
    """학습용 데이터셋에 추론과 동일한 시뮬레이터 플래그를 채워 넣는다.

    추론(inference/runner.py)은 기본적으로
      termination_mode = "current_wip_assigned", enable_wip_inflow = False
    로 도는데, 학습은 simulator 기본값("all_wip", inflow=True)으로 돌고 있었다.
    학습한 MDP와 평가받는 MDP가 다르면(train/serve skew) 학습 지표는 좋아도
    벤치마크가 안 따라온다. 데이터셋이 값을 명시한 경우엔 그 값을 존중한다.
    """
    aligned: List[dict] = []
    for data in datasets:
        copy = dict(data)
        copy.setdefault("termination_mode", "current_wip_assigned")
        copy.setdefault("enable_wip_inflow", False)
        aligned.append(copy)
    return aligned


def _collect_expert_transitions(
    data: dict,
    env_cls: type,
    max_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DedicationAgent 시연 1 에피소드 (obs, action, action_mask).

    `agent.bc.collect_expert_dataset()`의 단일-에피소드·무교란 래퍼 — 기존
    호출부·테스트 호환용으로 남겨둔다. 실제 학습 경로는 여러 에피소드와
    ε 교란(DAgger식 상태분포 확장)을 쓰는 `collect_expert_dataset()`을 직접
    호출한다.
    """
    dataset = collect_expert_dataset(
        [data], env_cls,
        episodes_per_dataset=1, noise_eps=0.0, max_steps=max_steps,
    )
    return dataset.obs, dataset.actions, dataset.masks


def _behavior_clone_pretrain(
    model: MaskablePPO,
    datasets: List[dict],
    env_cls: type,
    epochs: int,
    lr: float,
    verbose: int = 1,
) -> None:
    """DedicationAgent 시연으로 PPO 정책을 지도학습 워밍스타트 (호환 래퍼).

    새 구현은 `agent.bc`에 있다 — 여기서는 CONFIG의 BC 설정을 그대로 써서
    데이터 수집 → 클로닝까지 한 번에 수행한다.
    """
    cfg = CONFIG.rl
    log = (lambda m: print(m)) if verbose else (lambda _m: None)
    dataset = collect_expert_dataset(
        datasets, env_cls,
        episodes_per_dataset=cfg.bc_episodes_per_dataset,
        noise_eps=cfg.bc_noise_eps,
        gamma=cfg.gamma,
        seed=cfg.seed or 0,
        expert_candidates=list(cfg.bc_expert_candidates),
        log=log,
    )
    behavior_clone(
        model, dataset,
        epochs=epochs, lr=lr,
        batch_size=cfg.bc_batch_size,
        entropy_coef=cfg.bc_entropy_coef,
        value_coef=cfg.bc_value_coef,
        val_fraction=cfg.bc_val_fraction,
        early_stop_patience=cfg.bc_early_stop_patience,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed or 0,
        log=log,
    )


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
    """현재 env obs 차원과 맞는 모델 로드 (없으면 예외).

    RL 모델은 SchedulingRLEnv 관측(= 기본 obs + 결정 컨텍스트 채널)을 쓰므로
    `rl_obs_dim()`을 기준으로 비교한다. `CONFIG.env.rl_extra_obs=False`면
    이전과 동일한 `compute_obs_dim()` 값이 된다.
    """
    expected = rl_obs_dim()
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
            extra_dim=expected - compute_obs_dim(),
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
        eval_datasets: Optional[List[dict]] = None,
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
            eval_datasets (list[dict], optional): KPI 평가·Dedication 기준선
                측정에 쓸 데이터셋. 미지정이면 학습 데이터셋 전체를 쓴다.
                학습 풀이 큰데(도메인 랜덤화) 평가는 대표 시나리오 몇 개로만
                하고 싶을 때 지정한다 — 평가는 매 eval_freq마다 전 데이터셋을
                굴리므로 풀이 크면 그대로 비용이 된다.
        Output:
            self (체이닝 가능)
        """
        cfg = CONFIG.rl
        model_dir = model_dir_for(fac_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        datasets: List[dict] = env_data if isinstance(env_data, list) else [env_data]
        eval_sets: List[dict] = list(eval_datasets) if eval_datasets else datasets
        if cfg.align_train_env_with_inference:
            datasets = align_datasets_with_inference(datasets)
            eval_sets = align_datasets_with_inference(eval_sets)
        n_envs = max(cfg.n_envs, 1)

        def log(message: str) -> None:
            if verbose:
                print(message, flush=True)
            if progress_state is not None:
                progress_state.add_log(message)

        def make_env(data: dict):
            def _init():
                env = ActionMasker(
                    build_env(env_cls, data),
                    _mask_fn,
                )
                return Monitor(env)
            return _init

        def make_pool_env(pool: List[dict], pool_seed: int):
            def _init():
                env = ActionMasker(
                    build_env(
                        RandomizedSchedulingRLEnv, pool, pool_seed=pool_seed,
                    ),
                    _mask_fn,
                )
                return Monitor(env)
            return _init

        # 데이터셋 샘플링 전략 (CONFIG.rl.dataset_sampling 주석 참고)
        use_pool = (
            cfg.dataset_sampling == "random_reset"
            and env_cls is SchedulingRLEnv
            and len(datasets) > cfg.dataset_sampling_min_pool
        )
        if use_pool:
            n_pool_envs = max(cfg.dataset_pool_envs, 1)
            train_fns = [
                make_pool_env(datasets, (cfg.seed or 0) * 1000 + i)
                for i in range(n_pool_envs)
                for _ in range(n_envs)
            ]
            total_envs = n_pool_envs * n_envs
            log(
                f"데이터셋 샘플링: random_reset — 시나리오 풀 {len(datasets)}개를 "
                f"env {total_envs}개가 에피소드마다 무작위로 돈다"
            )
        else:
            # n_envs > 1 이면 같은 데이터를 n_envs 개 프로세스에서 병렬 롤아웃
            # 기간이 여러 개면 기간 × n_envs 조합으로 확장
            train_fns = [make_env(d) for d in datasets for _ in range(n_envs)]
            total_envs = len(datasets) * n_envs

        if n_envs > 1:
            train_env = SubprocVecEnv(train_fns, start_method="fork")
        else:
            train_env = DummyVecEnv(train_fns)

        # 보상 정규화: step reward가 ±20이고 에피소드가 수백~수천 스텝이라
        # 할인 return이 수백 규모가 된다. 정규화 없이는 value loss가 폭주해
        # explained_variance≈0 → advantage가 잡음이 되고 학습 곡선이 x축과
        # 평행해진다. 관측은 이미 [0,1]이라 norm_obs는 끈다(추론 시 통계
        # 없이도 동일 동작).
        if cfg.normalize_reward:
            train_env = VecNormalize(
                train_env, norm_obs=False, norm_reward=True, gamma=cfg.gamma,
            )
        eval_env = DummyVecEnv([make_env(eval_sets[0])])

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

        if use_episode_budget and progress_state is None:
            from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
            callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=n_episodes))

        # ── best 모델 선택: KPI 기준(기본) 또는 기존 보상 기준 ────────────────
        kpi_callback: Optional[KPIEvalCallback] = None
        if cfg.kpi_eval_enabled:
            baseline = None
            if cfg.kpi_baseline_enabled:
                baseline = dedication_baseline_kpi(env_cls, eval_sets)
                log(
                    "[kpi] Dedication 기준선 — 생산 "
                    f"{baseline.produced}/{baseline.producible} · 전환 "
                    f"{baseline.conversions} · 점수 "
                    f"{baseline.score(cfg.kpi_conversion_weight):.1f}"
                )
            kpi_callback = KPIEvalCallback(
                env_cls, eval_sets,
                eval_freq=cfg.eval_freq,
                best_model_save_path=str(model_dir / "best"),
                history_path=str(model_dir / "logs" / "kpi_evaluations.json"),
                conv_weight=cfg.kpi_conversion_weight,
                baseline=baseline,
                state=progress_state,
                log=log,
            )
            callbacks.append(kpi_callback)
        elif progress_state is not None:
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
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(model_dir / "best"),
                    log_path=str(model_dir / "logs"),
                    eval_freq=cfg.eval_freq,
                    deterministic=True,
                    verbose=0,
                )
            )

        if progress_state is None:
            callbacks.append(
                CheckpointCallback(
                    save_freq=cfg.eval_freq,
                    save_path=str(model_dir / "checkpoints"),
                    name_prefix=cfg.model_name,
                    verbose=0,
                )
            )

        # 롤아웃 버퍼(n_steps × total_envs)가 커지므로 batch_size도 비례 확장
        effective_batch = cfg.batch_size * max(total_envs, 1)
        # batch_size는 rollout buffer(n_steps × total_envs)의 약수여야 함
        rollout_size = cfg.n_steps * total_envs
        effective_batch = min(effective_batch, rollout_size)
        while effective_batch > 1 and rollout_size % effective_batch != 0:
            effective_batch -= 1

        learning_rate = (
            _linear_schedule(cfg.learning_rate)
            if cfg.lr_schedule == "linear"
            else cfg.learning_rate
        )
        self.model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=cfg.n_steps,
            batch_size=effective_batch,
            n_epochs=cfg.n_epochs_stable,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            target_kl=cfg.target_kl,
            ent_coef=cfg.ent_coef,
            policy_kwargs={"net_arch": list(cfg.policy_net_arch)},
            seed=cfg.seed,
            verbose=verbose,
            device=cfg.device,
        )

        if total_envs > 1:
            log(
                f"VecEnv {total_envs}개 "
                f"({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'}, "
                f"시나리오 {len(datasets)}개, sampling="
                f"{'random_reset' if use_pool else 'per_env'}, n_envs={n_envs})"
            )
        budget_label = (
            f"n_episodes={n_episodes:,}"
            if use_episode_budget
            else f"total_timesteps={cfg.total_timesteps:,}"
        )
        log(
            f"하이퍼파라미터: {budget_label}, lr={cfg.learning_rate}({cfg.lr_schedule}), "
            f"n_steps={cfg.n_steps}, batch={effective_batch}(base={cfg.batch_size}×{total_envs}envs), "
            f"n_epochs={cfg.n_epochs_stable}, gamma={cfg.gamma}, target_kl={cfg.target_kl}, "
            f"net_arch={list(cfg.policy_net_arch)}, norm_reward={cfg.normalize_reward}, "
            f"eval_freq={cfg.eval_freq}, device={cfg.device}, n_envs={n_envs}"
        )

        # ── 모방학습 워밍스타트 + 전문가 앵커 ────────────────────────────────
        expert_data: Optional[ExpertDataset] = None
        needs_expert = cfg.bc_pretrain_epochs > 0 or (
            cfg.expert_anchor_coef > 0 and cfg.expert_anchor_steps > 0
        )
        if needs_expert:
            expert_data = collect_expert_dataset(
                datasets, env_cls,
                episodes_per_dataset=cfg.bc_episodes_per_dataset,
                noise_eps=cfg.bc_noise_eps,
                gamma=cfg.gamma,
                seed=cfg.seed or 0,
                expert_candidates=list(cfg.bc_expert_candidates),
                log=log,
            )

        if cfg.bc_pretrain_epochs > 0 and expert_data is not None:
            summary = behavior_clone(
                self.model, expert_data,
                epochs=cfg.bc_pretrain_epochs,
                lr=cfg.bc_pretrain_lr,
                batch_size=cfg.bc_batch_size,
                entropy_coef=cfg.bc_entropy_coef,
                value_coef=cfg.bc_value_coef,
                val_fraction=cfg.bc_val_fraction,
                early_stop_patience=cfg.bc_early_stop_patience,
                max_grad_norm=cfg.max_grad_norm,
                seed=cfg.seed or 0,
                log=log,
            )
            if not summary.get("skipped"):
                nll = summary.get("train_nll")
                ent = summary.get("entropy")
                log(
                    f"[bc] 워밍스타트 완료 — {summary['epochs_run']} epoch, "
                    f"train_nll={nll:.4f}" if nll is not None else
                    f"[bc] 워밍스타트 완료 — {summary['epochs_run']} epoch"
                )
                if ent is not None:
                    log(f"[bc] 최종 정책 엔트로피 {ent:.3f} (0에 가까우면 결정적 붕괴)")

        if (
            expert_data is not None
            and cfg.expert_anchor_coef > 0
            and cfg.expert_anchor_steps > 0
        ):
            anchor = ExpertAnchorCallback(
                expert_data,
                coef_start=cfg.expert_anchor_coef,
                coef_final=cfg.expert_anchor_final,
                decay_fraction=cfg.expert_anchor_decay_fraction,
                steps_per_rollout=cfg.expert_anchor_steps,
                batch_size=cfg.expert_anchor_batch,
                entropy_coef=cfg.bc_entropy_coef,
                lr=cfg.bc_pretrain_lr,
                max_grad_norm=cfg.max_grad_norm,
                seed=cfg.seed or 0,
            )
            if anchor.active:
                callbacks.append(anchor)
                log(
                    f"전문가 앵커 활성 — coef {cfg.expert_anchor_coef}→"
                    f"{cfg.expert_anchor_final} (구간 {cfg.expert_anchor_decay_fraction}), "
                    f"롤아웃당 {cfg.expert_anchor_steps} 스텝"
                )

        # KPI 콜백을 마지막에 두면 앵커 적용 후 상태로 채점된다.
        if kpi_callback is not None and kpi_callback in callbacks:
            callbacks.remove(kpi_callback)
            callbacks.append(kpi_callback)

        self.model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            progress_bar=(verbose > 0 and progress_state is None),
        )

        if restore_best:
            best_path = model_dir / "best" / "best_model.zip"
            if best_path.exists():
                # eval 시점마다 저장된 최고 체크포인트로 되돌린다. PPO는 이미
                # 좋은 정책을 찾은 뒤에도 계속 학습하며 오히려 나빠지는 경우가
                # 있어서, 학습 종료 시점 모델을 그대로 쓰지 않는다.
                # kpi_eval_enabled=True면 '최고'의 기준이 shaping 보상이 아니라
                # 실제 벤치마크 KPI(생산량−전환수)다.
                log(f"[agent] 학습 종료 시점보다 나은 체크포인트로 복원 ← {best_path}")
                self.model = MaskablePPO.load(str(best_path), device=cfg.device)

        if kpi_callback is not None:
            self.kpi_history = kpi_callback.history
            best = kpi_callback.best_kpi
            if best is not None:
                log(
                    f"[kpi] 채택 모델 — step {kpi_callback.best_timestep:,} · 생산 "
                    f"{best.produced}/{best.producible} · 전환 {best.conversions} · "
                    f"점수 {kpi_callback.best_score:.1f}"
                )
            if kpi_callback._baseline is not None and best is not None:
                delta = kpi_callback.best_score - kpi_callback._baseline.score(
                    cfg.kpi_conversion_weight,
                )
                if delta < 0:
                    log(
                        f"[경고] 최종 모델이 Dedication 기준선보다 {abs(delta):.1f}점 "
                        "낮습니다 — 학습 예산(total_timesteps)이나 보상 설정을 "
                        "재검토하세요.",
                    )
                else:
                    log(f"[kpi] Dedication 기준선 대비 {delta:+.1f}점")
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
