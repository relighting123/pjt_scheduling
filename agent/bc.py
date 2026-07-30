"""
agent/bc.py – 휴리스틱 시연 기반 모방학습(Behavior Cloning) 워밍스타트

기존 `rl_agent._behavior_clone_pretrain()`의 한계와 이 모듈의 대응
--------------------------------------------------------------------
① **표본 부족**: 데이터셋당 결정적(deterministic) 전문가 에피소드 1개만 모아서
   SYM_5x5 기준 수십 스텝이 전부였다. 936차원 관측을 쓰는 정책을 수십 표본으로
   맞추면 일반화가 안 되고, PPO가 조금만 다른 상태에 가면 즉시 무너진다.
   → `collect_expert_dataset()`이 에피소드를 여러 번 굴리되 매 스텝 ε 확률로
   **무작위 feasible 행동으로 교란**하고, 라벨은 항상 그 상태에서 전문가가
   골랐을 행동으로 부여한다(DAgger식 상태분포 확장). 전문가 궤적 위뿐 아니라
   "전문가에서 살짝 벗어난 상태에서 어떻게 복귀하는지"까지 배운다.

② **크리틱 미학습**: 정책(actor)만 지도학습하고 value head는 무작위로 남겼다.
   PPO의 첫 업데이트에서 advantage = return − V(s) 가 통째로 잡음이 되어,
   애써 복제한 정책을 방향 없는 그래디언트로 밀어버린다. 실제로 "BC 직후엔
   좋은데 RL을 돌리면 나빠진다"의 가장 큰 원인.
   → 시연의 할인 return-to-go를 타깃으로 **value head도 같이 회귀**한다.

③ **결정적 붕괴**: full-batch NLL을 오래 돌리면 로그확률이 0에 수렴하며
   엔트로피≈0이 되고, 그 뒤 PPO는 탐색도 학습도 못 한다(학습 곡선이 평평).
   → 미니배치 + 엔트로피 정규화 + 검증 NLL 조기종료로 "적당히 닮은" 지점에서
   멈춘다.

④ **PPO 중 파국적 망각**: 워밍스타트만으로는 RL 잡음에 정책이 서서히(또는
   급격히) 전문가에서 이탈한다.
   → `ExpertAnchorCallback`이 롤아웃마다 BC 그래디언트를 소량 섞고, 그 비중을
   학습 진행에 따라 0으로 감쇠시킨다. 초반엔 전담 정책을 붙잡고, 후반엔 RL이
   자유롭게 그 위를 개선한다.

⑤ **전담이 항상 최선은 아님**: 시연자를 `DedicationAgent`로 고정하면 정책의
   출발점 상한이 전담 휴리스틱이 된다. 그런데 처리시간이 이질적인 시나리오
   (LOAD_stmix)에서는 Earliest-ST 계열이 전담의 2배 점수를 냈다.
   → `agent/experts.py`의 후보들을 시나리오마다 실제로 굴려보고 KPI가 가장
   좋은 전문가를 그 시나리오의 시연자로 삼는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from agent.experts import (
    DEFAULT_EXPERT_CANDIDATES,
    ExpertPolicy,
    make_expert,
    select_best_expert,
)
from env.scheduling_rl_env import build_env

LogFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


@dataclass
class ExpertDataset:
    """전문가 시연 텐서 묶음 (모두 CPU numpy)."""

    obs: np.ndarray            # (N, obs_dim) float32
    actions: np.ndarray        # (N, 2) int64  [bucket, size_level]
    masks: np.ndarray          # (N, n_bucket + L) bool
    returns: np.ndarray        # (N,) float32 — 할인 return-to-go (크리틱 타깃)
    n_episodes: int = 0

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


def _rollout_expert_episode(
    data: dict,
    env_cls: type,
    rng: np.random.Generator,
    max_steps: int,
    noise_eps: float,
    gamma: float,
    expert_name: str = "dedication",
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[np.ndarray], List[float], dict]:
    """전문가 1 에피소드(ε 교란 포함) 수집.

    반환되는 action 라벨은 **교란된 실제 행동이 아니라 전문가가 골랐을 행동**
    이다 — 교란은 상태분포를 넓히는 용도이고 학습 타깃은 언제나 전문가다.
    """
    env = build_env(env_cls, data)
    expert: ExpertPolicy = make_expert(expert_name, data)

    obs_list: List[np.ndarray] = []
    action_list: List[Tuple[int, int]] = []
    mask_list: List[np.ndarray] = []
    reward_list: List[float] = []

    obs, _ = env.reset()
    n_bucket = env._n_bucket
    info: dict = {}

    for _ in range(max_steps):
        mask = env.action_masks()
        eqp_id = env.sim.current_idle_eqp()

        expert_bucket = expert.resolve(env.sim, eqp_id, mask, n_bucket)
        expert_level = expert.size_level(mask, n_bucket)

        obs_list.append(obs)
        action_list.append((expert_bucket, expert_level))
        mask_list.append(mask)

        # ── ε 교란: 실제로 밟는 행동만 흔든다(라벨은 전문가 그대로)
        play_bucket, play_level = expert_bucket, expert_level
        if noise_eps > 0 and rng.random() < noise_eps:
            feasible = np.flatnonzero(mask[:n_bucket])
            if feasible.size > 1:
                play_bucket = int(rng.choice(feasible))
            allowed = np.flatnonzero(mask[n_bucket:])
            if allowed.size > 1:
                play_level = int(rng.choice(allowed))

        obs, reward, terminated, truncated, info = env.step(
            np.array([play_bucket, play_level], dtype=np.int64)
        )
        reward_list.append(float(reward))
        if terminated or truncated:
            break

    # 할인 return-to-go (크리틱 회귀 타깃)
    returns: List[float] = []
    acc = 0.0
    for r in reversed(reward_list):
        acc = r + gamma * acc
        returns.append(acc)
    returns.reverse()

    return obs_list, action_list, mask_list, returns, info


def collect_expert_dataset(
    datasets: List[dict],
    env_cls: type,
    *,
    episodes_per_dataset: int = 12,
    noise_eps: float = 0.15,
    gamma: float = 0.99,
    max_steps: int = 4000,
    seed: int = 0,
    expert_candidates: Optional[List[str]] = None,
    log: LogFn = _noop,
) -> ExpertDataset:
    """전문가 시연을 여러 에피소드 모아 하나의 데이터셋으로.

    expert_candidates: 후보가 2개 이상이면 **시나리오마다 후보를 실제로 굴려
        KPI가 가장 좋은 전문가**를 그 시나리오의 시연자로 쓴다. 전담이 항상
        최선은 아니기 때문 — 처리시간이 이질적인 시나리오에서는 Earliest-ST
        계열이 전담의 2배 점수를 내기도 한다. 후보가 1개면 그 전문가로 고정.
        None이면 `agent.experts.DEFAULT_EXPERT_CANDIDATES`.

    전문가는 결정적이므로 ε 교란(noise_eps)이 0이면 에피소드를 몇 번 굴려도
    같은 궤적이 나온다 → 그 경우 데이터셋당 1회만 수집한다.
    """
    rng = np.random.default_rng(seed)
    candidates = list(expert_candidates or DEFAULT_EXPERT_CANDIDATES)
    obs_parts, act_parts, mask_parts, ret_parts = [], [], [], []
    n_eps = 0
    chosen: dict = {}

    for di, data in enumerate(datasets):
        if len(candidates) > 1:
            expert_name, scores = select_best_expert(
                data,
                env_factory=lambda d: build_env(env_cls, d),
                candidates=candidates,
                max_steps=max_steps,
            )
            chosen[expert_name] = chosen.get(expert_name, 0) + 1
        else:
            expert_name = candidates[0]
            scores = None

        # 교란이 없으면 반복해봐야 동일 궤적 — 1회로 충분(불필요한 시뮬 비용 제거)
        reps = 1 if noise_eps <= 0 else max(int(episodes_per_dataset), 1)
        for rep in range(reps):
            # 첫 에피소드는 순수 전문가 궤적(교란 0)을 확보한다 — 정답 궤적이
            # 데이터에 반드시 포함되도록.
            eps = 0.0 if rep == 0 else noise_eps
            o, a, m, r, _info = _rollout_expert_episode(
                data, env_cls, rng, max_steps, eps, gamma, expert_name,
            )
            if not o:
                continue
            obs_parts.append(np.asarray(o, dtype=np.float32))
            act_parts.append(np.asarray(a, dtype=np.int64))
            mask_parts.append(np.asarray(m, dtype=bool))
            ret_parts.append(np.asarray(r, dtype=np.float32))
            n_eps += 1
        detail = f" (expert={expert_name})" if scores is not None else ""
        log(
            f"[bc] 시연 수집 dataset {di + 1}/{len(datasets)}{detail} — "
            f"누적 {sum(len(x) for x in obs_parts):,} 스텝"
        )

    if chosen:
        summary = ", ".join(f"{k} {v}개" for k, v in sorted(chosen.items()))
        log(f"[bc] 시나리오별 최적 시연자 선택 결과 — {summary}")

    if not obs_parts:
        return ExpertDataset(
            np.zeros((0, 0), np.float32), np.zeros((0, 2), np.int64),
            np.zeros((0, 0), bool), np.zeros((0,), np.float32), 0,
        )

    return ExpertDataset(
        obs=np.concatenate(obs_parts),
        actions=np.concatenate(act_parts),
        masks=np.concatenate(mask_parts),
        returns=np.concatenate(ret_parts),
        n_episodes=n_eps,
    )


class _TorchExpertData:
    """ExpertDataset을 모델 device 위 텐서로 올려두고 미니배치를 뽑는다."""

    def __init__(self, dataset: ExpertDataset, device) -> None:
        self.obs = torch.as_tensor(dataset.obs, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(dataset.actions, dtype=torch.long, device=device)
        self.masks = torch.as_tensor(dataset.masks, dtype=torch.bool, device=device)
        returns = torch.as_tensor(dataset.returns, dtype=torch.float32, device=device)
        # 크리틱 타깃 정규화 — return 스케일(수백 규모)을 그대로 회귀시키면
        # value loss가 정책 손실을 압도한다. 정규화한 타깃으로 "상대적 가치
        # 순서"만 미리 익히게 하고, 절대 스케일은 PPO가 맞춘다.
        self.return_mean = float(returns.mean()) if returns.numel() else 0.0
        self.return_std = float(returns.std()) if returns.numel() > 1 else 1.0
        if self.return_std < 1e-6:
            self.return_std = 1.0
        self.returns = (returns - self.return_mean) / self.return_std
        self.n = self.obs.shape[0]

    def batches(self, batch_size: int, generator: torch.Generator):
        perm = torch.randperm(self.n, generator=generator, device=self.obs.device)
        for start in range(0, self.n, batch_size):
            idx = perm[start:start + batch_size]
            yield self.obs[idx], self.actions[idx], self.masks[idx], self.returns[idx]

    def sample(self, batch_size: int, generator: torch.Generator):
        size = min(batch_size, self.n)
        idx = torch.randint(0, self.n, (size,), generator=generator, device=self.obs.device)
        return self.obs[idx], self.actions[idx], self.masks[idx], self.returns[idx]


def _bc_losses(policy, obs, actions, masks, returns, entropy_coef: float, value_coef: float):
    values, log_prob, entropy = policy.evaluate_actions(obs, actions, masks)
    nll = -log_prob.mean()
    ent = entropy.mean()
    loss = nll - entropy_coef * ent
    v_loss = torch.zeros((), device=nll.device)
    if value_coef > 0:
        v_loss = torch.nn.functional.mse_loss(values.flatten(), returns)
        loss = loss + value_coef * v_loss
    return loss, nll, ent, v_loss


def behavior_clone(
    model,
    dataset: ExpertDataset,
    *,
    epochs: int = 60,
    lr: float = 3e-4,
    batch_size: int = 256,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    val_fraction: float = 0.15,
    early_stop_patience: int = 8,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    log: LogFn = _noop,
) -> dict:
    """전문가 시연으로 MaskablePPO 정책+크리틱을 지도학습.

    Returns: 학습 요약 dict (best_val_nll, epochs_run, final_entropy 등)
    """
    if dataset.is_empty or epochs <= 0:
        log("[bc] 시연 데이터 없음 — 워밍스타트 생략")
        return {"skipped": True, "n_samples": len(dataset)}

    device = model.device
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction) if n >= 20 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def subset(idx: np.ndarray) -> ExpertDataset:
        return ExpertDataset(
            dataset.obs[idx], dataset.actions[idx],
            dataset.masks[idx], dataset.returns[idx], dataset.n_episodes,
        )

    train_t = _TorchExpertData(subset(train_idx), device)
    val_t = _TorchExpertData(subset(val_idx), device) if n_val > 0 else None

    policy = model.policy
    policy.set_training_mode(True)
    generator = torch.Generator(device=train_t.obs.device)
    generator.manual_seed(int(seed))

    # BC 전용 옵티마이저 — PPO 옵티마이저의 Adam 모멘트를 BC 단계 통계로
    # 오염시키지 않는다(오염되면 PPO 첫 업데이트가 과도하게 튄다).
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    patience = 0
    epochs_run = 0
    last: dict = {}

    log(f"[bc] 워밍스타트 시작 — 시연 {n:,}건(에피소드 {dataset.n_episodes}), "
        f"train {len(train_idx):,} / val {n_val:,}, epochs={epochs}, lr={lr}, batch={batch_size}")

    for epoch in range(epochs):
        epochs_run = epoch + 1
        sums = {"loss": 0.0, "nll": 0.0, "ent": 0.0, "v": 0.0}
        n_batch = 0
        for obs_b, act_b, mask_b, ret_b in train_t.batches(batch_size, generator):
            loss, nll, ent, v_loss = _bc_losses(
                policy, obs_b, act_b, mask_b, ret_b, entropy_coef, value_coef,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            sums["loss"] += float(loss.detach())
            sums["nll"] += float(nll.detach())
            sums["ent"] += float(ent.detach())
            sums["v"] += float(v_loss.detach())
            n_batch += 1
        if n_batch == 0:
            break
        last = {k: v / n_batch for k, v in sums.items()}

        # 검증 NLL 기준 조기종료 — "완전 수렴 = 결정적 붕괴"를 피하는 자동 장치
        if val_t is not None:
            policy.set_training_mode(False)
            with torch.no_grad():
                _l, val_nll, _e, _v = _bc_losses(
                    policy, val_t.obs, val_t.actions, val_t.masks, val_t.returns,
                    entropy_coef, value_coef,
                )
            policy.set_training_mode(True)
            val_nll = float(val_nll)
            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                patience = 0
            else:
                patience += 1
        else:
            val_nll = last["nll"]

        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(epochs // 6, 1) == 0:
            log(f"[bc] epoch {epoch + 1}/{epochs} nll={last['nll']:.4f} "
                f"val_nll={val_nll:.4f} ent={last['ent']:.3f} v_loss={last['v']:.4f}")

        if val_t is not None and patience >= early_stop_patience:
            log(f"[bc] 검증 NLL {early_stop_patience} epoch 개선 없음 → 조기종료 "
                f"(best_val_nll={best_val:.4f})")
            break

    if best_state is not None:
        policy.load_state_dict(best_state)
        log(f"[bc] 검증 최적 가중치로 복원 (val_nll={best_val:.4f})")

    policy.set_training_mode(False)
    return {
        "skipped": False,
        "n_samples": n,
        "n_episodes": dataset.n_episodes,
        "epochs_run": epochs_run,
        "best_val_nll": None if best_val == float("inf") else best_val,
        "train_nll": last.get("nll"),
        "entropy": last.get("ent"),
        "value_loss": last.get("v"),
    }


class ExpertAnchorCallback(BaseCallback):
    """PPO 롤아웃마다 전문가 시연에 BC 그래디언트를 소량 섞는 앵커.

    순수 PPO는 보상 잡음·advantage 추정 오차로 워밍스타트된 정책에서 서서히
    이탈한다(catastrophic forgetting). 매 롤아웃 뒤 몇 스텝의 BC 손실을
    적용하면 "전문가에서 크게 벗어나지 말 것"이라는 부드러운 제약이 되고,
    계수를 학습 진행에 따라 start→final로 감쇠시키면 후반에는 RL이 그 위를
    자유롭게 개선할 수 있다.

    coef=0이거나 시연이 없으면 아무 것도 하지 않는다.
    """

    def __init__(
        self,
        dataset: ExpertDataset,
        *,
        coef_start: float = 1.0,
        coef_final: float = 0.0,
        decay_fraction: float = 0.6,
        steps_per_rollout: int = 4,
        batch_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.0,
        lr: float = 1e-4,
        max_grad_norm: float = 0.5,
        seed: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._dataset = dataset
        self._coef_start = coef_start
        self._coef_final = coef_final
        self._decay_fraction = min(max(decay_fraction, 1e-6), 1.0)
        self._steps = max(int(steps_per_rollout), 0)
        self._batch = max(int(batch_size), 1)
        self._entropy_coef = entropy_coef
        # 앵커 단계에서 크리틱까지 건드리면 PPO의 value 학습과 타깃이 충돌한다
        # (PPO는 정규화된 실제 return, 앵커는 시연 return) → 기본 0.
        self._value_coef = value_coef
        self._lr = lr
        self._max_grad_norm = max_grad_norm
        self._seed = seed
        self._data: Optional[_TorchExpertData] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._generator: Optional[torch.Generator] = None
        self.last_coef: float = coef_start
        self.last_nll: Optional[float] = None

    @property
    def active(self) -> bool:
        return (
            self._steps > 0
            and self._coef_start > 0
            and self._dataset is not None
            and not self._dataset.is_empty
        )

    def _on_training_start(self) -> None:
        if not self.active:
            return
        self._data = _TorchExpertData(self._dataset, self.model.device)
        self._optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=self._lr)
        self._generator = torch.Generator(device=self._data.obs.device)
        self._generator.manual_seed(int(self._seed))

    def _current_coef(self) -> float:
        remaining = getattr(self.model, "_current_progress_remaining", 1.0)
        remaining = min(max(float(remaining), 0.0), 1.0)
        elapsed = 1.0 - remaining
        t = min(elapsed / self._decay_fraction, 1.0)
        return self._coef_start + (self._coef_final - self._coef_start) * t

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        """앵커 그래디언트는 PPO의 train() **직후**(= 다음 롤아웃 시작 시점)에
        적용한다.

        SB3 순서는 `on_rollout_start → collect_rollouts → on_rollout_end →
        train()` 이다. on_rollout_end에서 정책을 건드리면 방금 수집한 롤아웃의
        저장된 log_prob과 정책이 어긋나(ratio≠1로 시작) 첫 epoch부터
        clip/target_kl에 걸려 PPO 업데이트가 통째로 잘려나간다. 반대로
        on_rollout_start에서 적용하면 그 뒤에 수집되는 롤아웃이 앵커 적용 후
        정책으로 만들어져 on-policy 가정이 유지된다.
        """
        if not self.active or self._data is None:
            return
        coef = self._current_coef()
        self.last_coef = coef
        if coef <= 0:
            return

        policy = self.model.policy
        policy.set_training_mode(True)
        total_nll = 0.0
        for _ in range(self._steps):
            obs_b, act_b, mask_b, ret_b = self._data.sample(self._batch, self._generator)
            loss, nll, _ent, _v = _bc_losses(
                policy, obs_b, act_b, mask_b, ret_b,
                self._entropy_coef, self._value_coef,
            )
            self._optimizer.zero_grad()
            (coef * loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self._max_grad_norm)
            self._optimizer.step()
            total_nll += float(nll.detach())
        policy.set_training_mode(False)
        self.last_nll = total_nll / max(self._steps, 1)
        self.logger.record("bc_anchor/coef", coef)
        self.logger.record("bc_anchor/nll", self.last_nll)
