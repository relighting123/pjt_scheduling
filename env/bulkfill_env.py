"""
env/bulkfill_env.py – 벌크 점유(Bulk-Fill) MDP 환경 (Phase 1 스캐폴딩)

기존 SchedulingEnv가 idle EQP마다 1 carrier를 배정하는 것과 달리,
정책이 (PPK/OPER 버킷, 블록 크기 레벨)을 함께 선택하면 해당 장비가
같은 setup으로 N carrier를 연속(블록) 처리하도록 '커밋'한다.

시뮬레이터 무결성(이산 사건: assign 후 즉시 busy)을 해치지 않기 위해
블록은 env 레이어에서 'masked replay'로 실현한다:
  - 블록 시작 결정 시 N을 산출하고 1 carrier 배정 + remaining=N-1 저장
  - 같은 장비의 다음 idle 결정들은 같은 버킷으로 강제(마스크) 재생
  - WIP/계획/tool 잔여 소진 또는 remaining=0이면 블록 종료

블록 크기 상한은 simulator.bulk_block_size()가
min(takt 예산, 가용 WIP, 잔여 계획, tool '잔여'(추가 가용))로 묶는다.

StableBaselines3-Contrib MaskablePPO(MultiDiscrete) 호환:
action_masks()는 [버킷 마스크(O*P) | 크기 마스크(L)] 평탄 연결을 반환한다.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from simulation.simulator import SchedulingSimulator
from env.scheduling_env import compute_obs_dim

# 블록 크기 레벨 수 (action 두 번째 차원). level→takt 예산 분율.
BULK_SIZE_LEVELS = 4


class BulkFillEnv(gym.Env):
    """벌크 점유 강화학습 환경 — action=MultiDiscrete([O*P, L])."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_data: dict,
        render_mode: Optional[str] = None,
        record_history: bool = True,
        record_event_log: bool = True,
        truncate_on_time: bool = True,
        size_levels: int = BULK_SIZE_LEVELS,
    ):
        super().__init__()
        self._env_data = env_data
        self._record_history = record_history
        self._record_event_log = record_event_log
        self._truncate_on_time = truncate_on_time
        self._L = max(int(size_levels), 1)

        env_cfg = CONFIG.env
        self._O = env_cfg.max_oper_count
        self._P = env_cfg.max_prod_count
        self._n_bucket = self._O * self._P

        obs_dim = compute_obs_dim()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self._n_bucket, self._L])

        self.sim: Optional[SchedulingSimulator] = None
        self.render_mode = render_mode
        self._total_reward = 0.0
        self._episode_steps = 0
        self._max_episode_steps = 0
        # 블록 상태: eqp_id -> [flat_bucket, remaining]
        self._block: dict = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.sim = SchedulingSimulator(
            self._env_data, CONFIG.reward,
            record_history=self._record_history,
            record_event_log=self._record_event_log,
        )
        self._total_reward = 0.0
        self._episode_steps = 0
        sim_end = int(self._env_data.get("sim_end_minutes", CONFIG.env.hard_horizon_minutes))
        self._max_episode_steps = sim_end + 500
        self._block = {}
        return self.sim.get_observation(), {}

    def _ensure_decision_eqp(self) -> Optional[str]:
        if self.sim.current_idle_eqp() is not None:
            return self.sim.current_idle_eqp()
        while self.sim.get_idle_eqps() and self.sim.current_idle_eqp() is None:
            self.sim._select_same_time_next_eqp()
            if self.sim.current_idle_eqp() is not None:
                return self.sim.current_idle_eqp()
        if self.sim._has_pending_processing():
            self.sim._advance_to_next_decision()
            return self.sim.current_idle_eqp()
        return None

    # ── 블록 상태 ─────────────────────────────────────────────────────────────

    def _active_block_flat(self, eqp_id: str) -> Optional[int]:
        """진행 중 블록의 버킷 flat. 더 이상 feasible하지 않으면 종료하고 None."""
        blk = self._block.get(eqp_id)
        if not blk or blk[1] <= 0:
            if blk is not None:
                self._block.pop(eqp_id, None)
            return None
        flat = blk[0]
        if flat in self.sim.get_feasible_ppk_oper(eqp_id):
            return flat
        self._block.pop(eqp_id, None)
        return None

    # ── masks ─────────────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """MaskablePPO(MultiDiscrete)용 평탄 마스크 [버킷(O*P) | 크기(L)]."""
        bucket_mask = np.zeros(self._n_bucket, dtype=bool)
        size_mask = np.zeros(self._L, dtype=bool)

        if self.sim is None:
            bucket_mask[0] = True
            size_mask[0] = True
            return np.concatenate([bucket_mask, size_mask])

        eqp_id = self.sim.current_idle_eqp()
        if eqp_id is None:
            bucket_mask[0] = True
            size_mask[0] = True
            return np.concatenate([bucket_mask, size_mask])

        active = self._active_block_flat(eqp_id)
        if active is not None:
            # 블록 진행 중 → 같은 버킷 강제, 크기 레벨 0 고정
            bucket_mask[active] = True
            size_mask[0] = True
        else:
            for flat in self.sim.get_feasible_ppk_oper(eqp_id):
                if 0 <= flat < self._n_bucket:
                    bucket_mask[flat] = True
            if not bucket_mask.any():
                bucket_mask[0] = True
            size_mask[:] = True  # 모든 크기 레벨 허용 (clamp가 실제 유효성 보정)

        return np.concatenate([bucket_mask, size_mask])

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self, action: Union[int, np.ndarray, list],
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        bucket_idx = int(action_arr[0]) if action_arr.size > 0 else 0
        size_level = int(action_arr[1]) if action_arr.size > 1 else 0

        eqp_id = self._ensure_decision_eqp()
        reward = 0.0

        if eqp_id is not None:
            feasible = self.sim.get_feasible_ppk_oper(eqp_id)
            active = self._active_block_flat(eqp_id)

            if active is not None:
                flat: Optional[int] = active
            else:
                flat = bucket_idx % self._n_bucket
                if flat not in feasible:
                    flat = feasible[0] if feasible else None

            if flat is not None and self.sim.eqps[eqp_id].status == "idle":
                ppk, oper_id = self.sim.ppk_oper_from_flat(flat)
                reward = self.sim.assign_ppk_oper(eqp_id, ppk, oper_id)
                if active is None:
                    # 새 블록 시작 — 크기 산출 후 벌크 shaping + remaining 설정
                    if reward >= 0:
                        n = self.sim.bulk_block_size(
                            eqp_id, ppk, oper_id, size_level, self._L,
                        )
                        reward += self.sim.bulk_decision_shaping(
                            eqp_id, ppk, oper_id, n,
                        )
                        if n > 1:
                            self._block[eqp_id] = [flat, n - 1]
                else:
                    # 블록 진행 — remaining 감소
                    blk = self._block.get(eqp_id)
                    if blk:
                        blk[1] -= 1
                        if blk[1] <= 0:
                            self._block.pop(eqp_id, None)
                    if reward < 0:
                        self._block.pop(eqp_id, None)
            elif feasible:
                reward = -0.5
            else:
                self.sim._current_eqp = None
                if self.sim._has_pending_processing():
                    self.sim._advance_to_next_decision()
        elif not self.sim.is_done():
            if self.sim._has_pending_processing() or self.sim.get_idle_eqps():
                self.sim._advance_to_next_decision()

        self._episode_steps += 1
        terminated = self.sim.is_done()
        truncated = (not terminated) and (
            (self._truncate_on_time and self.sim.current_time >= self.sim.sim_end)
            or self._episode_steps >= self._max_episode_steps
        )

        if self._record_history:
            self.sim.save_history_step()
        else:
            self.sim.clear_step_assignment()

        clip = CONFIG.reward.reward_clip
        if clip and clip > 0:
            reward = float(np.clip(reward, -clip, clip))
        self._total_reward += reward

        obs = self.sim.get_observation()
        info = {
            "total_reward":  self._total_reward,
            "oper_switches": self.sim.stats["oper_switches"],
            "prod_switches": self.sim.stats["prod_switches"],
            "conversions":   self.sim.stats.get("conversions", 0),
            "idle_total":    self.sim.stats["idle_total"],
            "completed_qty": dict(self.sim.stats["completed_qty"]),
            "current_eqp":   eqp_id,
        }
        return obs, reward, terminated, truncated, info

    # ── accessors ──────────────────────────────────────────────────────────────

    def get_schedule(self) -> list:
        return self.sim.schedule if self.sim else []

    def get_history(self) -> list:
        return self.sim.history if self.sim else []
