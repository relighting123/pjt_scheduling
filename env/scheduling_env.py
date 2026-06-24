"""

env/scheduling_env.py – Gymnasium 커스텀 환경

SchedulingSimulator를 감싸 RL 표준 인터페이스(reset/step)를 제공합니다.

StableBaselines3 MaskablePPO와 호환됩니다.

"""

from typing import List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from simulation.simulator import SchedulingSimulator
from simulation.decision_log import build_step_decision_entry


def compute_obs_dim() -> int:
    """Global(6) + Bucket(O×P×K×F) + current EQP(6) + Context(4)"""

    O = CONFIG.env.max_oper_count
    P = CONFIG.env.max_prod_count
    K = CONFIG.env.max_model_count
    F = SchedulingSimulator.BUCKET_FEATURES
    return 6 + O * P * K * F + 6 + 4


class SchedulingEnv(gym.Env):
    """
    Scheduling 강화학습 환경

    - 관측: 전역 + PPK/OPER/MODEL bucket + 현재 idle EQP 상태
    - 행동: Discrete(O×P) – (PPK/OPER bucket). EQP·LOT은 규칙 자동 배정
  """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_data: dict,
        render_mode: Optional[str] = None,
        record_history: bool = True,
        record_decision_log: bool = False,
        max_episode_steps: Optional[int] = None,
        truncate_on_time: bool = True,
    ):
        super().__init__()
        self._env_data = env_data
        self._record_history = record_history
        self._record_decision_log = record_decision_log
        self._decision_log: List[dict] = []
        self._max_episode_steps_override = max_episode_steps
        self._max_episode_steps = 0
        self._episode_steps = 0
        self._truncate_on_time = truncate_on_time

        env_cfg = CONFIG.env
        O = env_cfg.max_oper_count
        P = env_cfg.max_prod_count

        obs_dim = compute_obs_dim()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(O * P)

        self.sim: Optional[SchedulingSimulator] = None
        self.render_mode = render_mode
        self._total_reward = 0.0
        self._O = O
        self._P = P

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.sim = SchedulingSimulator(
            self._env_data, CONFIG.reward, record_history=self._record_history,
        )
        self._total_reward = 0.0
        sim_end = int(self._env_data.get("sim_end_minutes", CONFIG.env.hard_horizon_minutes))
        self._max_episode_steps = (
            self._max_episode_steps_override
            if self._max_episode_steps_override is not None
            else sim_end + 500
        )
        self._episode_steps = 0
        self._decision_log = []
        obs = self.sim.get_observation()
        return obs, {}

    def _ensure_decision_eqp(self) -> Optional[str]:
        """idle 결정 호기가 없으면 시간 전진 또는 동시 idle 탐색."""
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

    def _resolve_ppk_oper(self, ppk_oper_idx: int, feasible: List[int]) -> Optional[int]:
        """invalid bucket → 현재 EQP feasible 중 보정."""
        if not feasible:
            return None
        flat = int(ppk_oper_idx) % (self._O * self._P)
        if flat in feasible:
            return flat
        return feasible[0]

    def step(self, action: Union[int, np.ndarray, list]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        ppk_oper_idx = int(action_arr[0]) if len(action_arr) > 0 else 0

        time_at_step_start = self.sim.current_time
        eqp_id = self._ensure_decision_eqp()
        time_advanced = self.sim.current_time != time_at_step_start
        schedule_before = len(self.sim.schedule)

        arrange_actual_before = self.sim.get_remaining_arrange_actual()
        arrange_abstract_before = self.sim.get_abstract_arrange()
        wip_waiting_before = self.sim.get_wip_waiting()

        reward = 0.0
        resolved_flat: Optional[int] = None
        if eqp_id is not None:
            feasible = self.sim.get_feasible_ppk_oper(eqp_id)
            resolved_flat = self._resolve_ppk_oper(ppk_oper_idx, feasible)
            if resolved_flat is not None and self.sim.eqps[eqp_id].status == "idle":
                ppk, oper_id = self.sim.ppk_oper_from_flat(resolved_flat)
                reward = self.sim.assign_ppk_oper(eqp_id, ppk, oper_id)
            elif feasible:
                reward = -0.5
            else:
                # tool cap 등으로 idle이지만 배정 불가 → 시간 전진
                self.sim._current_eqp = None
                if self.sim._has_pending_processing():
                    self.sim._advance_to_next_decision()
                    time_advanced = self.sim.current_time != time_at_step_start
        elif not self.sim.is_done():
            if self.sim._has_pending_processing() or self.sim.get_idle_eqps():
                self.sim._advance_to_next_decision()
                time_advanced = self.sim.current_time != time_at_step_start

        self._episode_steps += 1
        terminated = self.sim.is_done()
        truncated = (not terminated) and (
            (self._truncate_on_time and self.sim.current_time >= self.sim.sim_end)
            or self._episode_steps >= self._max_episode_steps
        )

        if self._record_decision_log:
            entry = build_step_decision_entry(
                step=self._episode_steps,
                sim_time=time_at_step_start,
                sim_time_after=self.sim.current_time,
                eqp_id=eqp_id,
                action_flat=ppk_oper_idx,
                resolved_flat=resolved_flat,
                reward=reward,
                sim=self.sim,
                terminated=terminated,
            )
            if len(self.sim.schedule) > schedule_before:
                entry["assigned_lot_id"] = self.sim.schedule[-1].get("LOT_ID")
                if entry["status"] not in ("assigned", "action_corrected"):
                    entry["status"] = "assigned"
            self._decision_log.append(entry)

        wip_for_history = (
            wip_waiting_before
            if time_advanced
            else self.sim.get_wip_waiting()
        )

        self.sim.save_history_step(
            arrange_snapshot=arrange_actual_before,
            arrange_abstract_snapshot=arrange_abstract_before,
            wip_waiting_snapshot=wip_for_history,
        )
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

    def get_schedule(self) -> list:
        return self.sim.schedule if self.sim else []

    def get_history(self) -> list:
        return self.sim.history if self.sim else []

    def get_decision_log(self) -> list:
        return list(self._decision_log)

    def action_masks(self) -> np.ndarray:
        """MaskablePPO용 – 현재 idle EQP에서 feasible한 ppk_oper mask (O×P)."""
        n_ppk = self._O * self._P
        ppk_mask = np.zeros(n_ppk, dtype=bool)

        if self.sim is None:
            ppk_mask[0] = True
            return ppk_mask

        eqp_id = self.sim.current_idle_eqp()
        if eqp_id is None:
            ppk_mask[0] = True
            return ppk_mask

        for flat in self.sim.get_feasible_ppk_oper(eqp_id):
            if 0 <= flat < n_ppk:
                ppk_mask[flat] = True

        if not ppk_mask.any():
            ppk_mask[0] = True

        return ppk_mask
