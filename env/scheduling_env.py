"""

env/scheduling_env.py – Gymnasium 커스텀 환경

SchedulingSimulator를 감싸 RL 표준 인터페이스(reset/step)를 제공합니다.

StableBaselines3 MaskablePPO와 호환됩니다.

"""

from typing import Optional, Tuple, Union



import numpy as np

import gymnasium as gym

from gymnasium import spaces



from config import CONFIG

from simulation.simulator import SchedulingSimulator





def compute_obs_dim() -> int:

    """Global(6) + WIP(O×P) + Plan×2(O×P) + EQP(M×5) + Context(4) + Proc(O×M)"""

    O = CONFIG.env.max_oper_count

    P = CONFIG.env.max_prod_count

    M = CONFIG.env.max_eqp_count

    return 6 + O * P * 3 + M * 5 + 4 + O * M





class SchedulingEnv(gym.Env):

    """

    Post-Scheduling 강화학습 환경



    - 관측: 전역 + PPK/OPER WIP 비율 + 계획 진도 + EQP 상태

    - 행동: MultiDiscrete([O×P, M]) – (PPK/OPER bucket, EQP)

    - LOT 배정: 우선순위 → ST 자동 규칙

    """



    metadata = {"render_modes": []}



    def __init__(self, env_data: dict, render_mode: Optional[str] = None, record_history: bool = True):

        super().__init__()

        self._env_data = env_data

        self._record_history = record_history



        env_cfg = CONFIG.env

        O = env_cfg.max_oper_count

        P = env_cfg.max_prod_count

        M = env_cfg.max_eqp_count



        obs_dim = compute_obs_dim()

        self.observation_space = spaces.Box(

            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,

        )

        self.action_space = spaces.MultiDiscrete([O * P, M])



        self.sim: Optional[SchedulingSimulator] = None

        self.render_mode = render_mode

        self._total_reward = 0.0

        self._O = O

        self._P = P

        self._M = M



    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:

        super().reset(seed=seed)

        self.sim = SchedulingSimulator(

            self._env_data, CONFIG.reward, record_history=self._record_history,

        )

        self._total_reward = 0.0

        obs = self.sim.get_observation()

        return obs, {}



    def step(self, action: Union[int, np.ndarray, list]) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action_arr = np.asarray(action, dtype=np.int64).flatten()

        ppk_oper_idx = int(action_arr[0]) if len(action_arr) > 0 else 0

        eqp_idx = int(action_arr[1]) if len(action_arr) > 1 else 0



        time_at_step_start = self.sim.current_time

        while not self.sim.get_idle_eqps() and self.sim._has_pending_processing():

            self.sim._advance_to_next_decision()

        time_advanced = self.sim.current_time != time_at_step_start



        eqp_ids = self._env_data["eqp_ids"]

        eqp_idx = eqp_idx % max(len(eqp_ids), 1)

        eqp_id = eqp_ids[eqp_idx] if eqp_idx < len(eqp_ids) else eqp_ids[0]



        ppk, oper_id = self.sim.ppk_oper_from_flat(ppk_oper_idx % (self._O * self._P))



        arrange_actual_before = self.sim.get_remaining_arrange_actual()

        arrange_abstract_before = self.sim.get_abstract_arrange()

        wip_waiting_before = self.sim.get_wip_waiting()



        reward = 0.0

        if self.sim.eqps[eqp_id].status == "idle":

            reward = self.sim.assign_ppk_oper(eqp_id, ppk, oper_id)

        elif self.sim.get_feasible_assignments():

            reward = -0.5



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



        terminated = self.sim.is_done()

        truncated = (self.sim.current_time >= self.sim.sim_end) and not terminated

        obs = self.sim.get_observation()

        info = {

            "total_reward":  self._total_reward,

            "oper_switches": self.sim.stats["oper_switches"],

            "prod_switches": self.sim.stats["prod_switches"],

            "conversions":   self.sim.stats.get("conversions", 0),

            "idle_total":    self.sim.stats["idle_total"],

            "completed_qty": dict(self.sim.stats["completed_qty"]),

        }

        return obs, reward, terminated, truncated, info



    def get_schedule(self) -> list:

        return self.sim.schedule if self.sim else []



    def get_history(self) -> list:

        return self.sim.history if self.sim else []



    def action_masks(self) -> np.ndarray:

        """

        MaskablePPO용 – [ppk_oper mask (O×P), eqp mask (M)]

        """

        n_ppk = self._O * self._P

        n_eqp = self._M

        ppk_mask = np.zeros(n_ppk, dtype=bool)

        eqp_mask = np.zeros(n_eqp, dtype=bool)



        if self.sim is None:

            ppk_mask[0] = True

            eqp_mask[0] = True

            return np.concatenate([ppk_mask, eqp_mask])



        feasible = self.sim.get_feasible_assignments()

        for flat, ei in feasible:

            if 0 <= flat < n_ppk:

                ppk_mask[flat] = True

            if 0 <= ei < n_eqp:

                eqp_mask[ei] = True



        if not ppk_mask.any():

            ppk_mask[0] = True

        if not eqp_mask.any():

            eqp_mask[0] = True



        return np.concatenate([ppk_mask, eqp_mask])


