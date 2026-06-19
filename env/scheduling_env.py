"""
env/scheduling_env.py – Gymnasium 커스텀 환경
SchedulingSimulator를 감싸 RL 표준 인터페이스(reset/step)를 제공합니다.
StableBaselines3의 PPO / MaskablePPO와 호환됩니다.
"""
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from simulation.simulator import SchedulingSimulator


class SchedulingEnv(gym.Env):
    """
    Post-Scheduling 강화학습 환경

    - 관측 공간: Box(0, 1, shape=(obs_dim,), float32)
    - 행동 공간: Discrete(max_queue_size)  – 배정할 LOT의 큐 인덱스
    - 보상:      동일 OPER 연속 보너스, idle 패널티, 완료 보상
    - 에피소드 종료: 모든 LOT 배정 완료 또는 시간 초과
    """

    metadata = {"render_modes": []}

    def __init__(self, env_data: dict, render_mode: Optional[str] = None):
        super().__init__()
        self._env_data = env_data

        env_cfg = CONFIG.env
        O = env_cfg.max_oper_count
        P = env_cfg.max_prod_count

        # 이분 그래프 집계 관측 차원 (EQP/LOT 수와 무관한 고정 크기)
        #   Group A [O×P×4] WIP 노드  + Group B [O×4] EQP 노드
        #   Group C [O×P×3] 계획 노드 + Group D [6]   컨텍스트
        obs_dim = O * P * 7 + O * 4 + 6

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(env_cfg.max_queue_size)

        self.sim: Optional[SchedulingSimulator] = None
        self.render_mode = render_mode
        self._total_reward = 0.0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        목적: 에피소드 초기화 및 초기 관측 반환
        Input:  seed, options (Gymnasium 표준)
        Output: (obs: np.ndarray, info: dict)
        """
        super().reset(seed=seed)
        self.sim = SchedulingSimulator(self._env_data, CONFIG.reward)
        self._total_reward = 0.0
        obs  = self.sim.get_observation()
        info = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        목적: 에이전트 행동 실행 후 (관측, 보상, 종료여부, ...) 반환
        Input:  action (int) – 배정할 LOT의 큐 인덱스 (0-based)
        Output: (obs, reward, terminated, truncated, info)
            obs:        np.ndarray  다음 상태 관측
            reward:     float       즉각 보상
            terminated: bool        시뮬레이션 자연 종료 여부
            truncated:  bool        시간 초과 종료 여부
            info:       dict        통계 정보
        """
        eqp_id = self.sim.current_idle_eqp()
        available = self.sim.available_lots(eqp_id) if eqp_id else []

        # 행동이 큐 범위를 벗어나면 모듈로로 클램핑
        if available:
            idx    = int(action) % len(available)
            lot_id = available[idx]["lot_id"]
            reward = self.sim.assign_lot(eqp_id, lot_id)
        else:
            reward = 0.0

        self.sim.save_history_step()
        self._total_reward += reward

        terminated = self.sim.is_done()
        truncated  = (self.sim.current_time >= self.sim.sim_end) and not terminated
        obs        = self.sim.get_observation()
        info       = {
            "total_reward":  self._total_reward,
            "oper_switches": self.sim.stats["oper_switches"],
            "prod_switches": self.sim.stats["prod_switches"],
            "idle_total":    self.sim.stats["idle_total"],
            "completed_qty": dict(self.sim.stats["completed_qty"]),
        }
        return obs, reward, terminated, truncated, info

    def get_schedule(self) -> list:
        """
        목적: 현재 에피소드의 배정 결과 반환 (추론 후 결과 추출용)
        Input:  없음
        Output: [{EQP_ID, LOT_ID, CARRIER_ID, PLAN_PROD_KEY, OPER_ID, ST, SEQ,
                  START_TM, END_TM}, ...]
        """
        return self.sim.schedule if self.sim else []

    def get_history(self) -> list:
        """
        목적: UI 단계별 재생용 히스토리 반환
        Input:  없음
        Output: [HistoryStep dict, ...]
        """
        return self.sim.history if self.sim else []

    def action_masks(self) -> np.ndarray:
        """
        목적: MaskablePPO 사용 시 유효 행동 마스크 반환
        Input:  없음
        Output: np.ndarray shape=(max_queue_size,) bool
            True  = 유효한 행동
            False = 패딩 (LOT 없음)
        """
        eqp_id = self.sim.current_idle_eqp() if self.sim else None
        n_lots = len(self.sim.available_lots(eqp_id)) if eqp_id else 0
        Q = CONFIG.env.max_queue_size
        mask = np.zeros(Q, dtype=bool)
        mask[:max(n_lots, 1)] = True  # 최소 1개는 True (PPO 요건)
        return mask
