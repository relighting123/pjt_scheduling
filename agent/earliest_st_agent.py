"""
agent/earliest_st_agent.py – Earliest-ST 휴리스틱

EQP×carrier(LOT) 조합별 ST×qty(스플릿 이후)로 예상 종료 시각을 계산하고,
idle 장비 전체에서 (end_time + 장수×ST)가 가장 작은 조합을 선택한다.
실제 (EQP, LOT) 배정은 simulator min_st 경로(_earliest_st_pick)에서 수행.
"""
import numpy as np

from simulation.simulator import SchedulingSimulator


class EarliestSTAgent:
    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        """배정은 simulator가 결정; env step 호환용 더미 action."""
        return np.array([0], dtype=np.int64)
