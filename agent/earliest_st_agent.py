"""
agent/earliest_st_agent.py – 최소 ST(Earliest-ST) 휴리스틱 에이전트

계획(plan)을 참조하지 않고, 현재 EQP에서 투입 가능한 조합 중
장비×자재 소요시간(ST, processing_time)이 가장 짧은 LOT을 배정합니다.

결정 규칙:
  - ST = availability 기준 (LOT, EQP) 조합 처리시간(분)
  - 동시 idle EQP가 여러 대이면, 각 EQP의 최소 ST 중 전역 최소인 EQP부터 결정
  - 동률 시 LOT_ID 순
"""
from typing import List, Optional, Tuple

from simulation.simulator import SchedulingSimulator


class EarliestSTAgent:
    def __init__(self, env_data: dict):
        pass

    def _lot_sort_key(self, lot: dict) -> Tuple[int, str]:
        st = int(lot.get("processing_time", 10**9))
        return (st, lot["lot_id"])

    def predict(
        self,
        sim: SchedulingSimulator,
        eqp_id: Optional[str],
        available_lots: List[dict],
    ) -> int:
        if not available_lots:
            return 0
        return min(
            range(len(available_lots)),
            key=lambda i: self._lot_sort_key(available_lots[i]),
        )
