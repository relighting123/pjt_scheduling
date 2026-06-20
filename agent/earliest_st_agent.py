"""
agent/earliest_st_agent.py – 최소 ST(Earliest-ST) 휴리스틱 에이전트

결정 규칙:
  - ST = (LOT, EQP) 조합 처리시간(분) 최소
  - ST 동률 시 기할당 parent split 연속 우선 → LOT_ID
"""
from typing import List, Optional, Tuple

from agent.split_priority import assigned_split_priority, prev_assigned_lot_id
from simulation.simulator import SchedulingSimulator


class EarliestSTAgent:
    def __init__(self, env_data: dict):
        self._lot_by_id = {lot["lot_id"]: lot for lot in env_data["lots"]}

    def _lot_sort_key(
        self,
        sim: SchedulingSimulator,
        eqp_id: Optional[str],
        lot: dict,
    ) -> Tuple:
        st = int(lot.get("processing_time", 10**9))
        split_key = assigned_split_priority(
            prev_assigned_lot_id(sim, eqp_id),
            lot["lot_id"],
            self._lot_by_id,
        )
        return (st, *split_key, lot["lot_id"])

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
            key=lambda i: self._lot_sort_key(sim, eqp_id, available_lots[i]),
        )
