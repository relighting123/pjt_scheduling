"""
agent/earliest_st_agent.py – Earliest-ST 휴리스틱

idle EQP에서 feasible (PPK/OPER) 중
예상 종료 시각(현재+conversion+장수×ST)이 가장 이른 조합을 선택.
LOT 자동 배정도 동일 기준(_auto_select_lot)으로 수행.
"""
from typing import Tuple

import numpy as np

from simulation.simulator import SchedulingSimulator


class EarliestSTAgent:
    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        eqp_id = sim.current_idle_eqp()
        if eqp_id is None:
            return np.array([0], dtype=np.int64)

        feasible = sim.get_feasible_ppk_oper(eqp_id)
        if not feasible:
            return np.array([0], dtype=np.int64)

        def score_key(flat: int) -> Tuple[int, int, int]:
            ppk, oper_id = sim.ppk_oper_from_flat(flat)
            lots = [
                l for l in sim.available_lots(eqp_id)
                if l["plan_prod_key"] == ppk and l["oper_id"] == oper_id
            ]
            if not lots:
                return (10**9, 10**9, flat)
            best = min(lots, key=lambda lot: sim.earliest_st_lot_key(eqp_id, lot))
            end, proc, _ = sim.earliest_st_lot_key(eqp_id, best)
            return (end, proc, flat)

        flat = min(feasible, key=score_key)
        return np.array([flat], dtype=np.int64)
