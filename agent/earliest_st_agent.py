"""
agent/earliest_st_agent.py – Earliest-ST 휴리스틱
(PPK/OPER, EQP) 2단계 액션 – feasible 중 최소 ST
"""
from typing import Tuple

import numpy as np

from simulation.simulator import SchedulingSimulator


class EarliestSTAgent:
    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        feasible = sim.get_feasible_assignments()
        if not feasible:
            return np.array([0, 0], dtype=np.int64)

        def st_key(item: Tuple[int, int]) -> Tuple:
            flat, ei = item
            ppk, oper_id = sim.ppk_oper_from_flat(flat)
            eqp_ids = sim._env_data["eqp_ids"]
            eqp_id = eqp_ids[ei] if ei < len(eqp_ids) else eqp_ids[0]
            lots = [
                l for l in sim.available_lots(eqp_id)
                if l["plan_prod_key"] == ppk and l["oper_id"] == oper_id
            ]
            if not lots:
                return (10**9, flat, ei)
            min_st = min(int(l.get("processing_time", 10**9)) for l in lots)
            return (min_st, flat, ei)

        flat, ei = min(feasible, key=st_key)
        return np.array([flat, ei], dtype=np.int64)
