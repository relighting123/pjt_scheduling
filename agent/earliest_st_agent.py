"""
agent/earliest_st_agent.py – Earliest-ST 휴리스틱
현재 idle EQP 기준 feasible (PPK/OPER) 중 최소 ST
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

        def st_key(flat: int) -> Tuple:
            ppk, oper_id = sim.ppk_oper_from_flat(flat)
            lots = [
                l for l in sim.available_lots(eqp_id)
                if l["plan_prod_key"] == ppk and l["oper_id"] == oper_id
            ]
            if not lots:
                return (10**9, flat)
            min_st = min(int(l.get("processing_time", 10**9)) for l in lots)
            return (min_st, flat)

        flat = min(feasible, key=st_key)
        return np.array([flat], dtype=np.int64)
