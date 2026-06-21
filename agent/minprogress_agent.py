"""
agent/minprogress_agent.py – 최소 진행률(Min-Progress) 휴리스틱
현재 idle EQP 기준 (PPK/OPER) 선택 – WIP·우선순위 참조 (계획량 미참조)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.simulator import SchedulingSimulator


class MinProgressAgent:
    def __init__(self, env_data: dict):
        self._plan: List[dict] = env_data.get("plan", [])

    def _plan_row(self, prod: str, oper_id: str) -> Optional[dict]:
        for p in self._plan:
            if p["plan_prod_key"] == prod and p["oper_id"] == oper_id:
                return p
        return None

    def _plan_priority(self, prod: str, oper_id: str) -> int:
        row = self._plan_row(prod, oper_id)
        return row["priority"] if row else 99

    def _chart_max_x(self, sim: SchedulingSimulator) -> int:
        max_end = max((rec["END_TM"] for rec in sim.schedule), default=0)
        return max(sim.current_time, max_end, 1)

    def _cumulative_qty_at(
        self, sim: SchedulingSimulator, prod: str, oper_id: str, t_end: int,
    ) -> int:
        total = 0
        for rec in sim.schedule:
            if rec["PLAN_PROD_KEY"] != prod or rec["OPER_ID"] != oper_id:
                continue
            if rec["START_TM"] <= t_end:
                total += rec.get("WF_QTY", 0)
        return total

    def _normalized_slope(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> float:
        t_end = self._chart_max_x(sim)
        cum = self._cumulative_qty_at(sim, prod, oper_id, t_end)
        return cum / t_end

    def _remaining_wip(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> int:
        wip = sim._wip_for(prod, oper_id)
        return wip["wip_qty"] if wip else 0

    def _score_assignment(self, sim: SchedulingSimulator, flat: int) -> Tuple:
        ppk, oper_id = sim.ppk_oper_from_flat(flat)
        return (
            self._plan_priority(ppk, oper_id),
            self._normalized_slope(sim, ppk, oper_id),
            -self._remaining_wip(sim, ppk, oper_id),
            flat,
        )

    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        eqp_id = sim.current_idle_eqp()
        if eqp_id is None:
            return np.array([0], dtype=np.int64)

        feasible = sim.get_feasible_ppk_oper(eqp_id)
        if not feasible:
            return np.array([0], dtype=np.int64)

        flat = min(feasible, key=lambda f: self._score_assignment(sim, f)[:3])
        return np.array([flat], dtype=np.int64)
