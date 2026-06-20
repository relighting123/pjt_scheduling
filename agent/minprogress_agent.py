"""
agent/minprogress_agent.py – 최소 진행률(Min-Progress) 휴리스틱
(PPK/OPER, EQP) 2단계 액션 반환
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.simulator import SchedulingSimulator


class MinProgressAgent:
    def __init__(self, env_data: dict):
        self._plan: List[dict] = env_data["plan"]
        self._lot_by_id: Dict[str, dict] = {
            lot["lot_id"]: lot for lot in env_data["lots"]
        }

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
        row = self._plan_row(prod, oper_id)
        plan_qty = row["d0_plan_qty"] if row else 1
        return (cum / max(plan_qty, 1)) / t_end

    def _remaining_plan(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> int:
        row = self._plan_row(prod, oper_id)
        if not row:
            return 0
        done = sim.stats["completed_qty"].get((prod, oper_id), 0)
        return max(row["d0_plan_qty"] - done, 0)

    def _select_product(
        self,
        sim: SchedulingSimulator,
        oper_id: str,
        available_lots: List[dict],
    ) -> Optional[str]:
        queue_prods = {lot["plan_prod_key"] for lot in available_lots}
        if not queue_prods:
            return None

        min_priority = min(self._plan_priority(p, oper_id) for p in queue_prods)
        tier = [p for p in queue_prods if self._plan_priority(p, oper_id) == min_priority]

        return min(
            tier,
            key=lambda prod: (
                self._normalized_slope(sim, prod, oper_id),
                -self._remaining_plan(sim, prod, oper_id),
                prod,
            ),
        )

    def _score_assignment(self, sim: SchedulingSimulator, flat: int, ei: int) -> Tuple:
        ppk, oper_id = sim.ppk_oper_from_flat(flat)
        return (
            self._plan_priority(ppk, oper_id),
            self._normalized_slope(sim, ppk, oper_id),
            -self._remaining_plan(sim, ppk, oper_id),
            flat,
            ei,
        )

    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        feasible = sim.get_feasible_assignments()
        if not feasible:
            return np.array([0, 0], dtype=np.int64)
        flat, ei = min(feasible, key=lambda a: self._score_assignment(sim, a[0], a[1])[:3])
        return np.array([flat, ei], dtype=np.int64)
