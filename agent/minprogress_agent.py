"""
agent/minprogress_agent.py – 최소 진행률(Min-Progress) 휴리스틱
현재 idle EQP 기준 (PPK/OPER) 선택 – 계획 있으면 계획 기준, 없으면 WIP 기준
"""
from typing import List, Optional, Tuple

import numpy as np

from simulation.simulator import SchedulingSimulator


class MinProgressAgent:
    def __init__(self, env_data: dict):
        self._plan: List[dict] = env_data.get("plan", [])

    def _plan_row(self, prod: str, oper_id: str) -> Optional[dict]:
        for p in self._plan:
            if p["plan_prod_attr_val"] == prod and p["oper_id"] == oper_id:
                return p
        return None

    def _has_plan(self, prod: str, oper_id: str) -> bool:
        row = self._plan_row(prod, oper_id)
        return bool(row and row.get("d0_plan_qty", 0) > 0)

    def _plan_priority(self, prod: str, oper_id: str) -> int:
        row = self._plan_row(prod, oper_id)
        return row["priority"] if row else 99

    def _chart_max_x(self, sim: SchedulingSimulator) -> int:
        max_end = max((rec["END_TM"] for rec in sim.schedule), default=0)
        return max(sim.current_time, max_end, 1)

    def _in_flight_qty(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> int:
        """현재 장비에서 처리 중(배정됐지만 완료 전)인 lot 수."""
        return sim.get_in_flight_qty(prod, oper_id)

    def _expected_by_now(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> float:
        """간트차트 max end time(x축) 시점까지 capa 기준 도달 가능했을 생산량."""
        cap_rate = sim._oper_capacity_per_min(prod, oper_id)  # 매/분
        t_end = self._chart_max_x(sim)
        expected = cap_rate * t_end
        row = self._plan_row(prod, oper_id)
        if row and row.get("d0_plan_qty", 0) > 0:
            expected = min(expected, row["d0_plan_qty"])
        return max(expected, 1.0)

    def _normalized_slope(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> float:
        # completed_qty는 배정 시점에 증가하므로 처리 중(in-flight) 포함
        cum = sim.stats["completed_qty"].get((prod, oper_id), 0)
        return cum / self._expected_by_now(sim, prod, oper_id)

    def _remaining_work(
        self, sim: SchedulingSimulator, prod: str, oper_id: str,
    ) -> int:
        row = self._plan_row(prod, oper_id)
        if row and row.get("d0_plan_qty", 0) > 0:
            done = sim.stats["completed_qty"].get((prod, oper_id), 0)
            return max(row["d0_plan_qty"] - done, 0)
        wip = sim._wip_for(prod, oper_id)
        return wip["wip_qty"] if wip else 0

    def _score_assignment(self, sim: SchedulingSimulator, flat: int) -> Tuple:
        ppk, oper_id = sim.ppk_oper_from_flat(flat)
        eqp_id = sim.current_idle_eqp()
        # 동일 setup(전환 불필요) 우선 + ST 짧은 조합 우선 → 불필요한 전환 최소화
        lot_cd, temp = sim._bucket_lot_cd_temp(ppk, oper_id)
        needs_conversion = sim._would_need_conversion(eqp_id, lot_cd, temp)
        st = sim._st_per_wafer_for_eqp(eqp_id, ppk, oper_id)
        if st is None:
            st = float("inf")
        # in_flight: 이미 다른 EQP가 처리 중인 (ppk, oper) → 추가 배정 후순위
        in_flight = self._in_flight_qty(sim, ppk, oper_id)
        return (
            self._plan_priority(ppk, oper_id),
            self._normalized_slope(sim, ppk, oper_id),
            int(needs_conversion),
            st,
            in_flight,
            -self._remaining_work(sim, ppk, oper_id),
            flat,
        )

    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        eqp_id = sim.current_idle_eqp()
        if eqp_id is None:
            return np.array([0], dtype=np.int64)

        feasible = sim.get_feasible_ppk_oper(eqp_id)
        if not feasible:
            return np.array([0], dtype=np.int64)

        flat = min(feasible, key=lambda f: self._score_assignment(sim, f))
        return np.array([flat], dtype=np.int64)
