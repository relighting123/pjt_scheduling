"""
agent/minprogress_agent.py – 최소 진행률(Min-Progress) 휴리스틱 에이전트

결정 규칙 (매 결정 시점):
  1. 큐 LOT의 공정(OPER) 기준 PLAN_PRIORITY 최우선 (숫자 낮을수록 우선)
  2. 동순위 제품 중 차트 마지막 x에서 계획 대비 생산 기울기 최소
     기울기 = (누적 투입량 / D0_PLAN_QTY) / x_end
  3. 기울기 동률 시 잔여 계획량(D0 - 투입)이 큰 제품 우선 (균형 배분)
  4. 선정 제품 내 LOT은 ST → SEQ → LOT_ID 순 (Earliest-ST와 제품 선택 기준 분리)
"""
from typing import Dict, List, Optional, Tuple

from simulation.simulator import SchedulingSimulator


class MinProgressAgent:
    def __init__(self, env_data: dict):
        self._plan: List[dict] = env_data["plan"]
        self._lot_by_id: Dict[str, dict] = {
            lot["lot_id"]: lot for lot in env_data["lots"]
        }
        self._initial_start: Dict[str, int] = {
            rec["LOT_ID"]: rec["START_TM"]
            for rec in env_data["initial_schedule"]
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

    def _lot_sort_key(self, lot: dict) -> Tuple[int, int, str]:
        lot_id = lot["lot_id"]
        if lot.get("is_abstract"):
            start_tm = int(lot.get("oper_in_time", 10**9))
        else:
            start_tm = self._initial_start.get(lot_id, 10**9)
        seq = self._lot_by_id.get(lot_id, {}).get("seq", 999)
        return (start_tm, seq, lot_id)

    def predict(
        self,
        sim: SchedulingSimulator,
        eqp_id: Optional[str],
        available_lots: List[dict],
    ) -> int:
        if not available_lots:
            return 0

        oper_id = available_lots[0]["oper_id"]
        target_prod = self._select_product(sim, oper_id, available_lots)
        matching = [
            (i, lot) for i, lot in enumerate(available_lots)
            if lot["plan_prod_key"] == target_prod
        ]
        return min(matching, key=lambda x: self._lot_sort_key(x[1]))[0]
