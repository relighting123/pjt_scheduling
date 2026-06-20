"""
simulation/simulator.py – 이산 사건 시뮬레이션(DES) 엔진
RL 환경의 내부 시뮬레이터로서 EQP 상태 및 LOT 배정 이력을 관리합니다.
"""
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import CONFIG, RewardConfig
from utils.helpers import encode_normalized


class ToolTracker:
    """LOT_CD × EQP_MODEL 동시 가공 상한 추적."""

    def __init__(self, capacity: dict, eqp_model_map: dict):
        self._capacity = dict(capacity)
        self._eqp_model_map = eqp_model_map
        self._busy: dict = {}

    def can_assign(self, lot_cd: str, eqp_id: str) -> bool:
        model = self._eqp_model_map.get(eqp_id, "A")
        cap = self._capacity.get((lot_cd, model), 999)
        return self._busy.get((lot_cd, model), 0) < cap

    def occupy(self, lot_cd: str, eqp_id: str) -> None:
        model = self._eqp_model_map.get(eqp_id, "A")
        key = (lot_cd, model)
        self._busy[key] = self._busy.get(key, 0) + 1

    def release(self, lot_cd: str, eqp_id: str) -> None:
        model = self._eqp_model_map.get(eqp_id, "A")
        key = (lot_cd, model)
        if key in self._busy:
            self._busy[key] = max(self._busy[key] - 1, 0)

    def utilization(self) -> float:
        if not self._capacity:
            return 0.0
        ratios = []
        for (lot_cd, model), cap in self._capacity.items():
            if cap <= 0:
                continue
            ratios.append(self._busy.get((lot_cd, model), 0) / cap)
        return sum(ratios) / max(len(ratios), 1)


# ── 도메인 객체 ───────────────────────────────────────────────────────────────

@dataclass
class Lot:
    lot_id:          str
    carrier_id:      str
    plan_prod_key:   str
    oper_id:         str
    seq:             int
    wf_qty:          int
    processing_time: int   # 기본값(원본 배정 EQP 기준) – 실제 사용 시 proc_time_matrix 참조
    priority:        int   = 1
    original_eqp:    str   = ""
    parent_lot_id:   str   = ""
    lot_cd:          str   = ""
    temp:            str   = ""


@dataclass
class Equipment:
    eqp_id:       str
    status:       str            = "idle"   # "idle" | "busy"
    current_lot:  Optional[str] = None
    current_oper: Optional[str] = None
    current_prod: Optional[str] = None
    free_at:      int            = 0
    prev_oper:    Optional[str] = None
    prev_prod:    Optional[str] = None
    prev_lot_id:  Optional[str] = None
    prev_lot_cd:  Optional[str] = None
    prev_temp:    Optional[str] = None
    idle_accum:   int            = 0
    oper_switches:int            = 0
    prod_switches:int            = 0
    conversion_count: int       = 0


# ── 시뮬레이터 ────────────────────────────────────────────────────────────────

class SchedulingSimulator:
    """
    이산 사건 시뮬레이터 – EQP가 idle이 되는 순간마다 에이전트 결정을 요청합니다.

    사용 흐름:
        sim = SchedulingSimulator(env_data, config)
        sim.reset()
        while not sim.is_done():
            eqp_id = sim.current_idle_eqp()
            lots   = sim.available_lots(eqp_id)
            obs    = sim.get_observation()
            # agent picks action -> lot_id
            reward = sim.assign_lot(eqp_id, lot_id)
            sim.save_history_step()
    """

    def __init__(self, env_data: dict, reward_cfg: RewardConfig = None, record_history: bool = True):
        self._env_data   = env_data
        self._reward_cfg = reward_cfg or CONFIG.reward
        self._record_history = record_history
        self.reset()

    # ── 초기화 ──────────────────────────────────────────────────────────────

    def reset(self):
        """
        목적: 시뮬레이터를 초기 상태로 되돌림 (에피소드 시작)
        Input:  없음
        Output: 없음 (내부 상태 초기화)
        """
        data = self._env_data

        self.current_time: int = 0
        self.sim_end:       int = data["sim_end_minutes"]
        self.soft_cutoff:   int = data.get("soft_cutoff_minutes", CONFIG.env.soft_cutoff_minutes)
        self._conversion_minutes: int = data.get("conversion_minutes", CONFIG.env.conversion_minutes)
        self._lot_attrs: Dict[str, dict] = dict(data.get("lot_attrs", {}))
        self._initial_start: Dict[str, int] = {
            r["LOT_ID"]: r["START_TM"] for r in data.get("initial_schedule", [])
        }

        self.eqps: Dict[str, Equipment] = {
            eid: Equipment(eqp_id=eid) for eid in data["eqp_ids"]
        }
        self._tool_tracker = ToolTracker(
            data.get("tool_capacity", {}),
            data.get("eqp_model_map", {}),
        )

        self.lot_pool: Dict[str, Lot] = {}
        for ld in data["lots"]:
            self.lot_pool[ld["lot_id"]] = Lot(
                lot_id=ld["lot_id"],
                carrier_id=ld["carrier_id"],
                plan_prod_key=ld["plan_prod_key"],
                oper_id=ld["oper_id"],
                seq=ld["seq"],
                wf_qty=ld["wf_qty"],
                processing_time=ld["processing_time"],
                priority=ld.get("priority", 1),
                original_eqp=ld.get("original_eqp", ""),
                parent_lot_id=ld.get("parent_lot_id", ""),
                lot_cd=ld.get("lot_cd", ""),
                temp=ld.get("temp", ""),
            )

        self.eqp_queues: Dict[str, List[str]] = {
            eid: list(lots)
            for eid, lots in data["eqp_lot_map"].items()
        }

        self._initial_arrange: List[dict] = list(
            data.get("arrange_actual_table", data.get("arrange_table", []))
        )
        self._eqp_model_map: Dict[str, str] = dict(data.get("eqp_model_map", {}))
        self._abstract_rows: List[dict] = []
        self._abstract_assigned: set = set()
        self._in_flight: Dict[str, dict] = {}
        self._inject_deadline: Dict[str, int] = dict(data.get("lot_inject_deadline", {}))

        self._eqp_selection: str = data.get("eqp_selection", "order")

        self._event_q: list = []
        for eid in data["eqp_ids"]:
            heapq.heappush(self._event_q, (0, eid))

        self.schedule:   List[dict] = []
        self.stats = {
            "idle_total":    0,
            "oper_switches": 0,
            "prod_switches": 0,
            "conversions":   0,
            "completed_qty": {},   # {(prod, oper): qty}
        }

        self.history:    List[dict] = []
        self._step_idx   = 0
        self._last_assigned: Optional[dict] = None
        self._current_eqp: Optional[str] = None
        self._advance_to_next_decision()
        if self._record_history:
            self._append_initial_history()

    # ── 결정 포인트 탐색 ─────────────────────────────────────────────────────

    def _eqp_min_proc_time(self, eqp_id: str) -> Optional[int]:
        """EQP에서 투입 가능한 LOT 중 최소 소요시간(ST)"""
        lots = self.available_lots(eqp_id)
        if not lots:
            return None
        return min(int(lot.get("processing_time", 10**9)) for lot in lots)

    def _idle_eqps_with_work(self) -> List[str]:
        return [
            eqp_id for eqp_id in self._env_data["eqp_ids"]
            if self.eqps[eqp_id].status == "idle"
            and (
                self._get_available_lots(eqp_id)
                or self._abstract_assignable_on_eqp(eqp_id)
            )
        ]

    def _pick_next_idle_eqp(self) -> Optional[str]:
        """다음 결정 EQP – order: 목록 순서, min_st: idle EQP 중 최소 ST 우선"""
        candidates = self._idle_eqps_with_work()
        if not candidates:
            return None
        if self._eqp_selection == "min_st":
            return min(
                candidates,
                key=lambda e: (self._eqp_min_proc_time(e) or 10**9, e),
            )
        return candidates[0]

    def _advance_to_next_decision(self):
        """
        목적: 가장 가까운 END_TM까지 시간 전진 후, 동시 완료를 모두 반영하고
              다음 에이전트 결정이 필요한 idle EQP를 선택
        """
        self._current_eqp = None
        while self._event_q:
            next_time = self._event_q[0][0]
            batch: List[tuple] = []
            while self._event_q and self._event_q[0][0] == next_time:
                batch.append(heapq.heappop(self._event_q))

            for time, eqp_id in batch:
                eqp = self.eqps[eqp_id]
                if eqp.status == "busy":
                    lot_id = eqp.current_lot
                    if lot_id:
                        self._on_oper_complete(lot_id, time)
                    eqp.status      = "idle"
                    eqp.current_lot = None
                    eqp.free_at     = time

                self.current_time = time
                if (
                    not self._get_available_lots(eqp_id)
                    and not self._abstract_assignable_on_eqp(eqp_id)
                    and time < self.sim_end
                ):
                    heapq.heappush(self._event_q, (self.sim_end, eqp_id))

            self._current_eqp = self._pick_next_idle_eqp()
            if self._current_eqp:
                return

    def _select_same_time_next_eqp(self):
        """배정 직후 동일 시각의 다른 idle EQP 결정 포인트 탐색 (시간 전진 없음)"""
        self._current_eqp = self._pick_next_idle_eqp()

    def _on_oper_complete(self, lot_id: str, complete_time: int) -> None:
        """전 공정 처리 완료(END_TM) → 다음 공정 추상 arrange 유입"""
        meta = self._in_flight.pop(lot_id, None)
        if not meta:
            return

        eqp_id = meta.get("eqp_id")
        lot_cd = meta.get("lot_cd")
        if eqp_id and lot_cd:
            self._tool_tracker.release(lot_cd, eqp_id)

        ppk = meta["plan_prod_key"]
        seq = meta["seq"]
        wf_qty = meta["wf_qty"]
        next_info = self._env_data.get("flow_next", {}).get(ppk, {}).get(seq)
        if not next_info:
            return

        next_oper = next_info["next_oper"]
        next_seq  = next_info["next_seq"]
        pm = self._env_data.get("plan_meta", {}).get(
            (ppk, next_oper), {"priority": 1, "d0_plan_qty": 0},
        )
        proc_map = self._env_data.get("abstract_proc_time", {})
        route_map = self._env_data.get("abstract_route_map", {})
        routes = self._env_data.get("abstract_routes_by_ppk_oper", {}).get((ppk, next_oper))

        if routes:
            model_st_pairs = routes
        else:
            models = self._env_data.get("oper_eqp_models", {}).get(next_oper, ["A"])
            model_st_pairs = [
                (m, proc_map.get((next_oper, m), 60)) for m in models
            ]

        for model, default_st in model_st_pairs:
            proc_time = route_map.get((ppk, next_oper, model), default_st)
            abs_key = f"{ppk}|{next_oper}|{model}|{complete_time}"
            unit = {
                "lot_id":       lot_id,
                "oper_in_time": complete_time,
                "from_oper":    meta["oper_id"],
                "wf_qty":       wf_qty,
                "carrier_id":   meta.get("carrier_id", ""),
                "assigned":     False,
            }
            existing = next(
                (r for r in self._abstract_rows if r["abs_key"] == abs_key), None,
            )
            if existing and abs_key not in self._abstract_assigned:
                existing["wip_qty"] += 1
                existing["wip_qty_init"] += 1
                existing.setdefault("lot_units", []).append(unit)
            else:
                self._abstract_rows.append({
                    "abs_key":         abs_key,
                    "plan_prod_key":   ppk,
                    "eqp_model":       model,
                    "oper_id":         next_oper,
                    "seq":             next_seq,
                    "from_oper":       meta["oper_id"],
                    "wip_qty":         1,
                    "wip_qty_init":    1,
                    "proc_time":       proc_time,
                    "min_inject_time": complete_time,
                    "wf_qty":          wf_qty,
                    "plan_priority":   pm["priority"],
                    "d0_plan_qty":     pm["d0_plan_qty"],
                    "lot_units":       [unit],
                })

    def _find_abstract_unit(self, lot_id: str):
        """추상 arrange에 대기 중인 실 LOT 단위 조회"""
        for row in self._abstract_rows:
            for unit in row.get("lot_units", []):
                if unit["lot_id"] == lot_id and not unit.get("assigned"):
                    return row, unit
        return None, None

    def _abstract_assignable_on_eqp(self, eqp_id: str) -> List[dict]:
        model = self._eqp_model_map.get(eqp_id)
        if not model:
            return []
        return [
            row for row in self._abstract_rows
            if row["wip_qty"] > 0
            and row["eqp_model"] == model
            and self.current_time >= row["min_inject_time"]
        ]

    def _lot_injectable(self, lot_id: str) -> bool:
        """LOT이 현재 시각에 투입 가능한지 (시뮬 종료 전 + 미배정)"""
        if lot_id not in self.lot_pool:
            return False
        return self.current_time <= self.sim_end

    def _get_available_lots(self, eqp_id: str) -> List[str]:
        """
        목적: EQP에 배정 가능하며 투입 기한 이내인 LOT 목록 반환
        """
        return [
            lid for lid in self.eqp_queues.get(eqp_id, [])
            if self._lot_injectable(lid)
        ]

    def get_remaining_arrange(self) -> List[dict]:
        """하위 호환 – Actual 조합"""
        return self.get_remaining_arrange_actual()

    def get_remaining_arrange_actual(self) -> List[dict]:
        """
        목적: 현재 시점 투입 가능한 (EQP, LOT) Actual arrange 조합
        """
        rows = []
        for row in self._initial_arrange:
            lid = row["lot_id"]
            if not self._lot_injectable(lid):
                continue
            if lid not in self.eqp_queues.get(row["eqp_id"], []):
                continue
            item = dict(row)
            attrs = self._lot_attrs.get(lid, {})
            item["lot_cd"] = attrs.get("lot_cd", "")
            item["temp"] = attrs.get("temp", "")
            rows.append(item)
        return rows

    def get_abstract_arrange(self) -> List[dict]:
        """PPK×장비MODEL 추상 유입 재공 (전공정 완료 시 DES로 생성)"""
        out = []
        for row in self._abstract_rows:
            item = dict(row)
            item["lot_units"] = [dict(u) for u in row.get("lot_units", [])]
            out.append(item)
        return out

    def get_wip_waiting(self) -> Dict[str, int]:
        """(PPK|OPER)별 대기 WIP 웨이퍼 수 – Actual 풀 + 추상 lot_units"""
        wip: Dict[str, int] = {}
        for lot in self.lot_pool.values():
            key = f"{lot.plan_prod_key}|{lot.oper_id}"
            wip[key] = wip.get(key, 0) + lot.wf_qty
        for row in self._abstract_rows:
            for unit in row.get("lot_units", []):
                if unit.get("assigned"):
                    continue
                key = f"{row['plan_prod_key']}|{row['oper_id']}"
                wip[key] = wip.get(key, 0) + unit.get("wf_qty", row.get("wf_qty", 0))
        return wip

    def _serialize_wip_waiting(self) -> Dict[str, int]:
        return self.get_wip_waiting()

    def _lot_cd_temp(self, lot_id: str, lot: Optional[Lot] = None) -> tuple:
        if lot is not None and lot.lot_cd:
            return lot.lot_cd, lot.temp
        attrs = self._lot_attrs.get(lot_id, {})
        return attrs.get("lot_cd", "LC01"), attrs.get("temp", "T650")

    def _accumulate_idle(self, eqp: Equipment, reward: float, until: int) -> float:
        cfg = self._reward_cfg
        idle_duration = max(until - eqp.free_at, 0)
        if idle_duration > 0:
            eqp.idle_accum += idle_duration
            self.stats["idle_total"] += idle_duration
            reward += cfg.w_idle_per_min * idle_duration
        return reward

    def _apply_conversion_start(
        self, eqp: Equipment, lot_cd: str, temp: str, reward: float,
    ) -> tuple:
        """LOT_CD/TEMP 변경 시 conversion 60분 + 패널티."""
        start_time = max(self.current_time, eqp.free_at)
        reward = self._accumulate_idle(eqp, reward, start_time)
        needs_conv = (
            eqp.prev_lot_cd is not None
            and (eqp.prev_lot_cd != lot_cd or eqp.prev_temp != temp)
        )
        if needs_conv:
            reward += self._reward_cfg.w_conversion
            eqp.conversion_count += 1
            self.stats["conversions"] += 1
            conv_end = start_time + self._conversion_minutes
            reward = self._accumulate_idle(eqp, reward, conv_end)
            start_time = conv_end
        return start_time, reward

    def _plan_hit_reward(self, ppk: str, oper_id: str, wf_qty: int, end_time: int) -> float:
        cfg = self._reward_cfg
        reward = 0.0
        if end_time <= self.soft_cutoff:
            pm = self._env_data.get("plan_meta", {}).get((ppk, oper_id))
            if pm:
                key = (ppk, oper_id)
                done_before = self.stats["completed_qty"].get(key, 0)
                plan_qty = max(pm["d0_plan_qty"], 1)
                gap_before = max(plan_qty - done_before, 0)
                done_after = done_before + wf_qty
                gap_after = max(plan_qty - done_after, 0)
                reward += cfg.w_plan_hit * (gap_before - gap_after) / plan_qty
        if end_time > self.soft_cutoff:
            reward += cfg.w_late_finish * (end_time - self.soft_cutoff) / 60.0
        return reward

    def _ppk_has_feasible_assignment(self, ppk: str) -> bool:
        """현재 시점에 해당 PPK로 투입 가능한 (OPER, EQP) 조합이 있는지."""
        for flat, _ei in self.get_feasible_assignments():
            candidate_ppk, _oper = self.ppk_oper_from_flat(flat)
            if candidate_ppk == ppk:
                return True
        return False

    def _same_prod_reward(self, eqp: Equipment, ppk: str) -> float:
        """같은 PPK는 재공이 남을 때만 보너스; 고갈된 PPK에서 전환 시 소규모 보너스."""
        cfg = self._reward_cfg
        if eqp.prev_prod == ppk:
            if self._ppk_has_feasible_assignment(ppk):
                return cfg.w_same_prod
            return 0.0
        if eqp.prev_prod is not None:
            eqp.prod_switches += 1
            self.stats["prod_switches"] += 1
            if not self._ppk_has_feasible_assignment(eqp.prev_prod):
                return cfg.w_prod_switch
        return 0.0

    def _pacing_shaping_reward(
        self, ppk: str, oper_id: str, wf_qty: int, at_time: Optional[int] = None,
    ) -> float:
        """계획 직선(y*=plan×t/T) 대비 누적 편차가 줄어들면 + 보상."""
        cfg = self._reward_cfg
        if cfg.w_pacing <= 0:
            return 0.0
        pm = self._env_data.get("plan_meta", {}).get((ppk, oper_id))
        if not pm:
            return 0.0
        plan_qty = max(pm["d0_plan_qty"], 1)
        horizon = max(self.soft_cutoff, 1)
        t = self.current_time if at_time is None else at_time
        ideal = plan_qty * min(max(t, 0), horizon) / horizon
        key = (ppk, oper_id)
        done_before = self.stats["completed_qty"].get(key, 0)
        done_after = done_before + wf_qty
        err_before = abs(ideal - done_before)
        err_after = abs(ideal - done_after)
        return cfg.w_pacing * (err_before - err_after) / plan_qty

    def _auto_select_lot(self, eqp_id: str, candidates: List[dict]) -> Optional[str]:
        if not candidates:
            return None

        def sort_key(lot: dict):
            if lot.get("is_abstract"):
                start_tm = int(lot.get("oper_in_time", 10**9))
            else:
                start_tm = self._initial_start.get(lot["lot_id"], 10**9)
            seq = self._env_data.get("lots", [])
            seq_val = 0
            for ld in self._env_data.get("lots", []):
                if ld["lot_id"] == lot["lot_id"]:
                    seq_val = ld.get("seq", 0)
                    break
            if lot["lot_id"] in self.lot_pool:
                seq_val = self.lot_pool[lot["lot_id"]].seq
            return (lot.get("priority", 99), start_tm, -seq_val, lot["lot_id"])

        best = min(candidates, key=sort_key)
        return best["lot_id"]

    def assign_ppk_oper(self, eqp_id: str, ppk: str, oper_id: str) -> float:
        """(PPK, OPER) 선택 후 LOT 자동 배정."""
        lots = [
            l for l in self.available_lots(eqp_id)
            if l["plan_prod_key"] == ppk and l["oper_id"] == oper_id
        ]
        lot_id = self._auto_select_lot(eqp_id, lots)
        if lot_id is None:
            return -1.0
        lot_cd, _ = self._lot_cd_temp(lot_id, self.lot_pool.get(lot_id))
        if not self._tool_tracker.can_assign(lot_cd, eqp_id):
            return -1.0
        return self.assign_lot(eqp_id, lot_id)

    def ppk_oper_flat_index(self, oper_id: str, ppk: str) -> int:
        data = self._env_data
        O = CONFIG.env.max_oper_count
        P = CONFIG.env.max_prod_count
        oi = data["oper_idx"].get(oper_id, -1)
        pi = data["prod_idx"].get(ppk, -1)
        if oi < 0 or pi < 0:
            return 0
        return oi * P + pi

    def ppk_oper_from_flat(self, flat_idx: int) -> tuple:
        data = self._env_data
        P = CONFIG.env.max_prod_count
        oi = flat_idx // P
        pi = flat_idx % P
        oper_ids = data["oper_ids"]
        prod_keys = data["prod_keys"]
        oper_id = oper_ids[oi] if oi < len(oper_ids) else oper_ids[0]
        ppk = prod_keys[pi] if pi < len(prod_keys) else prod_keys[0]
        return ppk, oper_id

    def get_feasible_assignments(self) -> List[tuple]:
        """유효 (ppk_oper_flat_idx, eqp_idx) 목록."""
        data = self._env_data
        eqp_ids = data["eqp_ids"]
        feasible = []
        for ei, eqp_id in enumerate(eqp_ids):
            if self.eqps[eqp_id].status != "idle":
                continue
            lots = self.available_lots(eqp_id)
            if not lots:
                continue
            buckets = {(l["plan_prod_key"], l["oper_id"]) for l in lots}
            for ppk, oper_id in buckets:
                lot_id = self._auto_select_lot(eqp_id, [
                    l for l in lots if l["plan_prod_key"] == ppk and l["oper_id"] == oper_id
                ])
                if lot_id is None:
                    continue
                lot_cd, _ = self._lot_cd_temp(lot_id, self.lot_pool.get(lot_id))
                if not self._tool_tracker.can_assign(lot_cd, eqp_id):
                    continue
                flat = self.ppk_oper_flat_index(oper_id, ppk)
                feasible.append((flat, ei))
        return feasible

    def get_idle_eqps(self) -> List[str]:
        return [
            eid for eid in self._env_data["eqp_ids"]
            if self.eqps[eid].status == "idle"
            and (self._get_available_lots(eid) or self._abstract_assignable_on_eqp(eid))
        ]

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def current_idle_eqp(self) -> Optional[str]:
        """
        목적: 현재 결정 대기 중인 idle EQP ID 반환
        Input:  없음
        Output: "EQP001" 또는 None (시뮬레이션 종료)
        """
        return self._current_eqp

    def available_lots(self, eqp_id: str) -> List[dict]:
        """
        목적: 에이전트 선택을 위한 LOT 상세 정보 리스트 반환 (처리시간은 EQP 조합 기준)
        Input:  eqp_id="EQP001"
        Output: [{"lot_id":"LOT001","plan_prod_key":"PPK001","oper_id":"OPER001",
                  "wf_qty":25,"priority":1,"processing_time":120}, ...]
                  processing_time은 (LOT, EQP) 조합 기준값
        """
        proc_time_matrix = self._env_data.get("proc_time_matrix", {})
        lots = []
        for lid in self._get_available_lots(eqp_id):
            lot = self.lot_pool[lid]
            pt  = proc_time_matrix.get((lid, eqp_id), lot.processing_time)
            lots.append({
                "lot_id":          lot.lot_id,
                "plan_prod_key":   lot.plan_prod_key,
                "oper_id":         lot.oper_id,
                "wf_qty":          lot.wf_qty,
                "priority":        lot.priority,
                "processing_time": pt,
                "parent_lot_id":   lot.parent_lot_id,
                "lot_cd":          lot.lot_cd,
                "temp":            lot.temp,
                "seq":             lot.seq,
                "is_abstract":     False,
            })
        for row in self._abstract_assignable_on_eqp(eqp_id):
            for unit in row.get("lot_units", []):
                if unit.get("assigned"):
                    continue
                lots.append({
                    "lot_id":          unit["lot_id"],
                    "plan_prod_key":   row["plan_prod_key"],
                    "oper_id":         row["oper_id"],
                    "wf_qty":          unit.get("wf_qty", row["wf_qty"]),
                    "priority":        row["plan_priority"],
                    "processing_time": row["proc_time"],
                    "oper_in_time":    unit["oper_in_time"],
                    "seq":             row.get("seq", 1),
                    "is_abstract":     True,
                    "abs_key":         row["abs_key"],
                })
                lot_cd, temp = self._lot_cd_temp(unit["lot_id"])
                lots[-1]["lot_cd"] = lot_cd
                lots[-1]["temp"] = temp
        return lots

    def _assign_abstract_lot(
        self, eqp_id: str, lot_id: str, row: dict, unit: dict,
    ) -> float:
        """추상 유입 재공 배정 – 실 LOT ID 사용, wip_qty 감소"""
        if row["wip_qty"] <= 0 or unit.get("assigned"):
            return -1.0

        eqp = self.eqps[eqp_id]
        cfg = self._reward_cfg
        reward = 0.0

        lot_cd, temp = self._lot_cd_temp(lot_id)
        if not self._tool_tracker.can_assign(lot_cd, eqp_id):
            return -1.0

        oper_id = row["oper_id"]
        ppk     = row["plan_prod_key"]

        if eqp.prev_oper == oper_id:
            reward += cfg.w_same_oper
        elif eqp.prev_oper is not None:
            eqp.oper_switches += 1
            self.stats["oper_switches"] += 1

        wf_qty    = row["wf_qty"]
        proc_time = row["proc_time"]
        reward += self._same_prod_reward(eqp, ppk)
        reward += self._pacing_shaping_reward(ppk, oper_id, wf_qty)
        reward += cfg.w_completion * wf_qty / 25.0

        start_time, reward = self._apply_conversion_start(eqp, lot_cd, temp, reward)
        end_time   = start_time + proc_time
        reward += self._plan_hit_reward(ppk, oper_id, wf_qty, end_time)

        eqp.status       = "busy"
        eqp.current_lot  = lot_id
        eqp.current_oper = oper_id
        eqp.current_prod = ppk
        eqp.free_at      = end_time
        eqp.prev_oper    = oper_id
        eqp.prev_prod    = ppk
        eqp.prev_lot_id  = lot_id
        eqp.prev_lot_cd  = lot_cd
        eqp.prev_temp    = temp

        self._tool_tracker.occupy(lot_cd, eqp_id)

        heapq.heappush(self._event_q, (end_time, eqp_id))

        unit["assigned"] = True
        row["wip_qty"] = max(row["wip_qty"] - 1, 0)
        if row["wip_qty"] == 0:
            self._abstract_assigned.add(row["abs_key"])

        self._in_flight[lot_id] = {
            "plan_prod_key": ppk,
            "oper_id":       oper_id,
            "seq":           row["seq"],
            "wf_qty":        wf_qty,
            "carrier_id":    unit.get("carrier_id", ""),
            "end_time":      end_time,
            "lot_cd":        lot_cd,
            "eqp_id":        eqp_id,
        }

        key = (ppk, oper_id)
        self.stats["completed_qty"][key] = (
            self.stats["completed_qty"].get(key, 0) + wf_qty
        )

        self.schedule.append({
            "EQP_ID":        eqp_id,
            "LOT_ID":        lot_id,
            "CARRIER_ID":    unit.get("carrier_id", ""),
            "PLAN_PROD_KEY": ppk,
            "OPER_ID":       oper_id,
            "ST":            proc_time,
            "EQP_MODEL":     row["eqp_model"],
            "SEQ":           row["seq"],
            "START_TM":      start_time,
            "END_TM":        end_time,
            "PROC_TIME":     proc_time,
            "WF_QTY":        wf_qty,
            "LOT_CD":        lot_cd,
            "TEMP":          temp,
            "CONVERSION":    start_time > self.current_time,
            "ABSTRACT":      True,
            "OPER_IN_TIME":  unit["oper_in_time"],
        })

        self._last_assigned = {
            "kind":            "abstract",
            "eqp_id":          eqp_id,
            "lot_id":          lot_id,
            "plan_prod_key":   ppk,
            "eqp_model":       row["eqp_model"],
            "st":              proc_time,
            "wf_qty":          wf_qty,
            "lot_cd":          lot_cd,
            "temp":            temp,
            "conversion":      start_time > self.current_time,
            "start_tm":        start_time,
            "oper_in_time":    unit["oper_in_time"],
            "abs_key":         row["abs_key"],
        }

        self._select_same_time_next_eqp()
        return reward

    def assign_lot(self, eqp_id: str, lot_id: str) -> float:
        """
        목적: 선택된 LOT을 EQP에 배정하고 즉각 보상 반환
        Input:  eqp_id="EQP001", lot_id="LOT001"
        Output: reward (float)
        """
        eqp = self.eqps[eqp_id]
        row, unit = self._find_abstract_unit(lot_id)
        if row is not None:
            return self._assign_abstract_lot(eqp_id, lot_id, row, unit)

        lot = self.lot_pool.get(lot_id)
        if lot is None:
            return -1.0

        cfg = self._reward_cfg
        reward = 0.0

        lot_cd, temp = lot.lot_cd, lot.temp
        if not lot_cd:
            lot_cd, temp = self._lot_cd_temp(lot_id, lot)
        if not self._tool_tracker.can_assign(lot_cd, eqp_id):
            return -1.0

        # 동일 OPER 연속 보너스
        if eqp.prev_oper == lot.oper_id:
            reward += cfg.w_same_oper
        elif eqp.prev_oper is not None:
            eqp.oper_switches += 1
            self.stats["oper_switches"] += 1

        reward += self._same_prod_reward(eqp, lot.plan_prod_key)
        reward += self._pacing_shaping_reward(lot.plan_prod_key, lot.oper_id, lot.wf_qty)

        # LOT 완료 보상
        reward += cfg.w_completion * lot.wf_qty / 25.0

        # 처리시간: (LOT, EQP) 조합 속성 – 단순 LOT 속성이 아님
        proc_time_matrix = self._env_data.get("proc_time_matrix", {})
        proc_time = proc_time_matrix.get((lot_id, eqp_id), lot.processing_time)

        start_time, reward = self._apply_conversion_start(eqp, lot_cd, temp, reward)
        end_time   = start_time + proc_time
        reward += self._plan_hit_reward(lot.plan_prod_key, lot.oper_id, lot.wf_qty, end_time)

        eqp.status       = "busy"
        eqp.current_lot  = lot_id
        eqp.current_oper = lot.oper_id
        eqp.current_prod = lot.plan_prod_key
        eqp.free_at      = end_time
        eqp.prev_oper    = lot.oper_id
        eqp.prev_prod    = lot.plan_prod_key
        eqp.prev_lot_id  = lot_id
        eqp.prev_lot_cd  = lot_cd
        eqp.prev_temp    = temp

        self._tool_tracker.occupy(lot_cd, eqp_id)

        heapq.heappush(self._event_q, (end_time, eqp_id))

        self._in_flight[lot_id] = {
            "plan_prod_key": lot.plan_prod_key,
            "oper_id":       lot.oper_id,
            "seq":           lot.seq,
            "wf_qty":        lot.wf_qty,
            "carrier_id":    lot.carrier_id,
            "end_time":      end_time,
            "lot_cd":        lot_cd,
            "eqp_id":        eqp_id,
        }

        # 선택된 LOT – 모든 EQP 큐에서 제거 (arrange 조합 전체 소멸)
        for eid in self.eqp_queues:
            if lot_id in self.eqp_queues[eid]:
                self.eqp_queues[eid].remove(lot_id)
        del self.lot_pool[lot_id]

        # 완료 수량: 처리 종료(END_TM)가 아닌 투입(START_TM) 시점에 반영
        key = (lot.plan_prod_key, lot.oper_id)
        self.stats["completed_qty"][key] = (
            self.stats["completed_qty"].get(key, 0) + lot.wf_qty
        )

        eqp_model = self._eqp_model_map.get(eqp_id, "A")
        for row in self._initial_arrange:
            if row["lot_id"] == lot_id and row["eqp_id"] == eqp_id:
                eqp_model = row.get("eqp_model") or eqp_model
                break

        self.schedule.append({
            "EQP_ID":        eqp_id,
            "LOT_ID":        lot_id,
            "CARRIER_ID":    lot.carrier_id,
            "PLAN_PROD_KEY": lot.plan_prod_key,
            "OPER_ID":       lot.oper_id,
            "ST":            proc_time,
            "EQP_MODEL":     eqp_model,
            "SEQ":           lot.seq,
            "START_TM":      start_time,
            "END_TM":        end_time,
            "PROC_TIME":     proc_time,
            "WF_QTY":        lot.wf_qty,
            "LOT_CD":        lot_cd,
            "TEMP":          temp,
            "CONVERSION":    start_time > self.current_time,
        })

        assign_st = proc_time
        assign_model = eqp_model
        for row in self._initial_arrange:
            if row["lot_id"] == lot_id and row["eqp_id"] == eqp_id:
                assign_st = row.get("st") or row.get("proc_time") or proc_time
                assign_model = row.get("eqp_model") or assign_model
                break

        self._last_assigned = {
            "kind":            "actual",
            "eqp_id":        eqp_id,
            "lot_id":        lot_id,
            "plan_prod_key": lot.plan_prod_key,
            "st":            assign_st,
            "eqp_model":     assign_model,
            "wf_qty":        lot.wf_qty,
            "lot_cd":        lot_cd,
            "temp":          temp,
            "conversion":    start_time > self.current_time,
            "start_tm":      start_time,
        }

        self._select_same_time_next_eqp()
        return reward

    def _append_initial_history(self):
        """에피소드 시작 시 arrange 전체 상태 스냅샷 (step 0)"""
        self.history.append({
            "step":       0,
            "time":       self.current_time,
            "schedule":   [],
            "completed":  {},
            "idle_total": 0,
            "oper_sw":    0,
            "prod_sw":    0,
            "arrange":           self.get_remaining_arrange_actual(),
            "arrange_actual":    self.get_remaining_arrange_actual(),
            "arrange_abstract":  self.get_abstract_arrange(),
            "wip_waiting":       self._serialize_wip_waiting(),
            "assigned":   None,
            "eqp_states": {
                eid: {
                    "status":       e.status,
                    "current_lot":  e.current_lot,
                    "current_oper": e.current_oper,
                    "current_prod": e.current_prod,
                    "free_at":      e.free_at,
                }
                for eid, e in self.eqps.items()
            },
        })

    def _has_assignable_work(self) -> bool:
        if self.get_remaining_arrange_actual():
            return True
        for eid in self._env_data["eqp_ids"]:
            if self._abstract_assignable_on_eqp(eid):
                return True
        return False

    def _has_pending_processing(self) -> bool:
        if self._in_flight:
            return True
        return any(e.status == "busy" for e in self.eqps.values())

    def is_done(self) -> bool:
        """
        목적: 시뮬레이션 종료 – Actual·추상 투입 가능 조합 없고 처리 중인 재공도 없을 때
        """
        if self._current_eqp is not None:
            return False
        if self._has_pending_processing():
            return False
        return not self._has_assignable_work()

    def save_history_step(
        self,
        arrange_snapshot: Optional[List[dict]] = None,
        arrange_abstract_snapshot: Optional[List[dict]] = None,
        wip_waiting_snapshot: Optional[Dict[str, int]] = None,
    ):
        """
        목적: UI 단계별 재생을 위해 현재 시뮬레이션 상태 스냅숏 저장
        Input:  arrange_snapshot – 배정 직전 Actual 조합
                arrange_abstract_snapshot – 배정 직전 추상 유입 재공
                wip_waiting_snapshot – 배정 직전 대기 WIP (END_TM 반영 후)
        """
        self._step_idx += 1
        assigned = self._last_assigned
        self._last_assigned = None
        if not self._record_history:
            return
        actual = arrange_snapshot if arrange_snapshot is not None else self.get_remaining_arrange_actual()
        abstract = (
            arrange_abstract_snapshot
            if arrange_abstract_snapshot is not None
            else self.get_abstract_arrange()
        )
        self.history.append({
            "step":       self._step_idx,
            "time":       self.current_time,
            "schedule":   list(self.schedule),
            "completed":  dict(self.stats["completed_qty"]),
            "idle_total": self.stats["idle_total"],
            "oper_sw":    self.stats["oper_switches"],
            "prod_sw":    self.stats["prod_switches"],
            "arrange":    actual,
            "arrange_actual":   actual,
            "arrange_abstract": abstract,
            "wip_waiting":      (
                wip_waiting_snapshot
                if wip_waiting_snapshot is not None
                else self._serialize_wip_waiting()
            ),
            "assigned":   assigned,
            "eqp_states": {
                eid: {
                    "status":       e.status,
                    "current_lot":  e.current_lot,
                    "current_oper": e.current_oper,
                    "current_prod": e.current_prod,
                    "free_at":      e.free_at,
                }
                for eid, e in self.eqps.items()
            },
        })

    # ── Bucket(=PPK×MODEL×OPER) feature 텐서 ──────────────────────────────────
    BUCKET_FEATURES = 10

    def get_bucket_features(self) -> np.ndarray:
        """
        Bucket = (oper, ppk, model) 단위 feature 텐서. shape (O, P, K, F).
        채널: 0 valid, 1 wip/total, 2 wip/ppk, 3 min_end_time,
              4 throughput_ratio, 5 same_ppk, 6 prev_takt, 7 post_takt,
              8 self_st(per-wafer), 9 plan_urgency.
        ST = per-wafer 처리시간 → lot 처리시간 = ST × wf_unit.
        """
        data = self._env_data
        cfg = CONFIG.env
        O, P, K = cfg.max_oper_count, cfg.max_prod_count, cfg.max_model_count
        F = self.BUCKET_FEATURES

        eqp_models = data.get("eqp_models", [])
        eqp_model_map = data.get("eqp_model_map", {})
        eqp_oper_cap = data.get("eqp_oper_cap", {})
        route_map = data.get("abstract_route_map", {})
        routes_by = data.get("abstract_routes_by_ppk_oper", {})
        plan_meta = data.get("plan_meta", {})
        n_eqp_per_oper = data.get("n_eqp_per_oper", {})
        flow_prev = data.get("flow_prev", {})
        flow_post = data.get("flow_post", {})
        max_route_st = max(data.get("max_route_st", 1), 1)
        wf_unit = max(data.get("max_wf_qty", 1), 1)
        completed = self.stats["completed_qty"]

        feats = np.zeros((O, P, K, F), dtype=np.float32)

        # (oper, model) → 처리 가능 EQP 목록 (min_end_time / throughput용)
        eqps_by_om: Dict[tuple, List[str]] = {}
        for e, model in eqp_model_map.items():
            for op in eqp_oper_cap.get(e, []):
                eqps_by_om.setdefault((op, model), []).append(e)

        # WIP 집계 (ppk, oper)
        wip_po: Dict[tuple, float] = {}
        ppk_wip: Dict[str, float] = {}
        total_wip = 0.0
        for key, q in self.get_wip_waiting().items():
            ppk, op = key.split("|", 1)
            wip_po[(ppk, op)] = wip_po.get((ppk, op), 0.0) + q
            ppk_wip[ppk] = ppk_wip.get(ppk, 0.0) + q
            total_wip += q
        total_wip = max(total_wip, 1.0)

        max_gantt_end = max((r["END_TM"] for r in self.schedule), default=0)
        T_avail = max(self.soft_cutoff - self.current_time, 1)
        max_takt = max(T_avail * wf_unit, 1.0)
        last_ppk = (
            self._last_assigned.get("plan_prod_key") if self._last_assigned else None
        )

        def st_per_wafer(ppk: str, op: Optional[str], model: str) -> Optional[float]:
            if op is None:
                return None
            st = route_map.get((ppk, op, model))
            if st is not None:
                return float(st)
            lst = routes_by.get((ppk, op))
            if lst:
                return sum(s for _, s in lst) / len(lst)
            return None

        def eff_takt(ppk: str, op: Optional[str]) -> float:
            """수요 페이싱 + capacity. per-lot 간격(분)."""
            if op is None:
                return 0.0
            pm = plan_meta.get((ppk, op))
            q_plan = max(
                (pm["d0_plan_qty"] if pm else 0) - completed.get((ppk, op), 0), 1,
            )
            demand_takt = T_avail / q_plan
            lst = routes_by.get((ppk, op))
            spw = (sum(s for _, s in lst) / len(lst)) if lst else None
            n = max(n_eqp_per_oper.get(op, 0), 1)
            cap_takt = (spw / n) if spw is not None else 0.0
            return max(demand_takt, cap_takt) * wf_unit

        oper_ids = data["oper_ids"]
        prod_keys = data["prod_keys"]
        for oi in range(min(O, len(oper_ids))):
            op = oper_ids[oi]
            for pi in range(min(P, len(prod_keys))):
                ppk = prod_keys[pi]
                is_step = (
                    (ppk, op) in routes_by
                    or (ppk, op) in plan_meta
                    or (ppk, op) in wip_po
                )
                if not is_step:
                    continue
                wip_q = wip_po.get((ppk, op), 0.0)
                ppk_total = max(ppk_wip.get(ppk, 0.0), 1.0)
                urgency = 0.0
                pm = plan_meta.get((ppk, op))
                if pm:
                    plan_qty = max(pm["d0_plan_qty"], 1)
                    gap = max(plan_qty - completed.get((ppk, op), 0), 0)
                    urgency = min(gap / T_avail / plan_qty, 1.0)
                same = 1.0 if ppk == last_ppk else 0.0
                prev_takt = eff_takt(ppk, flow_prev.get(ppk, {}).get(op)) / max_takt
                post_takt = eff_takt(ppk, flow_post.get(ppk, {}).get(op)) / max_takt

                for mi in range(min(K, len(eqp_models))):
                    model = eqp_models[mi]
                    eqp_list = eqps_by_om.get((op, model))
                    if not eqp_list:
                        continue  # 이 model로 처리 불가 → invalid 패딩(0)
                    st = st_per_wafer(ppk, op, model)
                    min_end = min(self.eqps[e].free_at for e in eqp_list)
                    proc_full = (st * wf_unit) if st is not None else 0.0
                    denom = max(max_gantt_end, min_end + proc_full, 1.0)

                    feats[oi, pi, mi, 0] = 1.0
                    feats[oi, pi, mi, 1] = wip_q / total_wip
                    feats[oi, pi, mi, 2] = wip_q / ppk_total
                    feats[oi, pi, mi, 3] = min(min_end / max(self.sim_end, 1), 1.0)
                    feats[oi, pi, mi, 4] = max(wip_q, 0.0) / denom
                    feats[oi, pi, mi, 5] = same
                    feats[oi, pi, mi, 6] = min(prev_takt, 1.0)
                    feats[oi, pi, mi, 7] = min(post_takt, 1.0)
                    feats[oi, pi, mi, 8] = (st / max_route_st) if st is not None else 0.0
                    feats[oi, pi, mi, 9] = urgency
        return feats

    # ── 관측 벡터 생성 (Global + Bucket + EQP local + Context) ────────────────

    def get_observation(self) -> np.ndarray:
        """관측: Global(6) + Bucket(O×P×K×F) + EQP local(M×5) + Context(4)."""
        data = self._env_data
        cfg = CONFIG.env
        O, P, M = cfg.max_oper_count, cfg.max_prod_count, cfg.max_eqp_count
        oper_idx = data["oper_idx"]
        prod_idx = data["prod_idx"]
        eqp_idx = data.get("eqp_idx", {eid: i for i, eid in enumerate(data["eqp_ids"])})
        lot_cd_idx = data.get("lot_cd_idx", {})
        temp_idx = data.get("temp_idx", {})
        total_plan = max(sum(p["d0_plan_qty"] for p in data["plan"]), 1)
        initial_lot_count = max(len(data["lots"]), 1)

        bucket = self.get_bucket_features().flatten()

        eqp_local = np.zeros((M, 5), dtype=np.float32)
        conv_eqps = 0
        for eqp_id, eqp in self.eqps.items():
            ei = eqp_idx.get(eqp_id, -1)
            if ei < 0 or ei >= M:
                continue
            eqp_local[ei, 0] = float(eqp.status == "idle")
            eqp_local[ei, 1] = float(eqp.status == "busy")
            eqp_local[ei, 2] = encode_normalized(eqp.prev_lot_cd, lot_cd_idx, max(len(lot_cd_idx), 1))
            eqp_local[ei, 3] = encode_normalized(eqp.prev_temp, temp_idx, max(len(temp_idx), 1))
            rem = max(eqp.free_at - self.current_time, 0)
            eqp_local[ei, 4] = min(rem / max(self.sim_end, 1), 1.0)
            if rem > 0 and eqp.status == "idle" and eqp.prev_lot_cd is not None:
                conv_eqps += 1

        group_global = np.zeros(6, dtype=np.float32)
        group_global[0] = min(self.current_time / max(self.sim_end, 1), 1.0)
        group_global[1] = min(
            max(self.soft_cutoff - self.current_time, 0) / max(self.soft_cutoff, 1), 1.0,
        )
        group_global[2] = min(len(self.lot_pool) / initial_lot_count, 1.0)
        group_global[3] = min(sum(self.stats["completed_qty"].values()) / total_plan, 1.0)
        group_global[4] = conv_eqps / max(len(self.eqps), 1)
        group_global[5] = min(self._tool_tracker.utilization(), 1.0)

        context = np.zeros(4, dtype=np.float32)
        if self._last_assigned:
            la = self._last_assigned
            context[0] = encode_normalized(
                la.get("plan_prod_key"), prod_idx, P,
            )
            oper_guess = None
            for ld in data.get("lots", []):
                if ld["lot_id"] == la.get("lot_id"):
                    oper_guess = ld.get("oper_id")
                    break
            context[1] = encode_normalized(oper_guess, oper_idx, O)
            context[2] = encode_normalized(la.get("eqp_id"), eqp_idx, M)
            context[3] = encode_normalized(la.get("lot_cd"), lot_cd_idx, max(len(lot_cd_idx), 1))

        obs = np.concatenate([
            group_global,
            bucket,
            eqp_local.flatten(),
            context,
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
