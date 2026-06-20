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
    idle_accum:   int            = 0
    oper_switches:int            = 0
    prod_switches:int            = 0


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

    def __init__(self, env_data: dict, reward_cfg: RewardConfig = None):
        self._env_data   = env_data
        self._reward_cfg = reward_cfg or CONFIG.reward
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

        self.eqps: Dict[str, Equipment] = {
            eid: Equipment(eqp_id=eid) for eid in data["eqp_ids"]
        }

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
            "completed_qty": {},   # {(prod, oper): qty}
        }

        self.history:    List[dict] = []
        self._step_idx   = 0
        self._last_assigned: Optional[dict] = None
        self._current_eqp: Optional[str] = None
        self._advance_to_next_decision()
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
        models = self._env_data.get("oper_eqp_models", {}).get(next_oper, ["A"])

        for model in models:
            proc_time = proc_map.get((next_oper, model), 60)
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
            rows.append(dict(row))
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
                    "is_abstract":     True,
                    "abs_key":         row["abs_key"],
                })
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

        idle_duration = self.current_time - eqp.free_at
        if idle_duration > 0:
            eqp.idle_accum += idle_duration
            self.stats["idle_total"] += idle_duration
            reward += cfg.w_idle_per_min * idle_duration

        oper_id = row["oper_id"]
        ppk     = row["plan_prod_key"]

        if eqp.prev_oper == oper_id:
            reward += cfg.w_same_oper
        elif eqp.prev_oper is not None:
            eqp.oper_switches += 1
            self.stats["oper_switches"] += 1

        if eqp.prev_prod == ppk:
            reward += cfg.w_same_prod
        elif eqp.prev_prod is not None:
            eqp.prod_switches += 1
            self.stats["prod_switches"] += 1

        wf_qty    = row["wf_qty"]
        proc_time = row["proc_time"]
        reward += cfg.w_completion * wf_qty / 25.0

        start_time = self.current_time
        end_time   = start_time + proc_time

        eqp.status       = "busy"
        eqp.current_lot  = lot_id
        eqp.current_oper = oper_id
        eqp.current_prod = ppk
        eqp.free_at      = end_time
        eqp.prev_oper    = oper_id
        eqp.prev_prod    = ppk
        eqp.prev_lot_id  = lot_id

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

        # Idle 패널티
        idle_duration = self.current_time - eqp.free_at
        if idle_duration > 0:
            eqp.idle_accum += idle_duration
            self.stats["idle_total"] += idle_duration
            reward += cfg.w_idle_per_min * idle_duration

        # 동일 OPER 연속 보너스
        if eqp.prev_oper == lot.oper_id:
            reward += cfg.w_same_oper
        elif eqp.prev_oper is not None:
            eqp.oper_switches += 1
            self.stats["oper_switches"] += 1

        # 동일 제품 연속 보너스
        if eqp.prev_prod == lot.plan_prod_key:
            reward += cfg.w_same_prod
        elif eqp.prev_prod is not None:
            eqp.prod_switches += 1
            self.stats["prod_switches"] += 1

        # LOT 완료 보상
        reward += cfg.w_completion * lot.wf_qty / 25.0

        # 처리시간: (LOT, EQP) 조합 속성 – 단순 LOT 속성이 아님
        proc_time_matrix = self._env_data.get("proc_time_matrix", {})
        proc_time = proc_time_matrix.get((lot_id, eqp_id), lot.processing_time)

        start_time = self.current_time
        end_time   = start_time + proc_time

        eqp.status       = "busy"
        eqp.current_lot  = lot_id
        eqp.current_oper = lot.oper_id
        eqp.current_prod = lot.plan_prod_key
        eqp.free_at      = end_time
        eqp.prev_oper    = lot.oper_id
        eqp.prev_prod    = lot.plan_prod_key
        eqp.prev_lot_id  = lot_id

        heapq.heappush(self._event_q, (end_time, eqp_id))

        self._in_flight[lot_id] = {
            "plan_prod_key": lot.plan_prod_key,
            "oper_id":       lot.oper_id,
            "seq":           lot.seq,
            "wf_qty":        lot.wf_qty,
            "carrier_id":    lot.carrier_id,
            "end_time":      end_time,
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

    # ── 관측 벡터 생성 (이분 그래프 WIP x EQP 엣지 노드 포함) ────────────────

    def get_observation(self) -> np.ndarray:
        """
        목적: 이분 그래프 기반 고정 크기 관측 벡터 생성
        Input:  없음
        Output: np.ndarray shape=(obs_dim,) dtype=float32

        이분 그래프 구조:
          Left (WIP 그룹)  -- Group E (엣지 노드) --  Right (EQP)
          (OPER, PROD)         WIP x EQP 조합별         EQP 가동 현황
          수량 / 우선순위        평균처리시간 / 호환수      Idle / 잔여시간

        처리시간은 LOT 단독이 아닌 (LOT, EQP) 조합 속성 -> Group E에서 표현

        벡터 구조 (O=max_oper_count, P=max_prod_count, M=max_eqp_count):
          Group A [O x P x 3]: WIP 노드  - (OPER, PROD)별 대기 재공 집계
            . wip_lot_count_norm    대기 LOT 수 비율
            . wip_total_qty_norm    총 웨이퍼 수량 비율
            . wip_avg_priority_norm 평균 우선순위
          Group B [O x 4]:     EQP 노드  - OPER 처리 능력별 설비 집계
            . eqp_capable_norm      처리 가능 EQP 수
            . eqp_idle_ratio        현재 idle 비율
            . eqp_avg_remaining     평균 잔여 처리시간
            . eqp_switch_rate       평균 공정전환 비율
          Group E [O x M x 2]: WIP x EQP 결합 노드 (엣지 피처)
            . avg_proc_time_norm    (OPER, EQP) 조합별 평균 처리시간  <- 핵심
            . n_compatible_norm     배정 가능 LOT 수 비율
          Group C [O x P x 2]: 계획 노드 - (OPER, PROD) 계획 달성 현황
            . achievement_rate      달성률 [0,1]
            . plan_priority_norm    우선순위 비율
          Group D [6]:         컨텍스트  - 현재 결정 EQP 로컬 상태

        총 크기 = O*P*3 + O*4 + O*M*2 + O*P*2 + 6
               = O*P*5 + O*(4 + M*2) + 6
        """
        data     = self._env_data
        cfg      = CONFIG.env
        O        = cfg.max_oper_count
        P        = cfg.max_prod_count
        M        = cfg.max_eqp_count
        oper_idx = data["oper_idx"]
        prod_idx = data["prod_idx"]
        eqp_idx  = data.get("eqp_idx", {eid: i for i, eid in enumerate(data["eqp_ids"])})

        total_lots       = max(len(data["lots"]), 1)
        max_proc         = max(data.get("max_proc_time", 1), 1)
        max_qty          = max(data.get("max_wf_qty", 1), 1)
        max_eqp_cnt      = max(len(self.eqps), 1)
        total_plan       = max(sum(p["d0_plan_qty"] for p in data["plan"]), 1)
        max_switches_val = max((e.oper_switches for e in self.eqps.values()), default=0)
        max_switches_val = max(max_switches_val, 1)
        proc_time_matrix = data.get("proc_time_matrix", {})
        eqp_oper_cap     = data.get("eqp_oper_cap", {})

        # ── Group A: WIP 노드 집계 ────────────────────────────────────────────
        A_count    = np.zeros((O, P), dtype=np.float32)
        A_qty      = np.zeros((O, P), dtype=np.float32)
        A_priority = np.zeros((O, P), dtype=np.float32)

        for lot in self.lot_pool.values():
            oi = oper_idx.get(lot.oper_id, -1)
            pi = prod_idx.get(lot.plan_prod_key, -1)
            if 0 <= oi < O and 0 <= pi < P:
                A_count[oi, pi]    += 1
                A_qty[oi, pi]      += lot.wf_qty
                A_priority[oi, pi] += lot.priority

        A_mask = A_count > 0
        groupA = np.stack([
            A_count / total_lots,
            A_qty   / (max_qty * total_lots),
            np.where(A_mask, A_priority / np.maximum(A_count, 1), 0.0) / 5.0,
        ], axis=-1)  # (O, P, 3)

        # ── Group B: EQP 노드 집계 ────────────────────────────────────────────
        B_capable   = np.zeros(O, dtype=np.float32)
        B_idle      = np.zeros(O, dtype=np.float32)
        B_remaining = np.zeros(O, dtype=np.float32)
        B_switches  = np.zeros(O, dtype=np.float32)

        for eqp_id, eqp in self.eqps.items():
            capable_opers = set(eqp_oper_cap.get(eqp_id, []))
            if eqp.current_oper:
                capable_opers.add(eqp.current_oper)
            for lid in self.eqp_queues.get(eqp_id, []):
                if lid in self.lot_pool:
                    capable_opers.add(self.lot_pool[lid].oper_id)
            rem = max(eqp.free_at - self.current_time, 0)
            for oper_id in capable_opers:
                oi = oper_idx.get(oper_id, -1)
                if 0 <= oi < O:
                    B_capable[oi]   += 1
                    B_idle[oi]      += float(eqp.status == "idle")
                    B_remaining[oi] += rem
                    B_switches[oi]  += eqp.oper_switches

        groupB = np.stack([
            B_capable   / max_eqp_cnt,
            B_idle      / np.maximum(B_capable, 1),
            B_remaining / (self.sim_end * np.maximum(B_capable, 1)),
            B_switches  / (max_switches_val * np.maximum(B_capable, 1)),
        ], axis=-1)  # (O, 4)

        # ── Group E: WIP x EQP 결합 노드 (엣지 피처) ─────────────────────────
        # (LOT, EQP) 조합별 처리시간을 (OPER, EQP) 그룹으로 집계
        # -> 처리시간이 EQP마다 다른 현실을 이분 그래프 엣지 속성으로 표현
        E_proc_sum = np.zeros((O, M), dtype=np.float32)
        E_count    = np.zeros((O, M), dtype=np.float32)

        for eqp_id, eqp in self.eqps.items():
            ei = eqp_idx.get(eqp_id, -1)
            if ei < 0 or ei >= M:
                continue
            for lot_id in self.eqp_queues.get(eqp_id, []):
                if lot_id not in self.lot_pool:
                    continue
                lot = self.lot_pool[lot_id]
                oi  = oper_idx.get(lot.oper_id, -1)
                if 0 <= oi < O:
                    pt = proc_time_matrix.get((lot_id, eqp_id), lot.processing_time)
                    E_proc_sum[oi, ei] += pt
                    E_count[oi, ei]    += 1

        E_avg_proc = np.where(E_count > 0,
                              E_proc_sum / np.maximum(E_count, 1),
                              0.0)
        groupE = np.stack([
            E_avg_proc / max_proc,      # (OPER, EQP) 조합별 평균 처리시간 (정규화)
            E_count    / total_lots,    # 배정 가능 LOT 수 비율
        ], axis=-1)  # (O, M, 2)

        # ── Group C: 계획 노드 ────────────────────────────────────────────────
        C_achieve  = np.zeros((O, P), dtype=np.float32)
        C_priority = np.zeros((O, P), dtype=np.float32)

        for p in data["plan"]:
            oi = oper_idx.get(p["oper_id"], -1)
            pi = prod_idx.get(p["plan_prod_key"], -1)
            if 0 <= oi < O and 0 <= pi < P:
                key  = (p["plan_prod_key"], p["oper_id"])
                done = self.stats["completed_qty"].get(key, 0)
                C_achieve[oi, pi]  = min(done / max(p["d0_plan_qty"], 1), 1.0)
                C_priority[oi, pi] = p["priority"] / 5.0

        groupC = np.stack([C_achieve, C_priority], axis=-1)  # (O, P, 2)

        # ── Group D: 현재 결정 컨텍스트 ──────────────────────────────────────
        groupD = np.zeros(6, dtype=np.float32)
        groupD[0] = min(self.current_time / max(self.sim_end, 1), 1.0)
        groupD[1] = min(sum(self.stats["completed_qty"].values()) / total_plan, 1.0)
        groupD[4] = self.stats["oper_switches"] / max(total_lots, 1)
        groupD[5] = len(self.lot_pool) / total_lots

        eqp_id = self._current_eqp
        if eqp_id:
            eqp = self.eqps[eqp_id]
            groupD[2] = encode_normalized(eqp.prev_oper, oper_idx, O)
            groupD[3] = encode_normalized(eqp.prev_prod, prod_idx, P)

        # ── 연결 (flatten & concat) ───────────────────────────────────────────
        obs = np.concatenate([
            groupA.flatten(),  # O*P*3
            groupB.flatten(),  # O*4
            groupE.flatten(),  # O*M*2  <- WIP x EQP 결합 노드
            groupC.flatten(),  # O*P*2
            groupD,            # 6
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
