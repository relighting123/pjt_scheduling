"""
simulation/simulator.py – 이산 사건 시뮬레이션(DES) 엔진

RL 환경의 내부 시뮬레이터로, EQP 상태 및 LOT 배정 이력을 관리합니다.
"""
import copy
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import CONFIG, RewardConfig
from simulation.events import (
    EVENT_CONV_ASSIGNED,
    EVENT_CONV_END,
    EVENT_DOWN_END,
    EVENT_DOWN_START,
    EVENT_FORCED_CONV_END,
    EVENT_JOB_ASSIGNED,
    EVENT_IDLE,
    EVENT_IDLE_DECISION,
    EVENT_MOVE_OUT,
    EVENT_PRIORITY,
    EVENT_PROCESS_END,
)
from utils.helpers import effective_proc_time


@dataclass(order=True)
class SimEvent:
    """DES 이벤트 큐 항목. (time, priority, seq) 순으로 heap 정렬."""
    time: int
    priority: int
    seq: int
    kind: str = field(compare=False, default="")
    eqp_id: str = field(compare=False, default="")


class ToolTracker:
    """LOT_CD × EQP_MODEL_CD 동시 가공 상한 추적."""

    def __init__(self, capacity: dict, eqp_model_map: dict):
        self._capacity = dict(capacity)
        self._eqp_model_map = eqp_model_map
        self._busy: dict = {}

    def _model_for(self, eqp_id: str) -> str:
        try:
            return self._eqp_model_map[eqp_id]
        except KeyError as exc:
            raise ValueError(f"EQP_MODEL_CD 누락: {eqp_id}") from exc

    def can_assign(self, lot_cd: str, eqp_id: str) -> bool:
        model = self._model_for(eqp_id)
        cap = self._capacity.get((lot_cd, model), 999)
        return self._busy.get((lot_cd, model), 0) < cap

    def remaining(self, lot_cd: str, eqp_id: str) -> int:
        """추가로 점유 가능한 tool 수 = MAX_TOOL − 현재 사용 중(busy).

        벌크 배정 시 블록 크기를 '잔여(추가 가용)' 슬롯으로만 제한하기 위함.
        총량(MAX_TOOL)이 아니라 이미 사용 중인 분을 제외한 여분만 반환한다.
        """
        model = self._model_for(eqp_id)
        cap = self._capacity.get((lot_cd, model), 999)
        return max(cap - self._busy.get((lot_cd, model), 0), 0)

    def occupy(self, lot_cd: str, eqp_id: str) -> None:
        model = self._model_for(eqp_id)
        key = (lot_cd, model)
        self._busy[key] = self._busy.get(key, 0) + 1

    def release(self, lot_cd: str, eqp_id: str) -> None:
        model = self._model_for(eqp_id)
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


# --- 도메인 객체 ---

@dataclass
class Lot:
    lot_id:          str
    carrier_id:      str
    PLAN_PROD_ATTR_VAL:   str
    oper_id:         str
    seq:             int
    wf_qty:          int
    processing_time: int   # 장당 ST(분/장). 실제 소요는 proc_time_matrix × wf_qty
    priority:        int   = 1
    original_eqp:    str   = ""
    parent_lot_id:   str   = ""
    logical_lot_id:  str   = ""
    lot_cd:          str   = ""
    temp:            str   = ""
    lot_stat_cd:     str   = "WAIT"


@dataclass
class Equipment:
    eqp_id:       str
    status:       str            = "idle"   # "idle" | "busy" | "converting" | "down"
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


# --- 시뮬레이터 ---

class SchedulingSimulator:
    """
    이산 사건 시뮬레이터. EQP가 idle이 되는 시점마다 에이전트 결정을 요청합니다.

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

    def __init__(self, env_data: dict, reward_cfg: RewardConfig = None, record_history: bool = True, record_event_log: bool = True):
        self._env_data   = env_data
        self._reward_cfg = reward_cfg or CONFIG.reward
        self._record_history = record_history
        self._record_event_log = record_event_log
        self.reset()

    # --- 초기화 ---

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
        self._max_conversions: Optional[int] = data.get("max_conversions", CONFIG.env.max_conversions)
        self._max_conversions_per_eqp: Optional[int] = data.get(
            "max_conversions_per_eqp", CONFIG.env.max_conversions_per_eqp,
        )
        # 전환 그룹 제약: (lot_cd, temp) → group_id. 같은 그룹끼리만 전환 허용.
        self._conv_group_map: Dict[Tuple[str, str], str] = dict(data.get("conv_group_map", {}))
        self._conv_groups_active: bool = bool(self._conv_group_map)
        self._lot_attrs: Dict[str, dict] = dict(data.get("lot_attrs", {}))
        self._initial_start: Dict[str, int] = {}

        # 외부 확정 EQP 전환 계획(진행 중/예정, eqp_conv_plan 입력) — EQP별 start_min 오름차순
        self._forced_conv_queue: Dict[str, List[dict]] = {}
        for row in data.get("eqp_conv_plan", []):
            self._forced_conv_queue.setdefault(row["eqp_id"], []).append(dict(row))
        for _queue in self._forced_conv_queue.values():
            _queue.sort(key=lambda r: r["start_min"])

        # EQP PM/개조 등 다운타임 구간(eqp_down 입력) — EQP별 down_start_min 오름차순
        # down_end_min이 None이면 무제한 다운(해당 EQP는 이후 배정 대상에서 영구 제외)
        self._down_queue: Dict[str, List[dict]] = {}
        for row in data.get("eqp_down", []):
            self._down_queue.setdefault(row["eqp_id"], []).append(dict(row))
        for _queue in self._down_queue.values():
            _queue.sort(key=lambda r: r["down_start_min"])

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
                PLAN_PROD_ATTR_VAL=ld["PLAN_PROD_ATTR_VAL"],
                oper_id=ld["oper_id"],
                seq=ld["seq"],
                wf_qty=ld["wf_qty"],
                processing_time=ld["processing_time"],
                priority=ld.get("priority", 1),
                original_eqp=ld.get("original_eqp", ""),
                parent_lot_id=ld.get("parent_lot_id", ""),
                logical_lot_id=ld.get("logical_lot_id", ld["lot_id"]),
                lot_cd=ld.get("lot_cd", ""),
                temp=ld.get("temp", ""),
                lot_stat_cd=ld.get("lot_stat_cd", "WAIT"),
            )

        # LOT_STAT_CD != WAIT LOT: 지정 EQP_ID에 입력 순서대로 강제 배정(자유 스케줄링 불가)
        self._eqp_forced_queue: Dict[str, List[str]] = {
            eid: list(lot_ids)
            for eid, lot_ids in data.get("eqp_forced_queue", {}).items()
        }
        # LOAD/RESV/SELE 등 가공 슬롯 대기 — reset 시 장비에 선부착, idle 시 자동 투입
        self._eqp_staged_forced: Dict[str, List[str]] = {}

        self.eqp_queues: Dict[str, List[str]] = {
            eid: list(lots)
            for eid, lots in data["eqp_lot_map"].items()
        }

        self._initial_arrange: List[dict] = list(data.get("arrange_actual_table", []))
        self._eqp_model_map: Dict[str, str] = dict(data.get("eqp_model_map", {}))
        self._abstract_template: List[dict] = copy.deepcopy(
            data.get("abstract_inventory", [])
        )
        self._wip_pool: Dict[Tuple[str, str], dict] = copy.deepcopy(
            data.get("abstract_wip_init", {})
        )
        self._wip_lot_meta: Dict[str, dict] = copy.deepcopy(
            data.get("abstract_lot_meta", {})
        )
        self._initial_lot_ids: set = set(data.get("abstract_lot_meta", {}))
        self._in_flight: Dict[str, dict] = {}
        self._inject_deadline: Dict[str, int] = dict(data.get("lot_inject_deadline", {}))

        self._eqp_selection: str = data.get("eqp_selection", "order")
        self._termination_mode: str = data.get("termination_mode", "all_wip")
        self._enable_wip_inflow: bool = bool(data.get("enable_wip_inflow", True))
        self._initial_wip_lot_keys: set = {
            (lid, meta.get("PLAN_PROD_ATTR_VAL"), meta.get("oper_id"))
            for lid, meta in data.get("abstract_lot_meta", {}).items()
        }

        self._event_q: List[SimEvent] = []
        self._event_seq: int = 0
        self.event_log: List[dict] = []
        self._pending_step_events: List[dict] = []
        self._eqp_pending_assign: Dict[str, dict] = {}
        for eid in data["eqp_ids"]:
            self._push_event(0, EVENT_IDLE_DECISION, eid)

        self.schedule:   List[dict] = []
        self.conversion_plans: List[dict] = []
        self.down_windows: List[dict] = []
        self.stats = {
            "idle_total":    0,
            "oper_switches": 0,
            "prod_switches": 0,
            "conversions":   0,
            "completed_qty": {},   # {(prod, oper): qty} — 배정 시점 증가(완료+처리중 포함)
        }

        self.history:    List[dict] = []
        self._step_idx   = 0
        self._last_assigned: Optional[dict] = None
        self._last_decision_assignment: Optional[dict] = None
        # 디버그용 리워드 항목별 분해 (추론 decision_log에서만 소비)
        self._cur_conv_terms: Dict[str, float] = {}
        self._last_reward_breakdown: Dict[str, float] = {}
        self._current_eqp: Optional[str] = None
        self._earliest_st_pick: Optional[Tuple[str, str]] = None
        self._initial_wip_total: int = sum(
            pool.get("wip_qty_init", pool.get("wip_qty", 0))
            for pool in self._wip_pool.values()
        )

        # 성능 최적화: 상태 버전 기반 캐시
        self._state_version: int = 0
        self._feasible_cache: Dict[str, tuple] = {}          # {eqp_id: (version, List[int])}
        self._wip_waiting_cache: Optional[Dict[str, int]] = None
        self._wip_waiting_version: int = -1
        self._bucket_feats_cache: Optional[np.ndarray] = None
        self._bucket_feats_state: tuple = (-1, None)         # (version, current_eqp)
        self._eqps_by_om: Dict[tuple, List[str]] = self._build_eqps_by_om()

        self._apply_eqp_initial_state(data.get("eqp_initial_state", []))
        self._prebind_forced_carriers()
        self._advance_to_next_decision()
        if self._record_history:
            self._append_initial_history()

    # --- 성능 최적화 헬퍼 ---

    def _build_eqps_by_om(self) -> Dict[tuple, List[str]]:
        """(oper, eqp_model) → EQP 목록. reset 시 1회 계산 (정적 구조)."""
        result: Dict[tuple, List[str]] = {}
        eqp_oper_cap = self._env_data.get("eqp_oper_cap", {})
        for eid, model in self._eqp_model_map.items():
            for op in eqp_oper_cap.get(eid, []):
                result.setdefault((op, model), []).append(eid)
        return result

    def _invalidate_caches(self) -> None:
        """상태 변경 시 버전 증가 → feasible/wip/bucket 캐시 자동 무효화."""
        self._state_version += 1
        self._wip_waiting_cache = None

    # --- 이벤트 큐 / 로그 ---

    def _push_event(self, time: int, kind: str, eqp_id: str = "") -> None:
        priority = EVENT_PRIORITY.get(kind, 5)
        heapq.heappush(
            self._event_q,
            SimEvent(time, priority, self._event_seq, kind, eqp_id),
        )
        self._event_seq += 1

    def _emit_event(self, kind: str, eqp_id: str = "", *, event_time: Optional[int] = None, **payload: Any) -> None:
        if not self._record_event_log:
            return
        record = {
            "time": self.current_time if event_time is None else event_time,
            "kind": kind,
            "eqp_id": eqp_id,
            **payload,
        }
        self.event_log.append(record)
        self._pending_step_events.append(record)

    def _pop_event_batch(self) -> List[SimEvent]:
        """이벤트 큐에서 동일 시각 배치를 꺼냄."""
        if not self._event_q:
            return []
        next_time = self._event_q[0].time
        batch: List[SimEvent] = []
        while self._event_q and self._event_q[0].time == next_time:
            batch.append(heapq.heappop(self._event_q))
        return batch

    # --- DES 이벤트 핸들러 ---

    def _on_process_end(self, eqp_id: str) -> None:
        """공정 완료 시 MOVE_OUT (다음 공정 유입 포함, 동일 시각)."""
        eqp = self.eqps.get(eqp_id)
        if not eqp or eqp.status != "busy":
            return
        lot_id = eqp.current_lot
        meta: dict = {}
        lot_cd = ""
        if lot_id:
            meta = self._in_flight.get(lot_id, {})
            lot_cd = meta.get("lot_cd", "")
            next_wip = self._inject_wip_after_complete(lot_id, self.current_time)
            move_payload: Dict[str, Any] = {
                "lot_id": lot_id,
                "lot_cd": lot_cd,
                "PLAN_PROD_ATTR_VAL": meta.get("PLAN_PROD_ATTR_VAL", ""),
                "oper_id": meta.get("oper_id", ""),
            }
            if next_wip:
                move_payload["next_oper_id"] = next_wip["oper_id"]
                move_payload["next_plan_prod_key"] = next_wip["PLAN_PROD_ATTR_VAL"]
                move_payload["next_oper_in_time"] = next_wip.get("oper_in_time")
            self._emit_event(EVENT_MOVE_OUT, eqp_id, **move_payload)
        eqp.status = "idle"
        eqp.current_lot = None
        eqp.current_oper = None
        eqp.current_prod = None
        eqp.free_at = self.current_time
        self._invalidate_caches()
        if self._try_assign_staged_forced(eqp_id):
            return

    def _on_idle_decision(self, eqp_id: str) -> None:
        """idle EQP → 에이전트 배정 결정 시점 (event_log: IDLE)."""
        eqp = self.eqps.get(eqp_id)
        if not eqp or eqp.status != "idle":
            return
        if self._apply_due_down(eqp_id):
            return
        if self._apply_due_forced_conv(eqp_id):
            return
        if self.get_feasible_ppk_oper(eqp_id):
            self._emit_event(EVENT_IDLE, eqp_id, eqp_status="idle")
        else:
            self._schedule_wait_event(eqp_id, self.current_time)

    def _apply_due_down(self, eqp_id: str) -> bool:
        """eqp_down 입력 반영. 시작 시각이 되면 EQP를 down 상태로 전환.

        아직 시작 전이면 정확한 시각에 재확인 이벤트만 예약하고, 그 사이에는
        정상적으로 idle 배정을 진행한다(진행 중이던 가공은 선점하지 않음).
        down_end_min이 None이면 무제한 다운으로 이후 배정 대상에서 영구 제외된다.

        가공이 끝날 때까지 적용이 미뤄진 사이 down_end_min이 이미 지나버린
        경우(비선점이라 전체 구간이 통째로 흘러간 경우)는 과거 시각으로 이벤트를
        예약하면 시뮬 시계가 역행하므로, 그 구간은 건너뛰고 다음 큐 항목을 본다.
        """
        queue = self._down_queue.get(eqp_id)
        while queue:
            row = queue[0]
            if row["down_start_min"] > self.current_time:
                if not row.get("_wake_scheduled"):
                    row["_wake_scheduled"] = True
                    self._push_event(row["down_start_min"], EVENT_IDLE_DECISION, eqp_id)
                return False
            queue.pop(0)
            down_end = row["down_end_min"]
            if down_end is not None and down_end <= self.current_time:
                # 비선점 정책으로 적용이 지연되는 사이 구간 전체가 이미 지나갔다 — 스킵.
                continue
            eqp = self.eqps[eqp_id]
            eqp.status = "down"
            eqp.free_at = self.current_time
            self.down_windows.append({
                "eqp_id":        eqp_id,
                "down_start_min": self.current_time,
                "down_end_min":  down_end,
            })
            self._emit_event(EVENT_DOWN_START, eqp_id, down_end_min=down_end)
            if down_end is not None:
                self._push_event(down_end, EVENT_DOWN_END, eqp_id)
            self._invalidate_caches()
            return True
        return False

    def _apply_due_forced_conv(self, eqp_id: str) -> bool:
        """eqp_conv_plan 입력 반영. 시작 시각이 되면 EQP를 강제 전환 상태로 전환.

        아직 시작 전이면 정확한 시각에 재확인 이벤트만 예약한다(진행 중이던 가공은
        선점하지 않음). start_min이 이미 지난 경우(현재 시각 이하)는 reset 직후
        첫 idle 결정 시점에 바로 걸려 즉시 전환이 시작된다.
        """
        queue = self._forced_conv_queue.get(eqp_id)
        if not queue:
            return False
        row = queue[0]
        if row["start_min"] > self.current_time:
            if not row.get("_wake_scheduled"):
                row["_wake_scheduled"] = True
                self._push_event(row["start_min"], EVENT_IDLE_DECISION, eqp_id)
            return False
        queue.pop(0)
        eqp = self.eqps[eqp_id]
        conv_start = self.current_time
        conv_end = conv_start + self._conversion_minutes
        from_lot_cd = row["from_lot_cd"] or eqp.prev_lot_cd or ""
        from_temp = row["from_temp"] or eqp.prev_temp or ""
        eqp.status = "converting"
        eqp.free_at = conv_end
        eqp.prev_lot_cd = row["to_lot_cd"]
        eqp.prev_temp = row["to_temp"]
        eqp.conversion_count += 1
        self.stats["conversions"] += 1
        self.conversion_plans.append({
            "eqp_id":        eqp_id,
            "eqp_model_cd":  self._eqp_model_map.get(eqp_id, ""),
            "oper_id":       eqp.prev_oper or "",
            "PLAN_PROD_ATTR_VAL": eqp.prev_prod or "",
            "from_lot_cd":   from_lot_cd,
            "from_temp":     from_temp,
            "to_lot_cd":     row["to_lot_cd"],
            "to_temp":       row["to_temp"],
            "conv_start_min": conv_start,
            "conv_end_min":  conv_end,
            "conv_time":     self._conversion_minutes,
            "source":        "SCHEDULED",
        })
        self._emit_event(
            EVENT_CONV_ASSIGNED, eqp_id,
            from_lot_cd=from_lot_cd, from_temp=from_temp,
            to_lot_cd=row["to_lot_cd"], to_temp=row["to_temp"],
            conv_duration_min=self._conversion_minutes, conv_end_tm=conv_end,
            source="SCHEDULED",
        )
        self._push_event(conv_end, EVENT_FORCED_CONV_END, eqp_id)
        self._invalidate_caches()
        return True

    def _on_down_end(self, eqp_id: str) -> None:
        """다운타임 종료 → idle 복귀."""
        eqp = self.eqps.get(eqp_id)
        if not eqp or eqp.status != "down":
            return
        eqp.status = "idle"
        eqp.free_at = self.current_time
        self._invalidate_caches()

    def _on_forced_conv_end(self, eqp_id: str) -> None:
        """외부 확정 전환 완료 → idle 복귀(배정할 LOT 없이 셋업만 변경)."""
        eqp = self.eqps.get(eqp_id)
        if not eqp or eqp.status != "converting":
            return
        eqp.status = "idle"
        eqp.free_at = self.current_time
        self._invalidate_caches()

    def _on_conv_end(self, eqp_id: str) -> None:
        """Conversion 완료 후 가공 시작 (conv_end 기반)."""
        pending = self._eqp_pending_assign.pop(eqp_id, None)
        if not pending:
            return
        eqp = self.eqps.get(eqp_id)
        if not eqp or eqp.status != "converting":
            return
        eqp.status = "idle"
        self._finalize_assignment(
            eqp_id,
            pending,
            start_time=self.current_time,
            proc_reward=pending.get("proc_reward", 0.0),
        )

    def _advance_to_next_decision(self) -> None:
        """
        DES 시간 전진:
          1) process_end → move_out/tool/wip
          2) 동일 시각 idle (idle gap 없음)
          3) 결정점 반환 (_current_eqp)
        """
        self._current_eqp = None
        while self._event_q:
            batch = self._pop_event_batch()
            t = batch[0].time
            self.current_time = t

            deferred_idle_decisions: List[str] = []
            for ev in batch:
                if ev.kind == EVENT_PROCESS_END:
                    self._on_process_end(ev.eqp_id)
                    deferred_idle_decisions.append(ev.eqp_id)
                elif ev.kind == EVENT_IDLE_DECISION:
                    self._on_idle_decision(ev.eqp_id)
                elif ev.kind == EVENT_CONV_END:
                    self._on_conv_end(ev.eqp_id)
                elif ev.kind == EVENT_FORCED_CONV_END:
                    self._on_forced_conv_end(ev.eqp_id)
                    deferred_idle_decisions.append(ev.eqp_id)
                elif ev.kind == EVENT_DOWN_END:
                    self._on_down_end(ev.eqp_id)
                    deferred_idle_decisions.append(ev.eqp_id)

            for eqp_id in deferred_idle_decisions:
                self._on_idle_decision(eqp_id)

            for eqp_id in deferred_idle_decisions:
                self._schedule_wait_event(eqp_id, t)

            if self._current_eqp is None:
                self._current_eqp = self._pick_next_idle_eqp()

            if self._current_eqp:
                return

    # --- 결정 시점 / 내부 헬퍼 ---

    def _apply_eqp_initial_state(self, rows: List[dict]) -> None:
        """JSON eqp_initial_state → Equipment prev_lot_cd/temp/prod.

        tool_capacity(MAX_TOOL)는 '현재 장착된 것을 제외한 추가 가용 수(net)'로
        해석한다. 따라서 초기 셋업으로 이미 장착된 장비는 occupy하지 않는다
        (이미 MAX_TOOL 밖에 있으므로 차감하면 이중 차감이 된다). prev_lot_cd 등은
        전환(conversion) 판단을 위해 설정한다.
        """
        for row in rows:
            eid = row.get("eqp_id")
            if not eid or eid not in self.eqps:
                continue
            eqp = self.eqps[eid]
            lot_cd = row.get("lot_cd") or None
            temp = row.get("temp") or None
            if lot_cd:
                eqp.prev_lot_cd = lot_cd
            if temp:
                eqp.prev_temp = temp
            if row.get("PLAN_PROD_ATTR_VAL"):
                eqp.prev_prod = row["PLAN_PROD_ATTR_VAL"]
            if row.get("oper_id"):
                eqp.prev_oper = row["oper_id"]

    def _forced_lot_stat_cd(self, lot_id: str) -> str:
        lot = self.lot_pool.get(lot_id)
        meta = self._wip_lot_meta.get(lot_id, {})
        if lot is not None:
            return lot.lot_stat_cd
        return meta.get("lot_stat_cd", "WAIT")

    def _try_assign_staged_forced(self, eqp_id: str) -> bool:
        """선부착(staged) 강제 carrier를 가공 슬롯이 비면 즉시 배정."""
        staged = self._eqp_staged_forced.get(eqp_id)
        if not staged:
            return False
        eqp = self.eqps.get(eqp_id)
        if eqp is None or eqp.status != "idle":
            return False
        lot_id = staged[0]
        saved_eqp = self._current_eqp
        self._current_eqp = None
        ok = self.assign_lot(eqp_id, lot_id) != -1.0
        self._current_eqp = saved_eqp
        return ok

    def _consume_forced_placement(self, eqp_id: str, lot_id: str) -> None:
        """강제 carrier 배정 완료 시 queue/staged에서 제거."""
        staged = self._eqp_staged_forced.get(eqp_id)
        if staged and staged[0] == lot_id:
            staged.pop(0)
            if not staged:
                self._eqp_staged_forced.pop(eqp_id, None)
            return
        forced_queue = self._eqp_forced_queue.get(eqp_id)
        if forced_queue and forced_queue[0] == lot_id:
            forced_queue.pop(0)

    def _prebind_forced_carriers(self) -> None:
        """LOT_STAT_CD!=WAIT carrier 전건을 t=0에 장비에 선반영.

        - PROC: 가공 슬롯이 비어 있으면 즉시 가공 시작
        - LOAD/RESV/SELE(및 대기 중인 PROC): 장비에 선부착(staged) — 슬롯이 비면
          자동 배정(_try_assign_staged_forced), RL step 불필요
        """
        from utils.helpers import FORCED_LOT_STAT_CDS

        self._eqp_staged_forced = {}
        self._current_eqp = None
        for eqp_id in self._env_data.get("eqp_ids", []):
            queue = list(self._eqp_forced_queue.get(eqp_id, []))
            if not queue:
                continue
            self._eqp_forced_queue[eqp_id] = []
            staged: List[str] = []
            for lot_id in queue:
                lot_stat_cd = self._forced_lot_stat_cd(lot_id)
                if lot_stat_cd not in FORCED_LOT_STAT_CDS:
                    continue
                eqp = self.eqps.get(eqp_id)
                if (
                    lot_stat_cd == "PROC"
                    and eqp is not None
                    and eqp.status == "idle"
                ):
                    if self.assign_lot(eqp_id, lot_id) != -1.0:
                        continue
                staged.append(lot_id)
            if staged:
                self._eqp_staged_forced[eqp_id] = staged
            while self._try_assign_staged_forced(eqp_id):
                pass
        self._current_eqp = None

    def _bucket_lot_cd_temp(self, ppk: str, oper_id: str) -> Tuple[str, str]:
        """(PPK, OPER) WIP 풀의 LOT_CD/TEMP. batch_info 우선."""
        route = self._env_data.get("batch_info_map", {}).get((ppk, oper_id))
        if route:
            return route["lot_cd"], route["temp"]
        wip = self._wip_for(ppk, oper_id)
        if wip:
            for lid in wip.get("lot_ids", []):
                return self._lot_cd_temp(lid, ppk=ppk, oper_id=oper_id)
        for ld in self._env_data.get("lots", []):
            if ld.get("PLAN_PROD_ATTR_VAL") == ppk and ld.get("oper_id") == oper_id:
                return self._lot_cd_temp(
                    ld["lot_id"], ppk=ppk, oper_id=oper_id,
                )
        return "", ""

    def _would_need_conversion(self, eqp_id: str, lot_cd: str, temp: str) -> bool:
        eqp = self.eqps.get(eqp_id)
        if not eqp or eqp.prev_lot_cd is None or not lot_cd:
            return False
        return eqp.prev_lot_cd != lot_cd or (eqp.prev_temp or "") != (temp or "")

    def _needs_tool_swap(self, eqp_id: str, lot_cd: str, temp: str) -> bool:
        """EQP에 장착된 Pcode/Temp와 배정 대상이 다를 때만 tool 반환·장착."""
        eqp = self.eqps.get(eqp_id)
        if eqp is None or not lot_cd:
            return False
        if eqp.prev_lot_cd is None:
            return True
        return eqp.prev_lot_cd != lot_cd or (eqp.prev_temp or "") != (temp or "")

    def _tool_cap_blocks(self, eqp_id: str, lot_cd: str, temp: str) -> bool:
        if not self._needs_tool_swap(eqp_id, lot_cd, temp):
            return False
        return not self._tool_tracker.can_assign(lot_cd, eqp_id)

    def _conversion_group_blocks(self, eqp_id: str, lot_cd: str, temp: str) -> bool:
        """전환 그룹 제약: 현재 셋업과 '다른 그룹'으로의 전환이면 배정 차단.

        규칙(conversion_group.json이 있을 때만 활성):
          - 전환이 필요 없으면(동일 셋업) 절대 차단하지 않음
          - 첫 배정(prev_lot_cd=None)은 _would_need_conversion=False → 차단 안 함
          - from·to 둘 다 그룹에 속할 때만, 두 그룹이 다르면 차단
          - 어느 한쪽이라도 미지정(ungrouped)이면 제약 없음(차단 안 함)
        """
        if not self._conv_groups_active:
            return False
        if not self._would_need_conversion(eqp_id, lot_cd, temp):
            return False
        eqp = self.eqps.get(eqp_id)
        if eqp is None:
            return False
        from_g = self._conv_group_map.get((eqp.prev_lot_cd, eqp.prev_temp or ""))
        to_g = self._conv_group_map.get((lot_cd, temp or ""))
        if from_g is None or to_g is None:
            return False
        return from_g != to_g

    def _conversion_limit_blocks(self, eqp_id: str, lot_cd: str, temp: str) -> bool:
        """전환 횟수 상한: 한도 초과 시 전환이 필요한 배정을 차단."""
        if not self._would_need_conversion(eqp_id, lot_cd, temp):
            return False
        if self._max_conversions is not None and self.stats["conversions"] >= self._max_conversions:
            return True
        if self._max_conversions_per_eqp is not None:
            eqp = self.eqps.get(eqp_id)
            if eqp is not None and eqp.conversion_count >= self._max_conversions_per_eqp:
                return True
        return False

    def _assign_blocked(self, eqp_id: str, lot_cd: str, temp: str) -> bool:
        """배정 불가(feasibility) 통합 판단: tool 동시성 + 전환 그룹 + 전환 횟수 제약."""
        return (
            self._tool_cap_blocks(eqp_id, lot_cd, temp)
            or self._conversion_group_blocks(eqp_id, lot_cd, temp)
            or self._conversion_limit_blocks(eqp_id, lot_cd, temp)
        )

    def estimate_lot_end_time(self, eqp_id: str, lot: dict) -> int:
        """예상 종료 시각 = 현재 시각 + conversion(필요 시) + 장수×ST."""
        t = self.current_time
        lot_cd = lot.get("lot_cd", "")
        temp = lot.get("temp", "")
        if self._would_need_conversion(eqp_id, lot_cd, temp):
            t += self._conversion_minutes
        return t + int(lot.get("processing_time", 0))

    def earliest_st_combo_score(self, eqp_id: str, lot: dict) -> Tuple[int, int, str, str]:
        """
        EQP×carrier(LOT) 조합 점수: 예상 종료 시각(현재+conversion+장수×ST) 최소 →
        동률이면 전환(conversion) 불필요한(=동일 setup) 조합 우선.
        lot['processing_time']은 split 이후 wf_qty×ST.
        """
        end = self.estimate_lot_end_time(eqp_id, lot)
        lot_cd = lot.get("lot_cd", "")
        temp = lot.get("temp", "")
        needs_conv = self._would_need_conversion(eqp_id, lot_cd, temp)
        carrier = str(lot.get("carrier_id") or lot.get("lot_id", ""))
        return (end, int(needs_conv), carrier, str(lot.get("lot_id", "")))

    def pick_earliest_st_assignment(self) -> Optional[Tuple[str, str, dict]]:
        """
        idle EQP × feasible carrier(LOT) 중 예상 종료 시각 최소 조합 1건.
        PPK/OPER 버킷 없이 실제 재공 단위로 선택.
        """
        best: Optional[Tuple[str, str, dict]] = None
        best_score = (10**9, 1, "", "")

        for eqp_id in self.get_idle_eqps():
            for lot in self.available_lots(eqp_id):
                if not self._lot_conv_discrete_eligible(eqp_id, lot):
                    continue
                lot_cd = lot.get("lot_cd", "")
                temp = lot.get("temp", "")
                if self._assign_blocked(eqp_id, lot_cd, temp):
                    continue
                score = self.earliest_st_combo_score(eqp_id, lot)
                if score < best_score:
                    best_score = score
                    best = (eqp_id, str(lot["lot_id"]), lot)

        return best

    def assign_earliest_st_pending(self, eqp_id: str) -> float:
        """pick_earliest_st_assignment으로 정한 EQP×LOT을 배정."""
        if not self._earliest_st_pick:
            return -1.0
        pick_eqp, lot_id = self._earliest_st_pick
        self._earliest_st_pick = None
        if pick_eqp != eqp_id:
            return -1.0
        return self.assign_lot(eqp_id, lot_id)

    def _idle_eqps_with_work(self) -> List[str]:
        return [
            eqp_id for eqp_id in self._env_data["eqp_ids"]
            if self.eqps[eqp_id].status == "idle"
            and self.get_feasible_ppk_oper(eqp_id)
        ]

    def _pick_next_idle_eqp(self) -> Optional[str]:
        """다음 결정 EQP. min_st: EQP×carrier 조합 점수 최소 설비."""
        candidates = self._idle_eqps_with_work()
        if not candidates:
            return None
        if self._eqp_selection == "min_st":
            pick = self.pick_earliest_st_assignment()
            if pick:
                eqp_id, lot_id, _ = pick
                self._earliest_st_pick = (eqp_id, lot_id)
                return eqp_id
            return candidates[0]
        return candidates[0]

    def _next_wip_ready_time(self, after: int) -> Optional[int]:
        """WIP 풀에서 after 이후 투입 가능해지는 가장 이른 oper_in_time."""
        candidates: List[int] = []
        for wip in self._wip_pool.values():
            if wip["wip_qty"] <= 0:
                continue
            for lid in wip["lot_ids"]:
                t = self._wip_lot_meta.get(lid, {}).get("oper_in_time", 0)
                meta = self._wip_lot_meta.get(lid, {})
                if not self._is_current_wip_lot(
                    lid,
                    meta.get("PLAN_PROD_ATTR_VAL", ""),
                    meta.get("oper_id", ""),
                ):
                    continue
                if t > after:
                    candidates.append(t)
        return min(candidates) if candidates else None

    def _schedule_wait_event(self, eqp_id: str, time: int) -> None:
        if self.get_feasible_ppk_oper(eqp_id):
            return
        next_ready = self._next_wip_ready_time(time)
        if next_ready is not None and next_ready <= self.sim_end:
            self._push_event(next_ready, EVENT_IDLE_DECISION, eqp_id)
        elif time < self.sim_end:
            self._push_event(self.sim_end, EVENT_IDLE_DECISION, eqp_id)

    def _select_same_time_next_eqp(self):
        """배정 직후 동일 시각에 다른 idle EQP 결정 시점 선점 (시간 전진 없음)."""
        self._current_eqp = self._pick_next_idle_eqp()

    def _wip_key(self, ppk: str, oper_id: str) -> Tuple[str, str]:
        return (ppk, oper_id)

    def _wip_for(self, ppk: str, oper_id: str) -> Optional[dict]:
        return self._wip_pool.get(self._wip_key(ppk, oper_id))

    def _eqp_can_process(self, eqp_id: str, ppk: str, oper_id: str) -> bool:
        """abstract arrange(MODEL) 또는 discrete eqp_oper_cap 기준 처리 가능."""
        model = self._eqp_model_map[eqp_id]
        arrange_map = self._env_data.get("abstract_arrange_map", {})
        if (ppk, oper_id, model) in arrange_map:
            return True
        return oper_id in self._env_data.get("eqp_oper_cap", {}).get(eqp_id, [])

    def _abstract_row_for(self, eqp_id: str, ppk: str, oper_id: str) -> Optional[dict]:
        model = self._eqp_model_map[eqp_id]
        for row in self._abstract_template:
            if (
                row["PLAN_PROD_ATTR_VAL"] == ppk
                and row["oper_id"] == oper_id
                and row["eqp_model"] == model
            ):
                return row
        return None

    def _materialize_abstract_rows(self) -> List[dict]:
        """템플릿 + WIP 풀 → UI/히스토리용 abstract arrange."""
        rows = []
        for tmpl in self._abstract_template:
            wip = self._wip_for(tmpl["PLAN_PROD_ATTR_VAL"], tmpl["oper_id"])
            item = dict(tmpl)
            if wip:
                item["wip_qty"] = wip["wip_qty"]
                item["wip_qty_init"] = wip.get("wip_qty_init", wip["wip_qty"])
                item["oper_in_time"] = wip.get("oper_in_time", 0)
                item["min_inject_time"] = wip.get("min_inject_time", 0)
            else:
                item["wip_qty"] = 0
                item["wip_qty_init"] = 0
                item["oper_in_time"] = 0
                item["min_inject_time"] = 0
            rows.append(item)
        return rows

    def _inject_wip(
        self, ppk: str, oper_id: str, lot_id: str, oper_in_time: int, meta: dict,
    ) -> None:
        """후속 재공 유입: (PPK,OPER) WIP +1, oper_in_time 갱신."""
        key = self._wip_key(ppk, oper_id)
        if key not in self._wip_pool:
            self._wip_pool[key] = {
                "wip_qty":         0,
                "wip_qty_init":    0,
                "oper_in_time":    0,
                "min_inject_time": oper_in_time,
                "lot_ids":         [],
            }
        wip = self._wip_pool[key]
        wip["wip_qty"] += 1
        wip["lot_ids"].append(lot_id)
        wip["oper_in_time"] = max(wip.get("oper_in_time", 0), oper_in_time)
        wip["min_inject_time"] = min(
            wip.get("min_inject_time", oper_in_time), oper_in_time,
        )
        self._wip_lot_meta[lot_id] = {
            "PLAN_PROD_ATTR_VAL": ppk,
            "oper_id":       oper_id,
            "seq":           meta.get("seq", 1),
            "wf_qty":        meta.get("wf_qty", 25),
            "carrier_id":    meta.get("carrier_id", ""),
            "logical_lot_id": meta.get("logical_lot_id", lot_id),
            "oper_in_time":  oper_in_time,
        }
        self._invalidate_caches()

    def _consume_wip(self, ppk: str, oper_id: str, lot_id: str) -> None:
        key = self._wip_key(ppk, oper_id)
        wip = self._wip_pool.get(key)
        if not wip:
            return
        wip["wip_qty"] = max(wip["wip_qty"] - 1, 0)
        if lot_id in wip["lot_ids"]:
            wip["lot_ids"].remove(lot_id)
        self._invalidate_caches()

    def _inject_wip_after_complete(self, lot_id: str, complete_time: int) -> Optional[dict]:
        """한 공정 완료 후 다음 공정 WIP +1. move_out 시점에 호출."""
        meta = self._in_flight.pop(lot_id, None)
        if not meta:
            return None
        if not self._enable_wip_inflow:
            return None

        ppk = meta["PLAN_PROD_ATTR_VAL"]
        seq = meta["seq"]
        next_info = self._env_data.get("flow_next", {}).get(ppk, {}).get(seq)
        if not next_info:
            return None

        next_oper = next_info["next_oper"]
        self._inject_wip(
            ppk, next_oper, lot_id, complete_time,
            {
                "seq":        next_info["next_seq"],
                "wf_qty":     meta.get("wf_qty", 25),
                "carrier_id": meta.get("carrier_id", ""),
                "logical_lot_id": meta.get("logical_lot_id", lot_id),
            },
        )
        return {
            "lot_id": lot_id,
            "PLAN_PROD_ATTR_VAL": ppk,
            "oper_id": next_oper,
            "oper_in_time": complete_time,
        }

    def _lot_ready(self, lot_id: str, oper_in_time: int) -> bool:
        return self.current_time >= oper_in_time

    def _current_wip_only(self) -> bool:
        return self._termination_mode == "current_wip_assigned"

    def _is_current_wip_lot(self, lot_id: str, ppk: str, oper_id: str) -> bool:
        if not self._current_wip_only():
            return True
        return (lot_id, ppk, oper_id) in self._initial_wip_lot_keys

    def _has_unassigned_current_wip(self) -> bool:
        for lid, ppk, oper_id in self._initial_wip_lot_keys:
            wip = self._wip_pool.get((ppk, oper_id))
            if wip and lid in wip.get("lot_ids", []):
                return True
        return False

    def get_remaining_current_wip(self) -> Dict[str, int]:
        """시뮬 시작 시점 현재 재공 중 아직 배정되지 않은 수량."""
        remaining: Dict[str, int] = {}
        for lid, ppk, oper_id in self._initial_wip_lot_keys:
            wip = self._wip_pool.get((ppk, oper_id))
            if not wip or lid not in wip.get("lot_ids", []):
                continue
            meta = self._wip_lot_meta.get(lid, {})
            wf = int(meta.get("wf_qty", 25))
            key = f"{ppk}|{oper_id}"
            remaining[key] = remaining.get(key, 0) + wf
        return remaining

    def get_in_flight_qty(self, ppk: str, oper_id: str) -> int:
        """(PPK, OPER) 기준 현재 장비에서 처리 중(배정됐지만 미완료)인 lot 수."""
        return sum(
            1
            for meta in self._in_flight.values()
            if meta.get("PLAN_PROD_ATTR_VAL") == ppk and meta.get("oper_id") == oper_id
        )

    def _abstract_assignable_on_eqp(self, eqp_id: str) -> List[dict]:
        model = self._eqp_model_map.get(eqp_id)
        if not model:
            return []
        rows = []
        for tmpl in self._abstract_template:
            if tmpl["eqp_model"] != model:
                continue
            ppk, oper_id = tmpl["PLAN_PROD_ATTR_VAL"], tmpl["oper_id"]
            wip = self._wip_for(ppk, oper_id)
            if not wip or wip["wip_qty"] <= 0:
                continue
            if not self._eqp_can_process(eqp_id, ppk, oper_id):
                continue
            ready = False
            for lid in wip["lot_ids"]:
                if not self._is_current_wip_lot(lid, ppk, oper_id):
                    continue
                meta = self._wip_lot_meta.get(lid, {})
                oper_in_time = meta.get("oper_in_time", 0)
                if self._lot_ready(lid, oper_in_time):
                    ready = True
                    break
            if not ready:
                continue
            row = dict(tmpl)
            row["wip_qty"] = wip["wip_qty"]
            row["oper_in_time"] = wip.get("oper_in_time", 0)
            row["min_inject_time"] = wip.get("min_inject_time", 0)
            rows.append(row)
        return rows

    def _lot_injectable(self, lot_id: str) -> bool:
        """LOT이 현재 시각에 투입 가능한지 (시뮬 종료 전 + 미배정)."""
        if lot_id not in self.lot_pool:
            return False
        return self.current_time <= self.sim_end

    def _has_discrete_combo(self, eqp_id: str, lot_id: str, oper_id: str) -> bool:
        return (lot_id, eqp_id, oper_id) in self._env_data.get("proc_time_matrix", {})

    def _lot_conv_discrete_eligible(self, eqp_id: str, lot: dict) -> bool:
        """전환이 필요 없는 WAIT LOT은 discrete(EQP×carrier 정밀 조합)이 있어야만
        배정 가능하다 — 같은 (ppk,oper)라도 carrier별 실제 가능 여부가 다를 수
        있어 abstract(모델 매칭) 하나만으로는 불충분하다. 전환이 필요한 경우엔
        반대로 discrete를 참조하지 않고 abstract 기준으로만 판단한다(전환 시
        기존 셋업 기준 discrete 조합이 더는 유효하지 않으므로).

        강제 배정 LOT(PROC/LOAD/RESV/SELE)과 discrete_wait_enabled=False인
        경우는 이 제약과 무관하게 항상 통과한다(기존 abstract 폴백 유지)."""
        if lot.get("lot_stat_cd", "WAIT") != "WAIT":
            return True
        if not CONFIG.env.discrete_wait_enabled:
            return True
        eqp = self.eqps.get(eqp_id)
        if eqp is None or eqp.prev_lot_cd is None:
            return True  # 첫 배정(기존 셋업 없음) — 전환 비교 대상이 없어 abstract 허용
        lot_cd = lot.get("lot_cd", "")
        temp = lot.get("temp", "")
        if self._would_need_conversion(eqp_id, lot_cd, temp):
            return True
        return not lot.get("is_abstract", True)

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
        """PPK×OPER×MODEL 템플릿 + (PPK,OPER) WIP 카운트."""
        return self._materialize_abstract_rows()

    def get_wip_waiting(self) -> Dict[str, int]:
        """(PPK|OPER)별 대기 WIP 웨이퍼 수. 결과를 버전 기반으로 캐싱."""
        if (self._wip_waiting_cache is not None
                and self._wip_waiting_version == self._state_version):
            return self._wip_waiting_cache
        result = self._build_wip_waiting()
        self._wip_waiting_cache = result
        self._wip_waiting_version = self._state_version
        return result

    def _build_wip_waiting(self) -> Dict[str, int]:
        """get_wip_waiting() 실계산 본체."""
        wip: Dict[str, int] = {}
        wf_defaults: Dict[str, int] = {}
        for tmpl in self._abstract_template:
            wf_defaults[tmpl["PLAN_PROD_ATTR_VAL"]] = tmpl.get("wf_qty", 25)
        for (ppk, oper_id), pool in self._wip_pool.items():
            if pool["wip_qty"] <= 0:
                continue
            wf = wf_defaults.get(ppk, 25)
            key = f"{ppk}|{oper_id}"
            wip[key] = wip.get(key, 0) + pool["wip_qty"] * wf
        return wip

    def _serialize_wip_waiting(self) -> Dict[str, int]:
        return self.get_wip_waiting()

    def _lot_cd_temp(
        self,
        lot_id: str,
        lot: Optional[Lot] = None,
        ppk: Optional[str] = None,
        oper_id: Optional[str] = None,
    ) -> tuple:
        if ppk and oper_id:
            route = self._env_data.get("batch_info_map", {}).get((ppk, oper_id))
            if route:
                return route["lot_cd"], route["temp"]
        if lot is not None and lot.lot_cd and not (ppk and oper_id):
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
        self,
        eqp: Equipment,
        lot_cd: str,
        temp: str,
        reward: float,
        *,
        oper_id: str = "",
        ppk: str = "",
        eqp_model: str = "",
    ) -> tuple:
        """LOT_CD/TEMP 변경 시 conversion 필요 여부·시작 시각·보상."""
        self._cur_conv_terms = {}
        start_time = max(self.current_time, eqp.free_at)
        idle_before = reward
        reward = self._accumulate_idle(eqp, reward, start_time)
        if reward != idle_before:
            self._cur_conv_terms["idle"] = round(reward - idle_before, 4)
        # 변환 필요 판단은 _would_need_conversion 하나로 일원화 (중복 제거)
        needs_conv = self._would_need_conversion(eqp.eqp_id, lot_cd, temp)
        if needs_conv:
            # 전환 1회당 고정 패널티
            reward += self._reward_cfg.w_conversion
            self._cur_conv_terms["conversion"] = round(float(self._reward_cfg.w_conversion), 4)
            if self._reward_cfg.w_avoidable_conversion < 0:
                # 회피 가능했던 전환이면 추가 패널티 (식: w_avoidable_conversion * avoidable[0,1])
                # avoidable: 다른 무변환 장비가 커버 가능한 비율 또는 변환 후 가동시간이
                # 너무 짧아 비용을 못 메우는 비율 중 큰 값 (_conversion_avoidable_fraction)
                avoidable = self._conversion_avoidable_fraction(
                    eqp.eqp_id, ppk, oper_id, lot_cd, temp,
                )
                if avoidable > 0:
                    av_term = self._reward_cfg.w_avoidable_conversion * avoidable
                    reward += av_term
                    self._cur_conv_terms["avoidable_conversion"] = round(float(av_term), 4)
            eqp.conversion_count += 1
            self.stats["conversions"] += 1
            conv_end = start_time + self._conversion_minutes
            from_lot_cd = eqp.prev_lot_cd
            if from_lot_cd:
                self._tool_tracker.release(from_lot_cd, eqp.eqp_id)
            self._tool_tracker.occupy(lot_cd, eqp.eqp_id)
            self._emit_event(
                EVENT_CONV_ASSIGNED, eqp.eqp_id,
                event_time=start_time,
                from_lot_cd=from_lot_cd,
                from_temp=eqp.prev_temp or "",
                to_lot_cd=lot_cd,
                to_temp=temp,
                oper_id=oper_id,
                PLAN_PROD_ATTR_VAL=ppk,
                eqp_model=eqp_model,
                conv_duration_min=self._conversion_minutes,
                conv_end_tm=conv_end,
                tool_from_delta=1,
                tool_to_delta=-1,
            )
            self.conversion_plans.append({
                "eqp_id":        eqp.eqp_id,
                "eqp_model_cd":  eqp_model,
                "oper_id":       oper_id,
                "PLAN_PROD_ATTR_VAL": ppk,
                "from_lot_cd":   eqp.prev_lot_cd,
                "from_temp":     eqp.prev_temp,
                "to_lot_cd":     lot_cd,
                "to_temp":       temp,
                "conv_start_min": start_time,
                "conv_end_min":   conv_end,
                "conv_time":      self._conversion_minutes,
            })
            return start_time, conv_end, reward, True
        return start_time, start_time, reward, False

    def _has_plan(self, ppk: str, oper_id: str) -> bool:
        """(PPK, OPER)에 유효 계획량이 있으면 True."""
        pm = self._env_data.get("plan_meta", {}).get((ppk, oper_id))
        return bool(pm and pm.get("d0_plan_qty", 0) > 0)

    def _flow_post(self, ppk: str, oper_id: str) -> Optional[str]:
        """같은 PPK의 다음(후속) 공정 OPER (없으면 None)."""
        return self._env_data.get("flow_post", {}).get(ppk, {}).get(oper_id)

    def _ready_wip_qty(self, ppk: str, oper_id: str) -> int:
        """(PPK, OPER)에서 oper_in_time 도래한 ready 재공 웨이퍼 수."""
        wip = self._wip_for(ppk, oper_id)
        if not wip or wip.get("wip_qty", 0) <= 0:
            return 0
        total = 0
        for lid in wip.get("lot_ids", []):
            meta = self._wip_lot_meta.get(lid, {})
            lot = self.lot_pool.get(lid)
            oper_in_time = meta.get("oper_in_time", wip.get("oper_in_time", 0))
            if not self._lot_ready(lid, oper_in_time):
                continue
            wf = int(meta.get("wf_qty", lot.wf_qty if lot else 25))
            total += wf
        return total

    def _st_per_wafer_for_eqp(self, eqp_id: str, ppk: str, oper_id: str) -> Optional[float]:
        """EQP×(PPK, OPER) 장당 ST(분/매)."""
        model = self._eqp_model_map.get(eqp_id)
        if not model:
            return None
        arrange_map = self._env_data.get("abstract_arrange_map", {})
        st = arrange_map.get((ppk, oper_id, model))
        if st is not None:
            return float(st)
        row = self._abstract_row_for(eqp_id, ppk, oper_id)
        if row is not None:
            return float(row.get("proc_time", row.get("ST", 0)))
        lst = self._env_data.get("abstract_arranges_by_ppk_oper", {}).get((ppk, oper_id))
        if lst:
            return sum(s for _, s in lst) / len(lst)
        return None

    def _eqp_feasible_bucket_keys(self, eqp_id: str) -> set:
        """idle EQP가 지금 배정 가능한 (PPK, OPER) 버킷 집합."""
        if self.eqps[eqp_id].status != "idle":
            return set()
        lots = self.available_lots(eqp_id)
        if not lots:
            return set()
        feasible: set = set()
        buckets = {(l["PLAN_PROD_ATTR_VAL"], l["oper_id"]) for l in lots}
        for bucket_ppk, bucket_oper in buckets:
            lot_id = self._auto_select_lot(eqp_id, [
                l for l in lots
                if l["PLAN_PROD_ATTR_VAL"] == bucket_ppk and l["oper_id"] == bucket_oper
            ])
            if lot_id is None:
                continue
            lot_cd, temp = self._lot_cd_temp(
                lot_id, self.lot_pool.get(lot_id), ppk=bucket_ppk, oper_id=bucket_oper,
            )
            if self._assign_blocked(eqp_id, lot_cd, temp):
                continue
            feasible.add((bucket_ppk, bucket_oper))
        return feasible

    def _oper_committed_capacity_per_min(self, ppk: str, oper_id: str) -> float:
        """마지막 setup(prev_prod/prev_oper)이 일치하는 장비 합산 분당 처리량(매/분)."""
        total = 0.0
        for eqp_id in self._env_data.get("eqp_ids", []):
            if not self._eqp_can_process(eqp_id, ppk, oper_id):
                continue
            eqp = self.eqps.get(eqp_id)
            if eqp is None or eqp.prev_prod != ppk or eqp.prev_oper != oper_id:
                continue
            st = self._st_per_wafer_for_eqp(eqp_id, ppk, oper_id)
            if st is not None and st > 0:
                total += 1.0 / st
        return total

    def _oper_idle_split_capacity_per_min(self, ppk: str, oper_id: str) -> float:
        """커밋 장비가 없을 때 idle+capable 장비의 capa를 feasible 버킷 수(K)로 균등 분배."""
        bucket = (ppk, oper_id)
        total = 0.0
        for eqp_id in self._env_data.get("eqp_ids", []):
            if not self._eqp_can_process(eqp_id, ppk, oper_id):
                continue
            eqp = self.eqps.get(eqp_id)
            if eqp is None or eqp.status != "idle":
                continue
            feasible = self._eqp_feasible_bucket_keys(eqp_id)
            if bucket not in feasible:
                continue
            k = len(feasible)
            if k <= 0:
                continue
            st = self._st_per_wafer_for_eqp(eqp_id, ppk, oper_id)
            if st is not None and st > 0:
                total += (1.0 / st) / k
        return total

    def _oper_capacity_per_min(self, ppk: str, oper_id: str) -> float:
        """(PPK, OPER) 유효 분당 처리량(매/분).

        1) setup 커밋(prev_prod/prev_oper 일치) 장비: 전액 1/ST
        2) 커밋 0이면 idle+capable 장비를 feasible 버킷 수로 균등 분배(방법 B)
        """
        committed = self._oper_committed_capacity_per_min(ppk, oper_id)
        if committed > 0:
            return committed
        return self._oper_idle_split_capacity_per_min(ppk, oper_id)

    def _setup_demand_wafers(self, lot_cd: str, temp: str) -> int:
        """이 (LOT_CD, TEMP) 세팅으로 처리해야 할 ready 재공 매수 합.
        같은 세팅을 공유하는 모든 (PPK, OPER) 버킷을 합산한다(변환 1회로 함께 열림)."""
        total = 0
        for (ppk, oper_id), pool in self._wip_pool.items():
            if pool.get("wip_qty", 0) <= 0:
                continue
            lc, tp = self._bucket_lot_cd_temp(ppk, oper_id)
            if lc == lot_cd and (tp or "") == (temp or ""):
                total += self._ready_wip_qty(ppk, oper_id)
        return total

    def _matched_no_conv_cover_wafers(
        self, exclude_eqp: str, lot_cd: str, temp: str,
    ) -> float:
        """이미 (LOT_CD, TEMP)로 세팅돼 '변환 없이' 이 세팅 재공을 처리 가능한
        '다른' 장비들이 soft_cutoff까지 남은 시간 안에 처리 가능한 매수 합.
        장비가 바쁠수록(free_at 큼) 기여 capa가 줄어 부하가 반영된다."""
        horizon = max(self.soft_cutoff - self.current_time, 0)
        if horizon <= 0:
            return 0.0
        total = 0.0
        for other in self._env_data.get("eqp_ids", []):
            if other == exclude_eqp:
                continue
            oe = self.eqps.get(other)
            if oe is None or oe.prev_lot_cd is None:
                continue
            if oe.prev_lot_cd != lot_cd or (oe.prev_temp or "") != (temp or ""):
                continue  # 변환 필요 → 무변환 대안 아님
            if self._tool_cap_blocks(other, lot_cd, temp):
                continue
            # 이 장비가 처리 가능한 (이 세팅의) 공정 중 가장 빠른 장당 ST
            best_st: Optional[float] = None
            for (ppk, oper_id), pool in self._wip_pool.items():
                if pool.get("wip_qty", 0) <= 0:
                    continue
                lc, tp = self._bucket_lot_cd_temp(ppk, oper_id)
                if lc != lot_cd or (tp or "") != (temp or ""):
                    continue
                if not self._eqp_can_process(other, ppk, oper_id):
                    continue
                st = self._st_per_wafer_for_eqp(other, ppk, oper_id)
                if st and st > 0 and (best_st is None or st < best_st):
                    best_st = st
            if best_st is None:
                continue
            avail = max(horizon - max(oe.free_at - self.current_time, 0), 0)
            total += avail / best_st
        return total

    def _conversion_avoidable_fraction(
        self, eqp_id: str, ppk: str, oper_id: str, lot_cd: str, temp: str,
    ) -> float:
        """이 장비가 (lot_cd, temp)로 변환하는 것이 '회피 가능한' 정도 [0,1].

        1에 가까울수록: 다른 장비가 시간 내 재공을 커버할 수 있거나, 변환해도
        내가 무변환으로 돌 가동시간이 너무 짧아 변환 비용을 amortize 못함
        → 다른 장비에 맡기는 게 낫다.
        0: 변환이 정당(대안 부하 심함 + 내가 충분히 길게 가동).
        """
        cfg = self._reward_cfg
        demand = self._setup_demand_wafers(lot_cd, temp)
        if demand <= 0:
            return 0.0
        alt_cap = self._matched_no_conv_cover_wafers(eqp_id, lot_cd, temp)
        coverage_frac = min(alt_cap / demand, 1.0)  # 대안이 커버하는 비율

        # 내가 변환 후 무변환으로 처리할 잔여 재공/가동시간
        residual = max(demand - alt_cap, 0.0)
        horizon = max(self.soft_cutoff - self.current_time, 0)
        my_st = self._st_per_wafer_for_eqp(eqp_id, ppk, oper_id) or 0.0
        my_cap = (horizon / my_st) if my_st > 0 else 0.0
        my_run_min = min(my_cap, residual) * my_st
        need = cfg.conversion_amortize_factor * self._conversion_minutes
        short_run_frac = (
            max(0.0, 1.0 - my_run_min / need) if need > 0 else 0.0
        )
        return max(coverage_frac, short_run_frac)

    def _oper_supply_rate(self, ppk: str, oper_id: str) -> float:
        """(PPK, OPER)에 재공을 공급하는 선행 공정의 유효 처리율(매/분).

        선행 공정이 흘려보내는 율 ≈ 선행 capa, 단 선행에 ready 재공이 있을 때만.
        선행이 없으면(첫 공정·외부 유입) 0 (소진 관점). 선행도 굶으면 0."""
        prev = self._flow_prev(ppk, oper_id)
        if prev is None:
            return 0.0
        if self._ready_wip_qty(ppk, prev) <= 0:
            return 0.0
        return self._oper_capacity_per_min(ppk, prev)

    def _downstream_wip_cover_minutes(self, ppk: str, oper_id: str) -> Optional[float]:
        """
        후속 공정 ready WIP ÷ '순감소율'(소비 − 공급) = 굶을 때까지 커버 시간(분).

        소비 = 후속 장비 합산 처리율, 공급 = 현재 공정(=후속의 선행)이 후속으로
        흘려보내는 율. 공급 ≥ 소비면 후속이 적체되므로 안 굶음(None).
        후속 공정·계획 없거나 처리 capa 없으면 None.
        """
        nxt = self._flow_post(ppk, oper_id)
        if nxt is None or not self._has_plan(ppk, nxt):
            return None
        consume = self._oper_capacity_per_min(ppk, nxt)
        if consume <= 0:
            return None
        # 후속(nxt)의 선행은 곧 현재 공정(oper_id) → 그 공급율을 차감
        supply = self._oper_supply_rate(ppk, nxt)
        net = consume - supply
        if net <= 1e-9:
            return None  # 공급 ≥ 소비 → 후속 적체, 안 굶음
        return self._ready_wip_qty(ppk, nxt) / net

    def _downstream_is_starving(self, ppk: str, oper_id: str) -> bool:
        """후속 net-rate 커버가 flow_balance_starving_cover_min 이하일 때만 starving.

        커버 시간은 소비−공급 순감소율 기준이므로, 현재 공정이 후속을 충분히
        먹이고 있으면(공급 ≥ 소비) starving이 아니다(과잉 feeding 억제)."""
        cover = self._downstream_wip_cover_minutes(ppk, oper_id)
        if cover is None:
            return False
        threshold = self._reward_cfg.flow_balance_starving_cover_min
        return cover <= threshold

    def _flow_prev(self, ppk: str, oper_id: str) -> Optional[str]:
        """같은 PPK의 이전(선행) 공정 OPER (없으면 None)."""
        return self._env_data.get("flow_prev", {}).get(ppk, {}).get(oper_id)

    def _wip_wafers(self, ppk: str, oper_id: str) -> int:
        """(PPK, OPER) 대기 WIP 웨이퍼 수 (없으면 0)."""
        return int(self.get_wip_waiting().get(f"{ppk}|{oper_id}", 0))

    def _achievable_qty(self, ppk: str, oper_id: str) -> int:
        """
        Step C: 오늘 실제 달성 가능한 상한 추정.
        = min(계획량, 이미생산 + 현 공정 WIP + 선행 공정들에서 흘러올 수 있는 WIP)
        재공이 부족하면 계획을 그 수준으로 낮춰 무리한 전환을 막는다.
        """
        pm = self._env_data.get("plan_meta", {}).get((ppk, oper_id))
        plan_qty = max(pm["d0_plan_qty"], 1) if pm else 1
        if not self._reward_cfg.use_achievable_target:
            return plan_qty
        done = self.stats["completed_qty"].get((ppk, oper_id), 0)
        reachable = self._wip_wafers(ppk, oper_id)
        # 선행 공정 체인의 WIP은 처리되면 이 공정으로 흘러올 수 있다.
        seen = {oper_id}
        prev = self._flow_prev(ppk, oper_id)
        while prev and prev not in seen:
            seen.add(prev)
            reachable += self._wip_wafers(ppk, prev)
            prev = self._flow_prev(ppk, prev)
        return min(plan_qty, done + reachable)

    def _plan_hit_reward(self, ppk: str, oper_id: str, wf_qty: int, end_time: int) -> float:
        """계획 달성 진척 보너스.

        식: gap = max(target - done, 0)   # achievable_qty 기준 잔여 목표
            reward = w_plan_hit * (gap_before - gap_after) / target
            → gap이 줄수록(목표에 가까워질수록) +. 이미 달성(gap=0)이면 추가 보상 없음.
        """
        reward = 0.0
        if end_time > self.soft_cutoff or not self._has_plan(ppk, oper_id):
            return reward
        cfg = self._reward_cfg
        if cfg.w_plan_hit <= 0:
            return reward
        key = (ppk, oper_id)
        done_before = self.stats["completed_qty"].get(key, 0)
        # Step C: 고정 계획 대신 달성가능 상한 기준으로 진척 보상
        target = max(self._achievable_qty(ppk, oper_id), 1)
        gap_before = max(target - done_before, 0)
        done_after = done_before + wf_qty
        gap_after = max(target - done_after, 0)
        reward += cfg.w_plan_hit * (gap_before - gap_after) / target
        return reward

    def _bucket_projected_cover(
        self, ppk: str, oper_id: str, exclude_eqp: Optional[str],
    ) -> float:
        """(PPK, OPER)에 커밋된 '다른' 장비가 하루 끝까지 쭉 돌 때의 예상 생산(매).

        - 본인(exclude_eqp)은 제외 → 결정 장비가 자기 제품을 회피하는 버그 방지.
        - in-flight가 아니라 setup(prev_prod/prev_oper) 기준 → idle 핸드오프 구멍 방지
          (방금 끝내고 idle인 전담 장비도 커버로 계속 카운트).
        """
        remaining = max(self.soft_cutoff - self.current_time, 0)
        if remaining <= 0:
            return 0.0
        total = 0.0
        for eid in self._env_data.get("eqp_ids", []):
            if eid == exclude_eqp:
                continue
            e = self.eqps.get(eid)
            if e is None or e.prev_prod != ppk or e.prev_oper != oper_id:
                continue
            st = self._st_per_wafer_for_eqp(eid, ppk, oper_id)
            if st and st > 0:
                total += remaining / st
        return total

    def _pacing_shaping_reward(
        self, ppk: str, oper_id: str, wf_qty: int,
        at_time: Optional[int] = None, eqp_id: Optional[str] = None,
    ) -> float:
        """계획이 있는 (PPK, OPER)만 페이싱 shaping.

        진척 기준을 'done'이 아니라 'done + 다른 장비의 투영 커버(pacing_coverage_scale)'
        로 봐서, 이미 다른 장비가 충분히 덮는 제품은 pace 충족으로 간주(lockstep 억제).
        식: ideal = target * (t/horizon)   # 선형 takt 목표선
            eff   = done + coverage_scale * 다른장비_투영커버
            reward = w_pacing * (|ideal-eff_before| - |ideal-eff_after|) / target
            → 오차가 줄면 +, 늘면 -(이미 앞선 제품을 더 만들면 감점)
        """
        cfg = self._reward_cfg
        if cfg.w_pacing <= 0 or not self._has_plan(ppk, oper_id):
            return 0.0
        # Step C: 선형 takt ideal을 달성가능 상한 기준으로 (재공 한계까지만 추종)
        target = max(self._achievable_qty(ppk, oper_id), 1)
        horizon = max(self.soft_cutoff, 1)
        t = self.current_time if at_time is None else at_time
        ideal = target * min(max(t, 0), horizon) / horizon
        key = (ppk, oper_id)
        done_before = self.stats["completed_qty"].get(key, 0)
        # 다른 장비(본인 제외)의 투영 커버를 진척에 가산 → coverage-aware pacing
        cover = 0.0
        if cfg.pacing_coverage_scale > 0:
            cover = cfg.pacing_coverage_scale * self._bucket_projected_cover(
                ppk, oper_id, exclude_eqp=eqp_id,
            )
        eff_before = done_before + cover
        eff_after = eff_before + wf_qty
        err_before = abs(ideal - eff_before)
        err_after = abs(ideal - eff_after)
        return cfg.w_pacing * (err_before - err_after) / target

    def _ppk_has_feasible_assignment(self, ppk: str) -> bool:
        """현재 시점에 해당 PPK로 투입 가능한 (OPER, EQP) 조합이 있는지."""
        for flat, _ei in self.get_feasible_assignments():
            candidate_ppk, _oper = self.ppk_oper_from_flat(flat)
            if candidate_ppk == ppk:
                return True
        return False

    def _same_setup_reward(self, eqp: Equipment, ppk: str, oper_id: str, wf_qty: int) -> float:
        """제품·공정이 '모두' 직전과 동일할 때만 연속 보너스.

        공정 전환·제품 전환을 따로 보상하지 않고, 둘 다 같은 경우(=전환 없음,
        동일 라우트 단계 유지)에만 +를 준다. switch 통계는 그대로 집계.
        해당 PPK 재공 고갈(투입 불가) 시에는 보너스를 죽인다.
        식: same_oper AND same_prod AND ppk_has_feasible_assignment → +w_same_setup, 아니면 0
        """
        cfg = self._reward_cfg
        same_oper = (eqp.prev_oper == oper_id)
        same_prod = (eqp.prev_prod == ppk)
        if eqp.prev_oper is not None and not same_oper:
            eqp.oper_switches += 1
            self.stats["oper_switches"] += 1
        if eqp.prev_prod is not None and not same_prod:
            eqp.prod_switches += 1
            self.stats["prod_switches"] += 1
        if not (same_oper and same_prod):
            return 0.0
        if not self._ppk_has_feasible_assignment(ppk):
            return 0.0
        return cfg.w_same_setup

    def _flow_balance_reward(self, ppk: str, oper_id: str) -> float:
        """
        Step B: flow-balance shaping.
        기준을 균등(1/n)이 아닌 계획 비중(plan_qty 기준)으로 비교.
        - WIP 비중 > 계획 비중 → 이 공정에 적체, 더 돌려야 함 → +
        - WIP 비중 < 계획 비중 → 이 공정은 충분, 다른 공정 우선 → -
        - 후속 공정 starving 시 추가 +
        """
        cfg = self._reward_cfg
        if cfg.w_flow_balance <= 0:
            return 0.0
        wips = self.get_wip_waiting()
        total_wip = sum(wips.values())
        if total_wip <= 0:
            return 0.0
        here = wips.get(f"{ppk}|{oper_id}", 0)
        wip_share = here / total_wip

        # 계획 비중: 현재 WIP가 있는 (ppk, oper) 중 plan_qty 합 대비 이 공정의 비중
        plan_meta = self._env_data.get("plan_meta", {})
        total_plan = sum(
            plan_meta[(p, o)]["d0_plan_qty"]
            for key in wips
            for p, o in [key.split("|", 1)]
            if (p, o) in plan_meta and plan_meta[(p, o)].get("d0_plan_qty", 0) > 0
        )
        plan_here = plan_meta.get((ppk, oper_id), {}).get("d0_plan_qty", 0)
        plan_share = plan_here / max(total_plan, 1)

        score = wip_share - plan_share  # 적체(WIP>계획비중) → +, 여유 → -
        if self._downstream_is_starving(ppk, oper_id):
            score += 0.5
        return cfg.w_flow_balance * score

    def _logical_lot_id(
        self,
        lot: Optional[dict] = None,
        *,
        lot_id: str = "",
        meta: Optional[dict] = None,
    ) -> str:
        """carrier 단위 lot_id → 논리 LOT_ID (LOT_ID:CARRIER = 1:N)."""
        if lot is not None:
            return (
                lot.get("logical_lot_id")
                or lot.get("parent_lot_id")
                or lot["lot_id"]
            )
        if meta is not None:
            return meta.get("logical_lot_id") or meta.get("parent_lot_id") or lot_id
        pool_lot = self.lot_pool.get(lot_id)
        if pool_lot is not None:
            return pool_lot.logical_lot_id or pool_lot.parent_lot_id or lot_id
        wip_meta = self._wip_lot_meta.get(lot_id, {})
        return wip_meta.get("logical_lot_id") or wip_meta.get("parent_lot_id") or lot_id

    def _active_logical_lot_ids(self, ppk: str, oper_id: str) -> set:
        """동일 (PPK, OPER)에서 이미 장비 큐/가공 중인 carrier가 속한 논리 LOT 집합."""
        active: set = set()
        for lid, inflight in self._in_flight.items():
            if (
                inflight.get("PLAN_PROD_ATTR_VAL") == ppk
                and inflight.get("oper_id") == oper_id
            ):
                active.add(self._logical_lot_id(lot_id=lid, meta=inflight))
        for queue in self._eqp_forced_queue.values():
            for lid in queue:
                meta = self._wip_lot_meta.get(lid, {})
                lot = self.lot_pool.get(lid)
                m_ppk = meta.get("PLAN_PROD_ATTR_VAL") or (lot.PLAN_PROD_ATTR_VAL if lot else "")
                m_oper = meta.get("oper_id") or (lot.oper_id if lot else "")
                if m_ppk == ppk and m_oper == oper_id:
                    active.add(self._logical_lot_id(lot_id=lid, meta=meta))
        for staged in self._eqp_staged_forced.values():
            for lid in staged:
                meta = self._wip_lot_meta.get(lid, {})
                lot = self.lot_pool.get(lid)
                m_ppk = meta.get("PLAN_PROD_ATTR_VAL") or (lot.PLAN_PROD_ATTR_VAL if lot else "")
                m_oper = meta.get("oper_id") or (lot.oper_id if lot else "")
                if m_ppk == ppk and m_oper == oper_id:
                    active.add(self._logical_lot_id(lot_id=lid, meta=meta))
        for pending in self._eqp_pending_assign.values():
            if pending.get("ppk") == ppk and pending.get("oper_id") == oper_id:
                active.add(self._logical_lot_id(lot_id=pending["lot_id"], meta=pending))
        return active

    def _auto_select_lot(self, eqp_id: str, candidates: List[dict]) -> Optional[str]:
        """버킷 내 carrier 자동 선택.

        (PPK, OPER) 버킷은 이미 정해진 상태에서 아래 순서로 1건 선택한다.
        priority/seq는 버킷 단위 속성이라 여기서는 사용하지 않는다.

        1. 초기 재공 (is_initial_wip)
        2. discrete carrier (is_abstract=False)
        3. 진행 중 논리 LOT 소속 carrier — 같은 LOT의 다른 carrier가
           장비 큐/가공 중이면 나머지 carrier를 우선 (LOT 완결 지향)
        4. earliest ST (예상 종료 시각 → 동률이면 전환 불필요)
        5. oper_in_time
        """
        if not candidates:
            return None

        candidates = [c for c in candidates if self._lot_conv_discrete_eligible(eqp_id, c)]
        if not candidates:
            return None

        ppk = candidates[0].get("PLAN_PROD_ATTR_VAL", "")
        oper_id = candidates[0].get("oper_id", "")
        active_logical_lots = self._active_logical_lot_ids(ppk, oper_id)

        def sort_key(lot: dict):
            end, needs_conv, carrier, lot_id = self.earliest_st_combo_score(eqp_id, lot)
            logical_id = self._logical_lot_id(lot)
            lot_in_progress = logical_id in active_logical_lots
            return (
                not lot.get("is_initial_wip", False),
                int(lot.get("is_abstract", True)),
                not lot_in_progress,
                end,
                needs_conv,
                int(lot.get("oper_in_time", 0)),
                carrier,
                lot_id,
            )

        best = min(candidates, key=sort_key)
        return best["lot_id"]

    def assign_ppk_oper(self, eqp_id: str, ppk: str, oper_id: str) -> float:
        """(PPK, OPER) 선택 후 LOT 자동 배정."""
        lots = [
            l for l in self.available_lots(eqp_id)
            if l["PLAN_PROD_ATTR_VAL"] == ppk and l["oper_id"] == oper_id
        ]
        lot_id = self._auto_select_lot(eqp_id, lots)
        if lot_id is None:
            return -1.0
        lot_cd, temp = self._lot_cd_temp(
            lot_id, self.lot_pool.get(lot_id), ppk=ppk, oper_id=oper_id,
        )
        if self._assign_blocked(eqp_id, lot_cd, temp):
            return -1.0
        return self.assign_lot(eqp_id, lot_id)

    # ── 벌크(블록) 점유 지원 ───────────────────────────────────────────────

    def _carrier_takt_minutes(self, ppk: str, oper_id: str) -> float:
        """(PPK,OPER) carrier당 takt(분) = max(수요takt, capa_takt).

        get_observation()의 eff_takt와 동일 정의. 블록 크기 상한 산출용.
        """
        data = self._env_data
        arranges_by = data.get("abstract_arranges_by_ppk_oper", {})
        n_eqp_per_oper = data.get("n_eqp_per_oper", {})
        plan_meta = data.get("plan_meta", {})
        wf_unit = max(data.get("max_wf_qty", 1), 1)
        completed = self.stats["completed_qty"]
        T_avail = max(self.soft_cutoff - self.current_time, 1)

        lst = arranges_by.get((ppk, oper_id))
        spw = (sum(s for _, s in lst) / len(lst)) if lst else None
        n = max(n_eqp_per_oper.get(oper_id, 0), 1)
        cap_takt = (spw / n) if spw is not None else 0.0
        pm = plan_meta.get((ppk, oper_id))
        if pm and pm.get("d0_plan_qty", 0) > 0:
            q_plan = max(pm["d0_plan_qty"] - completed.get((ppk, oper_id), 0), 1)
            demand_takt = T_avail / q_plan
            return max(demand_takt, cap_takt) * wf_unit
        return cap_takt * wf_unit

    def _takt_budget_carriers(self, ppk: str, oper_id: str) -> int:
        """이번 horizon에서 takt상 허용되는 carrier 수 = floor(T_avail / takt)."""
        takt = self._carrier_takt_minutes(ppk, oper_id)
        T_avail = max(self.soft_cutoff - self.current_time, 1)
        if takt <= 0:
            return 10 ** 6  # takt 미정의 → 사실상 무제한(다른 상한이 묶음)
        return max(int(T_avail // takt), 1)

    def model_breadth(self, eqp_id: str) -> int:
        """이 장비 모델이 가공 가능한 (PPK,OPER) 조합 수 = 범용성(클수록 범용)."""
        model = self._eqp_model_map.get(eqp_id)
        arrange_map = self._env_data.get("abstract_arrange_map", {})
        n = sum(1 for (_p, _o, m) in arrange_map if m == model)
        if n > 0:
            return n
        return len(self._env_data.get("eqp_oper_cap", {}).get(eqp_id, []))

    def narrower_idle_specialist_exists(
        self, eqp_id: str, ppk: str, oper_id: str,
    ) -> bool:
        """이 버킷을 처리 가능한 '더 전용적인(breadth 작은)' 장비가 현재 idle인가.

        True면 범용 장비 eqp_id가 이 버킷을 잡을 때 전용 장비가 놀게 되므로
        벌크 보상에서 '전용 오용' 페널티 신호로 사용.
        """
        my_breadth = self.model_breadth(eqp_id)
        for other in self._env_data["eqp_ids"]:
            if other == eqp_id:
                continue
            if self.eqps[other].status != "idle":
                continue
            if not self._eqp_can_process(other, ppk, oper_id):
                continue
            if self.model_breadth(other) < my_breadth:
                return True
        return False

    def bulk_block_size(
        self, eqp_id: str, ppk: str, oper_id: str, level: int, n_levels: int,
    ) -> int:
        """size_level(0..n_levels-1) → 한 번에 커밋할 블록 carrier 수.

        상한 = min(takt 예산, 가용 WIP, 잔여 계획).

        주의: tool capacity는 '동시에 같은 LOT_CD를 돌리는 장비 수' 제한(동시성)
        이지, 한 장비가 순차 처리하는 블록 길이 제한이 아니다. 한 장비는 블록
        내내 tool 슬롯을 1개만(그것도 carrier 사이 점유/해제) 쓰므로 블록 길이를
        tool 잔여로 깎으면 안 된다. 동시성은 블록 재생 중 매 carrier마다
        _tool_cap_blocks(can_assign, 잔여 기준)가 검사해 초과 시 블록을 끊는다.
        """
        lots = [
            l for l in self.available_lots(eqp_id)
            if l["PLAN_PROD_ATTR_VAL"] == ppk and l["oper_id"] == oper_id
        ]
        wip_carriers = len(lots)
        if wip_carriers == 0:
            return 0

        data = self._env_data
        wf_unit = max(data.get("max_wf_qty", 1), 1)
        pm = data.get("plan_meta", {}).get((ppk, oper_id))
        if pm and pm.get("d0_plan_qty", 0) > 0:
            done = self.stats["completed_qty"].get((ppk, oper_id), 0)
            plan_carriers = int(np.ceil(max(pm["d0_plan_qty"] - done, 0) / wf_unit))
        else:
            plan_carriers = wip_carriers  # 계획 없으면 WIP 한도

        cap = min(wip_carriers, plan_carriers)
        if cap <= 0:
            return 0

        budget = self._takt_budget_carriers(ppk, oper_id)
        frac = (level + 1) / max(n_levels, 1)        # level0→작게 … 마지막→budget 전량
        target = max(int(round(frac * budget)), 1)
        return max(min(target, cap), 1)

    def bulk_decision_shaping(
        self, eqp_id: str, ppk: str, oper_id: str, block_size: int,
    ) -> float:
        """벌크 블록 '시작' 결정에 대한 추가 보상 shaping (SchedulingRLEnv 전용).

        세 항(모두 가중치 0이면 비활성):
          ① 블록 크기 보너스(+): 같은 제품군을 큰 블록으로 커밋할수록 보상.
          ② 전용 오용 페널티(−): 범용 장비가 더 전용적인 idle 장비도 가능한
             버킷을 잡으면 감점(전용 장비가 놀지 않게).
          ③ 중복 커버 페널티(−): 이미 다른 셋업 장비가 horizon 내 충분히
             덮는 버킷을 잡으면 감점(다른 제품으로 전환할 '용기').
        """
        cfg = self._reward_cfg
        shaping = 0.0
        bulk_terms: Dict[str, float] = {}

        # ① 블록 크기 보너스: takt 예산(takt_budget_carriers) 대비 블록 크기 비율
        #    식: shaping += w_bulk_block_bonus * min(block_size / budget, 1.0)
        if cfg.w_bulk_block_bonus > 0 and block_size > 1:
            budget = max(self._takt_budget_carriers(ppk, oper_id), 1)
            t = cfg.w_bulk_block_bonus * min(block_size / budget, 1.0)
            shaping += t
            bulk_terms["bulk_block_bonus"] = round(float(t), 4)

        # ② 전용 오용: 더 전용적인(breadth 작은) idle 장비가 있는데 범용 장비가
        #    이 버킷을 잡으면 고정 패널티 (narrower_idle_specialist_exists=True일 때만)
        if cfg.w_dedication_misuse < 0 and self.narrower_idle_specialist_exists(
            eqp_id, ppk, oper_id,
        ):
            shaping += cfg.w_dedication_misuse
            bulk_terms["dedication_misuse"] = round(float(cfg.w_dedication_misuse), 4)

        # ③ 중복 커버: 다른 셋업 장비가 day-end까지 투영 생산할 수 있는 양(cover)이
        #    잔여 필요량(need)을 얼마나 커버하는지 비율로 패널티.
        #    need = max(target-done, 1)  (0-division 방지, 별도 분기 아님)
        #    shaping += w_redundant_cover * min(cover/need, 2.0)  (cap=2.0)
        if cfg.w_redundant_cover < 0:
            done = self.stats["completed_qty"].get((ppk, oper_id), 0)
            target = max(self._achievable_qty(ppk, oper_id), 1)
            need = max(target - done, 1)
            cover = self._bucket_projected_cover(ppk, oper_id, exclude_eqp=eqp_id)
            t = cfg.w_redundant_cover * min(cover / need, 2.0)
            shaping += t
            bulk_terms["redundant_cover"] = round(float(t), 4)

        # 직전 배정의 리워드 분해에 벌크 항 병합 (decision_log 디버그용)
        if bulk_terms:
            self._last_reward_breakdown.update(bulk_terms)
            if self._last_decision_assignment is not None:
                bd = self._last_decision_assignment.setdefault("reward_breakdown", {})
                bd.update(bulk_terms)

        return shaping

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

    def get_feasible_ppk_oper(self, eqp_id: str) -> List[int]:
        """지정 EQP에서 유효한 ppk_oper_flat_idx 목록. 버전 기반으로 캐싱."""
        cached = self._feasible_cache.get(eqp_id)
        if cached is not None and cached[0] == self._state_version:
            return cached[1]
        result = self._compute_feasible_ppk_oper(eqp_id)
        self._feasible_cache[eqp_id] = (self._state_version, result)
        return result

    def _compute_feasible_ppk_oper(self, eqp_id: str) -> List[int]:
        """get_feasible_ppk_oper() 실계산 본체."""
        return [
            self.ppk_oper_flat_index(oper_id, ppk)
            for ppk, oper_id in self._eqp_feasible_bucket_keys(eqp_id)
        ]

    def get_feasible_assignments(self) -> List[tuple]:
        """유효 (ppk_oper_flat_idx, eqp_idx) 목록. 하위 호환."""
        data = self._env_data
        eqp_ids = data["eqp_ids"]
        feasible = []
        for ei, eqp_id in enumerate(eqp_ids):
            for flat in self.get_feasible_ppk_oper(eqp_id):
                feasible.append((flat, ei))
        return feasible

    def get_idle_eqps(self) -> List[str]:
        return [
            eid for eid in self._env_data["eqp_ids"]
            if self.eqps[eid].status == "idle"
            and self.get_feasible_ppk_oper(eid)
        ]

    # --- 공개 API ---

    def current_idle_eqp(self) -> Optional[str]:
        """
        목적: 현재 결정 대기 중인 idle EQP ID 반환
        Input:  없음
        Output: "EQP001" 또는 None (시뮬레이션 종료)
        """
        return self._current_eqp

    def _lot_candidates_for_eqp(self, eqp_id: str) -> List[dict]:
        """EQP에 배정 가능한 WIP LOT 후보 (abstract arrange + WIP 풀)."""
        proc_time_matrix = self._env_data.get("proc_time_matrix", {})
        busy_carriers = {
            v["carrier_id"]
            for v in self._in_flight.values()
            if v.get("carrier_id")
        }
        lots: List[dict] = []
        for row in self._abstract_assignable_on_eqp(eqp_id):
            ppk, oper_id = row["PLAN_PROD_ATTR_VAL"], row["oper_id"]
            wip = self._wip_for(ppk, oper_id)
            if not wip:
                continue
            for lid in list(wip["lot_ids"]):
                if not self._is_current_wip_lot(lid, ppk, oper_id):
                    continue
                meta = self._wip_lot_meta.get(lid, {})
                lot = self.lot_pool.get(lid)
                if not meta and not lot:
                    continue
                carrier = meta.get("carrier_id") or (lot.carrier_id if lot else "")
                if carrier and carrier in busy_carriers:
                    continue
                oper_in_time = meta.get("oper_in_time", 0)
                if not self._lot_ready(lid, oper_in_time):
                    continue
                has_discrete = self._has_discrete_combo(eqp_id, lid, oper_id)
                wf_qty = meta.get("wf_qty", lot.wf_qty if lot else row["wf_qty"])
                st_per_wafer = proc_time_matrix.get(
                    (lid, eqp_id, oper_id), row["proc_time"],
                )
                pt = effective_proc_time(st_per_wafer, wf_qty)
                lot_cd, temp = self._lot_cd_temp(
                    lid, lot, ppk=ppk, oper_id=oper_id,
                )
                lots.append({
                    "lot_id":          lid,
                    "carrier_id":      carrier,
                    "PLAN_PROD_ATTR_VAL": meta.get("PLAN_PROD_ATTR_VAL", lot.PLAN_PROD_ATTR_VAL if lot else ppk),
                    "oper_id":         meta.get("oper_id", lot.oper_id if lot else oper_id),
                    "wf_qty":          wf_qty,
                    "priority":        (
                        lot.priority if lot
                        else meta.get("priority", row.get("plan_priority", 1))
                    ),
                    "processing_time": pt,
                    "st_per_wafer":    st_per_wafer,
                    "parent_lot_id":   lot.parent_lot_id if lot else meta.get("parent_lot_id", ""),
                    "logical_lot_id": (
                        lot.logical_lot_id if lot
                        else meta.get("logical_lot_id", meta.get("parent_lot_id", lid))
                    ),
                    "lot_cd":          lot_cd,
                    "temp":            temp,
                    "seq":             meta.get("seq", lot.seq if lot else row["seq"]),
                    "is_abstract":     not has_discrete,
                    "is_initial_wip":  lid in self._initial_lot_ids,
                    "oper_in_time":    oper_in_time,
                    "lot_stat_cd":     (
                        lot.lot_stat_cd if lot else meta.get("lot_stat_cd", "WAIT")
                    ),
                })
        return lots

    def _forced_lot_pending(self, eqp_id: str) -> Optional[str]:
        """LOT_STAT_CD!=WAIT carrier 중 이 EQP에서 다음에 강제 배정할 carrier."""
        eqp = self.eqps.get(eqp_id)
        if eqp is not None and eqp.status == "idle":
            staged = self._eqp_staged_forced.get(eqp_id)
            if staged:
                return staged[0]
        queue = self._eqp_forced_queue.get(eqp_id)
        return queue[0] if queue else None

    def available_lots(self, eqp_id: str) -> List[dict]:
        """
        목적: 에이전트 선택을 위한 LOT 상세 정보 리스트 반환.
        abstract WIP 풀 + MODEL arrange 기준 (discrete eqp 없어도 가능).

        LOT_STAT_CD!=WAIT LOT은 지정된 EQP에서만, 입력 순서대로 한 번에 하나씩만 노출된다
        (다른 EQP·다른 순번에서는 후보에서 완전히 제외되어 알고리즘이 자유 배정할 수 없다).
        """
        lots = self._lot_candidates_for_eqp(eqp_id)
        forced_lot_id = self._forced_lot_pending(eqp_id)
        lots = [
            l for l in lots
            if l.get("lot_stat_cd", "WAIT") == "WAIT" or l["lot_id"] == forced_lot_id
        ]
        if forced_lot_id is not None:
            lots = [l for l in lots if l["lot_id"] == forced_lot_id]
        for item in lots:
            if not item.get("lot_cd"):
                lot_cd, temp = self._lot_cd_temp(
                    item["lot_id"],
                    ppk=item.get("PLAN_PROD_ATTR_VAL"),
                    oper_id=item.get("oper_id"),
                )
                item["lot_cd"] = lot_cd
                item["temp"] = temp
        return lots

    def _has_pending_processing(self) -> bool:
        if self._in_flight:
            return True
        if self._eqp_pending_assign:
            return True
        return any(
            e.status in ("busy", "converting") for e in self.eqps.values()
        )

    def _finalize_assignment(
        self,
        eqp_id: str,
        pending: dict,
        *,
        start_time: int,
        proc_reward: float,
    ) -> float:
        """tool 점유 + 가공 스케줄 등록 (즉시 또는 conv_end 후)."""
        eqp = self.eqps[eqp_id]
        lot_id = pending["lot_id"]
        ppk = pending["ppk"]
        oper_id = pending["oper_id"]
        seq = pending["seq"]
        wf_qty = pending["wf_qty"]
        proc_time = pending["proc_time"]
        st_per_wafer = pending.get("st_per_wafer", proc_time)
        carrier_id = pending["carrier_id"]
        row = pending["row"]
        oper_in_time = pending["oper_in_time"]
        is_abstract = pending["is_abstract"]
        lot_cd = pending["lot_cd"]
        temp = pending["temp"]
        lot_stat_cd = pending.get("lot_stat_cd", "WAIT")

        end_time = start_time + proc_time
        plan_hit = self._plan_hit_reward(ppk, oper_id, wf_qty, end_time)
        reward = proc_reward + plan_hit
        # plan_hit를 직전 결정의 리워드 분해에 추가(같은 LOT일 때만)
        if plan_hit:
            lb = self._last_decision_assignment
            if lb is not None and lb.get("lot_id") == lot_id:
                lb.setdefault("reward_breakdown", {})["plan_hit"] = round(float(plan_hit), 4)
                self._last_reward_breakdown["plan_hit"] = round(float(plan_hit), 4)

        from_lot_cd = eqp.prev_lot_cd
        from_temp = eqp.prev_temp

        eqp.status = "busy"
        eqp.current_lot = lot_id
        eqp.current_oper = oper_id
        eqp.current_prod = ppk
        eqp.free_at = end_time
        eqp.prev_oper = oper_id
        eqp.prev_prod = ppk
        eqp.prev_lot_id = lot_id
        eqp.prev_lot_cd = lot_cd
        eqp.prev_temp = temp
        self._invalidate_caches()

        had_conv = pending.get("had_conversion", False)
        if not had_conv and from_lot_cd is None and lot_cd:
            self._tool_tracker.occupy(lot_cd, eqp_id)
        self._push_event(end_time, EVENT_PROCESS_END, eqp_id)

        self._in_flight[lot_id] = {
            "PLAN_PROD_ATTR_VAL": ppk,
            "oper_id":       oper_id,
            "seq":           seq,
            "wf_qty":        wf_qty,
            "carrier_id":    carrier_id,
            "logical_lot_id": pending.get("logical_lot_id") or self._logical_lot_id(lot_id=lot_id),
            "end_time":      end_time,
            "lot_cd":        lot_cd,
            "eqp_id":        eqp_id,
        }

        self.stats["completed_qty"][(ppk, oper_id)] = (
            self.stats["completed_qty"].get((ppk, oper_id), 0) + wf_qty
        )

        self.schedule.append({
            "EQP_ID":        eqp_id,
            "LOT_ID":        pending.get("logical_lot_id") or lot_id,
            "CARRIER_ID":    carrier_id,
            "PLAN_PROD_ATTR_VAL": ppk,
            "OPER_ID":       oper_id,
            "ST":            st_per_wafer,
            "EQP_MODEL":     row["eqp_model"],
            "SEQ":           seq,
            "START_TM":      start_time,
            "END_TM":        end_time,
            "PROC_TIME":     proc_time,
            "WF_QTY":        wf_qty,
            "LOT_CD":        lot_cd,
            "TEMP":          temp,
            "CONVERSION":    had_conv,
            "ABSTRACT":      is_abstract,
            "OPER_IN_TIME":  oper_in_time,
            "LOT_STAT_CD":   lot_stat_cd,
        })

        self._last_assigned = {
            "kind":          "abstract" if is_abstract else "actual",
            "eqp_id":        eqp_id,
            "lot_id":        lot_id,
            "oper_id":       oper_id,
            "PLAN_PROD_ATTR_VAL": ppk,
            "eqp_model":     row["eqp_model"],
            "st":            st_per_wafer,
            "wf_qty":        wf_qty,
            "lot_cd":        lot_cd,
            "temp":          temp,
            "lot_stat_cd":   lot_stat_cd,
            "conversion":    had_conv,
            "start_tm":      start_time,
            "oper_in_time":  oper_in_time,
            "abs_key":       row.get("abs_key"),
        }
        self._select_same_time_next_eqp()
        return reward

    def _execute_assignment(
        self,
        eqp_id: str,
        lot_id: str,
        ppk: str,
        oper_id: str,
        seq: int,
        wf_qty: int,
        st_per_wafer: int,
        carrier_id: str,
        row: dict,
        oper_in_time: int,
        is_abstract: bool,
        lot_stat_cd: str = "WAIT",
    ) -> float:
        """공통 배정: conversion + tool + WIP -1."""
        eqp = self.eqps[eqp_id]
        cfg = self._reward_cfg
        reward = 0.0
        proc_time = effective_proc_time(st_per_wafer, wf_qty)

        lot_cd, temp = self._lot_cd_temp(lot_id, ppk=ppk, oper_id=oper_id)
        if self._assign_blocked(eqp_id, lot_cd, temp):
            return -1.0

        wip = self._wip_for(ppk, oper_id)
        if not wip or wip["wip_qty"] <= 0:
            return -1.0

        # 리워드 항목별 분해(디버그용) — 각 항의 기여분을 개별 기록
        terms: Dict[str, float] = {}
        t = self._same_setup_reward(eqp, ppk, oper_id, wf_qty)
        if t: terms["same_setup"] = round(float(t), 4)
        reward += t
        t = self._pacing_shaping_reward(ppk, oper_id, wf_qty, eqp_id=eqp_id)
        if t: terms["pacing"] = round(float(t), 4)
        reward += t
        t = self._flow_balance_reward(ppk, oper_id)                  # Step B
        if t: terms["flow_balance"] = round(float(t), 4)
        reward += t

        self._emit_event(
            EVENT_JOB_ASSIGNED, eqp_id,
            lot_id=lot_id,
            lot_cd=lot_cd,
            PLAN_PROD_ATTR_VAL=ppk,
            oper_id=oper_id,
        )

        conv_start, conv_end, reward, needs_conv = self._apply_conversion_start(
            eqp, lot_cd, temp, reward,
            oper_id=oper_id, ppk=ppk, eqp_model=row["eqp_model"],
        )
        terms.update(self._cur_conv_terms)
        self._last_reward_breakdown = terms
        self._consume_wip(ppk, oper_id, lot_id)

        # LOT_STAT_CD 강제 큐/선부착 소진
        self._consume_forced_placement(eqp_id, lot_id)

        pending = {
            "lot_id": lot_id,
            "ppk": ppk,
            "oper_id": oper_id,
            "seq": seq,
            "wf_qty": wf_qty,
            "st_per_wafer": st_per_wafer,
            "proc_time": proc_time,
            "carrier_id": carrier_id,
            "logical_lot_id": self._logical_lot_id(lot_id=lot_id),
            "row": row,
            "oper_in_time": oper_in_time,
            "is_abstract": is_abstract,
            "lot_cd": lot_cd,
            "temp": temp,
            "from_lot_cd": eqp.prev_lot_cd,
            "had_conversion": needs_conv,
            "proc_reward": reward,
            "lot_stat_cd": lot_stat_cd,
        }
        self._last_decision_assignment = {
            "eqp_id":        eqp_id,
            "lot_id":        lot_id,
            "PLAN_PROD_ATTR_VAL": ppk,
            "oper_id":       oper_id,
            "eqp_model":     row["eqp_model"],
            "st":            st_per_wafer,
            "wf_qty":        wf_qty,
            "lot_cd":        lot_cd,
            "temp":          temp,
            "conversion":    needs_conv,
            "start_tm":      conv_end if needs_conv else conv_start,
            "oper_in_time":  oper_in_time,
            "reward_breakdown": dict(terms),
        }

        if needs_conv:
            eqp.status = "converting"
            eqp.free_at = conv_end
            self._eqp_pending_assign[eqp_id] = pending
            self._push_event(conv_end, EVENT_CONV_END, eqp_id)
            self._current_eqp = None
            return reward

        return self._finalize_assignment(
            eqp_id, pending, start_time=conv_start, proc_reward=reward,
        )

    def assign_lot(self, eqp_id: str, lot_id: str) -> float:
        """LOT 배정. abstract WIP 풀에서 -1, conversion/tool 적용."""
        lot = self.lot_pool.get(lot_id)
        meta = self._wip_lot_meta.get(lot_id, {})

        if not meta and not lot:
            return -1.0

        if meta:
            ppk = meta["PLAN_PROD_ATTR_VAL"]
            oper_id = meta["oper_id"]
            seq = meta.get("seq", 1)
            wf_qty = meta.get("wf_qty", 25)
            carrier_id = meta.get("carrier_id", "")
            lot_stat_cd = lot.lot_stat_cd if lot else meta.get("lot_stat_cd", "WAIT")
        else:
            ppk, oper_id = lot.PLAN_PROD_ATTR_VAL, lot.oper_id
            seq, wf_qty, carrier_id = lot.seq, lot.wf_qty, lot.carrier_id
            lot_stat_cd = lot.lot_stat_cd

        row = self._abstract_row_for(eqp_id, ppk, oper_id)
        if row is None:
            return -1.0

        proc_time_matrix = self._env_data.get("proc_time_matrix", {})
        has_discrete = self._has_discrete_combo(eqp_id, lot_id, oper_id)

        if lot_stat_cd == "WAIT" and CONFIG.env.discrete_wait_enabled:
            lot_cd, temp = self._lot_cd_temp(lot_id, lot, ppk=ppk, oper_id=oper_id)
            eqp = self.eqps.get(eqp_id)
            needs_conv = self._would_need_conversion(eqp_id, lot_cd, temp)
            first_setup = eqp is None or eqp.prev_lot_cd is None
            if needs_conv or first_setup:
                # 전환 발생 시엔 기존 셋업 기준 discrete 조합이 더는 유효하지
                # 않으므로 abstract(모델 평균 ST)만 참조한다.
                is_abstract = True
                st_per_wafer = row["proc_time"]
            else:
                # 전환 없음 → EQP×carrier 정밀 조합(discrete)이 있어야만 배정 가능.
                if not has_discrete:
                    return -1.0
                is_abstract = False
                st_per_wafer = proc_time_matrix[(lot_id, eqp_id, oper_id)]
        else:
            is_abstract = not has_discrete
            st_per_wafer = proc_time_matrix.get((lot_id, eqp_id, oper_id), row["proc_time"])

        oper_in_time = meta.get("oper_in_time", 0)
        reward = self._execute_assignment(
            eqp_id, lot_id, ppk, oper_id, seq, wf_qty, st_per_wafer,
            carrier_id, row, oper_in_time, is_abstract, lot_stat_cd,
        )
        if reward < 0:
            return reward

        if lot is not None:
            for eid in self.eqp_queues:
                if lot_id in self.eqp_queues[eid]:
                    self.eqp_queues[eid].remove(lot_id)
            del self.lot_pool[lot_id]

        return reward

    def _append_initial_history(self):
        """에피소드 시작 시 arrange 전체 상태 스냅샷 (step 0)."""
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
            "events":     list(self._pending_step_events),
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
        self._pending_step_events = []

    def _has_assignable_work(self) -> bool:
        for eid in self._env_data["eqp_ids"]:
            if self._abstract_assignable_on_eqp(eid):
                return True
        return False

    def is_done(self) -> bool:
        """
        목적: 시뮬레이션 종료 — Actual·추상 투입 가능 조합 없고 처리 중인 재공도 없을 때.
        """
        if self._current_wip_only() and not self._has_unassigned_current_wip():
            return not self._eqp_pending_assign
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
        목적: UI 단계별 재생을 위해 현재 시뮬레이션 상태 스냅샷 저장.
        Input:  arrange_snapshot — 배정 직전 Actual 조합
                arrange_abstract_snapshot — 배정 직전 추상 투입 재공
                wip_waiting_snapshot — 배정 직전 대기 WIP (END_TM 반영 등)
        """
        self._step_idx += 1
        assigned = self._last_assigned
        self._last_assigned = None
        self._last_decision_assignment = None
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
            "events":     list(self._pending_step_events),
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
        self._pending_step_events = []

    def clear_step_assignment(self) -> None:
        """history 미기록 추론에서 step 임시 배정 상태만 정리."""
        self._step_idx += 1
        self._last_assigned = None
        self._last_decision_assignment = None
        self._pending_step_events = []

    # --- Bucket(=PPK×MODEL×OPER) feature ---
    PPK_OPER_FEATURES = 6
    PPK_OPER_MODEL_FEATURES = 5
    BUCKET_FEATURES = PPK_OPER_FEATURES + PPK_OPER_MODEL_FEATURES

    def get_bucket_features(self) -> np.ndarray:
        """
        Bucket = (oper, ppk, model) 단위 feature 텐서.
        po_feats: shape (O, P, 6) -> (PPK, OPER) 종속 피처 6개
        pom_feats: shape (O, P, K, 5) -> (PPK, OPER, MODEL) 종속 피처 5개
        이 둘을 각각 flatten하여 결합한 np.ndarray를 반환합니다.

        prev_takt/post_takt(인접 공정 takt), lot_cd/temp 인코딩 채널은 제거됨:
        cap_takt 계산이 n_eqp_per_oper(오퍼레이션당 총 capable 설비 수)를 쓰는데
        이 값이 설비 간 공유(한 설비가 여러 오퍼레이션 capable)를 고려하지 않고
        정적으로 고정돼 있어 실제 가용 capa와 어긋나는 값을 주입했고, lot_cd/temp는
        범주형 ID를 순서 있는 스칼라로 인코딩해 노이즈에 가까웠다. LOT_CD/TEMP
        전환 관련 정보는 pom_feats(needs_conversion/avoidable_frac 등)에 이미
        더 정확한 형태로 남아있다.
        """
        # 버전 + current_eqp 가 같으면 캐시 반환
        cache_state = (self._state_version, self._current_eqp)
        if self._bucket_feats_cache is not None and self._bucket_feats_state == cache_state:
            return self._bucket_feats_cache

        data = self._env_data
        cfg = CONFIG.env
        O, P, K = cfg.max_oper_count, cfg.max_prod_count, cfg.max_model_count

        eqp_models = data.get("eqp_models", [])
        eqp_model_map = data.get("eqp_model_map", {})
        arrange_map = data.get("abstract_arrange_map", {})
        arranges_by = data.get("abstract_arranges_by_ppk_oper", {})
        plan_meta = data.get("plan_meta", {})
        flow_prev = data.get("flow_prev", {})
        max_arrange_st = max(data.get("max_arrange_st", 1), 1)
        completed = self.stats["completed_qty"]
        current_eqp = self._current_eqp

        po_feats = np.zeros((O, P, self.PPK_OPER_FEATURES), dtype=np.float32)
        pom_feats = np.zeros((O, P, K, 5), dtype=np.float32)

        # (oper, model) → min(free_at) 사전 계산
        min_end_by_om: Dict[tuple, int] = {
            key: min(self.eqps[e].free_at for e in eqp_list)
            for key, eqp_list in self._eqps_by_om.items()
        }

        # current_eqp 의 모델/인덱스
        current_eqp_model: Optional[str] = eqp_model_map.get(current_eqp) if current_eqp else None
        current_mi: int = -1
        if current_eqp_model:
            try:
                idx = eqp_models.index(current_eqp_model)
                if idx < K:
                    current_mi = idx
            except ValueError:
                pass

        # WIP 집계 (ppk, oper)
        wip_po: Dict[tuple, float] = {}
        ppk_wip: Dict[str, float] = {}
        total_wip = 0.0
        for key, q in self.get_wip_waiting().items():
            ppk_w, op_w = key.split("|", 1)
            wip_po[(ppk_w, op_w)] = wip_po.get((ppk_w, op_w), 0.0) + q
            ppk_wip[ppk_w] = ppk_wip.get(ppk_w, 0.0) + q
            total_wip += q
        total_wip = max(total_wip, 1.0)

        T_avail = max(self.soft_cutoff - self.current_time, 1)

        def st_per_wafer(ppk: str, op: Optional[str], model: str) -> Optional[float]:
            if op is None:
                return None
            st = arrange_map.get((ppk, op, model))
            if st is not None:
                return float(st)
            lst = arranges_by.get((ppk, op))
            if lst:
                return sum(s for _, s in lst) / len(lst)
            return None

        oper_ids = data["oper_ids"]
        prod_keys = data["prod_keys"]
        for oi in range(min(O, len(oper_ids))):
            op = oper_ids[oi]
            curr_eqp_in_om: bool = (
                current_eqp is not None
                and current_eqp_model is not None
                and current_eqp in self._eqps_by_om.get((op, current_eqp_model), [])
            )
            for pi in range(min(P, len(prod_keys))):
                ppk = prod_keys[pi]
                is_step = (
                    (ppk, op) in arranges_by
                    or (ppk, op) in plan_meta
                    or (ppk, op) in wip_po
                )
                if not is_step:
                    continue

                wip_q = wip_po.get((ppk, op), 0.0)
                ppk_total = max(ppk_wip.get(ppk, 0.0), 1.0)
                urgency = 0.0
                pm = plan_meta.get((ppk, op))
                if pm and pm.get("d0_plan_qty", 0) > 0:
                    plan_qty = max(pm["d0_plan_qty"], 1)
                    gap = max(plan_qty - completed.get((ppk, op), 0), 0)
                    urgency = min(gap / T_avail / plan_qty, 1.0)
                ppk_fp = flow_prev.get(ppk, {})

                # Step C: achievable_ratio
                achievable_ratio = 1.0
                if pm and pm.get("d0_plan_qty", 0) > 0:
                    plan_qty = max(pm["d0_plan_qty"], 1)
                    done = completed.get((ppk, op), 0)
                    reachable = wip_po.get((ppk, op), 0.0)
                    seen = {op}
                    prev_op = ppk_fp.get(op)
                    while prev_op and prev_op not in seen:
                        seen.add(prev_op)
                        reachable += wip_po.get((ppk, prev_op), 0.0)
                        prev_op = ppk_fp.get(prev_op)
                    achievable_ratio = min(min(plan_qty, done + reachable) / plan_qty, 1.0)

                # LOT_CD / TEMP (pom_feats의 conversion 관련 채널 계산에 필요)
                lc, tp = self._bucket_lot_cd_temp(ppk, op)

                # po_feats 할당
                po_feats[oi, pi, 0] = wip_q / total_wip
                po_feats[oi, pi, 1] = wip_q / ppk_total
                po_feats[oi, pi, 2] = urgency
                po_feats[oi, pi, 3] = achievable_ratio

                done11 = completed.get((ppk, op), 0)
                need11 = max(self._achievable_qty(ppk, op) - done11, 1.0)
                cov11 = self._bucket_projected_cover(ppk, op, exclude_eqp=current_eqp)
                po_feats[oi, pi, 4] = min(cov11 / need11, 2.0) / 2.0

                consume_rate = self._oper_capacity_per_min(ppk, op)
                supply_rate = self._oper_supply_rate(ppk, op)
                if consume_rate > supply_rate:
                    net_rate = consume_rate - supply_rate
                    starve_time = wip_q / net_rate
                    # 계층값: 소진까지 남은 시간을 1시간/2시간/4시간 구간으로 나눠
                    # 0 / 0.25 / 0.5 / 0.75 중 하나로 이산화 (급할수록 값이 작음)
                    if starve_time <= 60:
                        starve_time_norm = 0.0
                    elif starve_time <= 120:
                        starve_time_norm = 0.25
                    elif starve_time <= 240:
                        starve_time_norm = 0.5
                    else:
                        starve_time_norm = 0.75
                else:
                    starve_time_norm = 1.0
                po_feats[oi, pi, 5] = starve_time_norm

                # 유효 모델 인덱스·값을 수집한 뒤 pom_feats 할당
                valid_mis: List[int] = []
                valid_sts: List[float] = []
                for mi in range(min(K, len(eqp_models))):
                    model = eqp_models[mi]
                    if min_end_by_om.get((op, model)) is None:
                        continue
                    st = st_per_wafer(ppk, op, model)
                    valid_mis.append(mi)
                    valid_sts.append(float(st) if st is not None else 0.0)

                if not valid_mis:
                    continue

                vmis = np.array(valid_mis, dtype=np.intp)
                vsts = np.array(valid_sts, dtype=np.float32)

                # pom_feats: 채널 4 (index 0) 할당
                pom_feats[oi, pi, vmis, 0] = vsts / max_arrange_st

                # pom_feats: 채널 8 (index 1), 채널 9 (index 2), avoidable_frac (index 3),
                # setup_changed (index 4) 할당
                if curr_eqp_in_om and current_mi >= 0 and current_mi in valid_mis:
                    eqp_obj = self.eqps.get(current_eqp)
                    if eqp_obj is not None:
                        setup_changed = (eqp_obj.prev_prod != ppk) or (eqp_obj.prev_oper != op)
                        pom_feats[oi, pi, current_mi, 4] = 1.0 if setup_changed else 0.0
                    if lc:
                        needs_conv = self._would_need_conversion(current_eqp, lc, tp)
                        pom_feats[oi, pi, current_mi, 1] = 1.0 if needs_conv else 0.0
                        pom_feats[oi, pi, current_mi, 2] = (
                            1.0 if not self._needs_tool_swap(current_eqp, lc, tp)
                            or self._tool_tracker.can_assign(lc, current_eqp) else 0.0
                        )
                        if needs_conv:
                            pom_feats[oi, pi, current_mi, 3] = self._conversion_avoidable_fraction(
                                current_eqp, ppk, op, lc, tp,
                            )

        feats = np.concatenate([po_feats.flatten(), pom_feats.flatten()])
        self._bucket_feats_cache = feats
        self._bucket_feats_state = cache_state
        return feats

    # --- 관측 벡터 생성 (Global + Bucket) ---

    def get_observation(self) -> np.ndarray:
        """관측: Global(6) + Bucket(O×P×K×F).

        EQP local(prev_prod/prev_oper)과 Context(last_ppk/oper/eqp/lot_cd)는
        bucket의 pom_feats(needs_conversion/avoidable_frac 등)와 정보가 겹치거나
        전역 마지막 배정 스칼라로 뭉개져 있어 제거 — bucket 채널이 이미 더
        세밀한 정보를 제공한다.
        """
        data = self._env_data
        initial_lot_count = max(len(data["lots"]), 1)
        total_plan = max(
            sum(p["d0_plan_qty"] for p in data.get("plan", []) if p.get("d0_plan_qty", 0) > 0),
            0,
        )

        bucket = self.get_bucket_features().flatten()

        group_global = np.zeros(6, dtype=np.float32)
        group_global[0] = min(self.current_time / max(self.sim_end, 1), 1.0)
        group_global[1] = min(
            max(self.soft_cutoff - self.current_time, 0) / max(self.soft_cutoff, 1), 1.0,
        )
        group_global[2] = min(len(self.lot_pool) / initial_lot_count, 1.0)
        produced = sum(self.stats["completed_qty"].values())
        if total_plan > 0:
            group_global[3] = min(produced / total_plan, 1.0)
        else:
            group_global[3] = min(produced / max(self._initial_wip_total, 1), 1.0)
        conv_eqps = 0
        for eqp_id, eqp in self.eqps.items():
            rem = max(eqp.free_at - self.current_time, 0)
            if rem > 0 and eqp.status == "idle" and eqp.prev_lot_cd is not None:
                conv_eqps += 1
        group_global[4] = conv_eqps / max(len(self.eqps), 1)
        group_global[5] = min(self._tool_tracker.utilization(), 1.0)

        obs = np.concatenate([
            group_global,
            bucket,
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
