"""
simulation/simulator.py – 이산 사건 시뮬레이션(DES) 엔진
RL 환경의 내부 시뮬레이터로서 EQP 상태 및 LOT 배정 이력을 관리합니다.
"""
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    processing_time: int   # 분
    priority:        int   = 1
    original_eqp:    str   = ""


@dataclass
class Equipment:
    eqp_id:       str
    status:       str   = "idle"   # "idle" | "busy"
    current_lot:  Optional[str] = None
    current_oper: Optional[str] = None
    current_prod: Optional[str] = None
    free_at:      int   = 0
    prev_oper:    Optional[str] = None
    prev_prod:    Optional[str] = None
    idle_accum:   int   = 0   # 누적 Idle 분
    oper_switches:int   = 0
    prod_switches:int   = 0


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
            # agent picks action → lot_id
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

        # EQP 초기화
        self.eqps: Dict[str, Equipment] = {
            eid: Equipment(eqp_id=eid) for eid in data["eqp_ids"]
        }

        # LOT 풀 초기화
        self.lot_pool: Dict[str, Lot] = {}
        for ld in data["lots"]:
            self.lot_pool[ld["lot_id"]] = Lot(**{
                k: ld[k] for k in [
                    "lot_id", "carrier_id", "plan_prod_key", "oper_id",
                    "seq", "wf_qty", "processing_time", "priority", "original_eqp"
                ]
            })

        # EQP별 배정 가능 LOT 큐 (원본 복사)
        self.eqp_queues: Dict[str, List[str]] = {
            eid: list(lots)
            for eid, lots in data["eqp_lot_map"].items()
        }

        # 이벤트 큐: (time, eqp_id)
        self._event_q: list = []
        for eid in data["eqp_ids"]:
            heapq.heappush(self._event_q, (0, eid))

        # 완료 기록 및 통계
        self.schedule:   List[dict] = []
        self.stats = {
            "idle_total":     0,
            "oper_switches":  0,
            "prod_switches":  0,
            "completed_qty":  {},   # {(prod, oper): qty}
        }

        # UI 재생용 히스토리
        self.history: List[dict] = []
        self._step_idx = 0

        # 현재 결정 대상 EQP
        self._current_eqp: Optional[str] = None
        self._advance_to_next_decision()

    # ── 결정 포인트 탐색 ─────────────────────────────────────────────────────

    def _advance_to_next_decision(self):
        """
        목적: 다음 에이전트 결정이 필요한 시점(EQP idle + 큐 존재)까지 시간 전진
        Input:  없음
        Output: 없음 (self._current_eqp, self.current_time 갱신)
        """
        self._current_eqp = None
        while self._event_q:
            time, eqp_id = heapq.heappop(self._event_q)
            eqp = self.eqps[eqp_id]

            # Idle 누적 계산 (이전 free_at ~ 현재 이벤트 시간)
            if eqp.status == "busy" and time > eqp.free_at:
                pass  # 이 EQP는 이전 LOT이 완료되는 시각에 이벤트가 발생
            if eqp.status == "busy":
                eqp.status = "idle"
                idle_start = eqp.free_at
                eqp.free_at = time

            self.current_time = time

            available = self._get_available_lots(eqp_id)
            if available:
                self._current_eqp = eqp_id
                return

            # 큐가 비어 있으면 해당 EQP 스킵 (idle 누적)
            if time < self.sim_end:
                # 다음 이벤트 없이 idle로 남을 경우 – 종료 시각에 재검사
                heapq.heappush(self._event_q, (self.sim_end, eqp_id))

    def _get_available_lots(self, eqp_id: str) -> List[str]:
        """
        목적: EQP에 배정 가능하며 아직 할당되지 않은 LOT 목록 반환
        Input:  eqp_id="EQP001"
        Output: ["LOT001", "LOT003", ...]
        """
        return [lid for lid in self.eqp_queues.get(eqp_id, [])
                if lid in self.lot_pool]

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
        목적: 에이전트 선택을 위한 LOT 상세 정보 리스트 반환
        Input:  eqp_id="EQP001"
        Output: [{"lot_id":"LOT001","plan_prod_key":"PPK001","oper_id":"OPER001",
                  "wf_qty":25,"priority":1,"processing_time":120}, ...]
        """
        lots = []
        for lid in self._get_available_lots(eqp_id):
            lot = self.lot_pool[lid]
            lots.append({
                "lot_id":          lot.lot_id,
                "plan_prod_key":   lot.plan_prod_key,
                "oper_id":         lot.oper_id,
                "wf_qty":          lot.wf_qty,
                "priority":        lot.priority,
                "processing_time": lot.processing_time,
            })
        return lots

    def assign_lot(self, eqp_id: str, lot_id: str) -> float:
        """
        목적: 선택된 LOT을 EQP에 배정하고 즉각 보상 반환
        Input:  eqp_id="EQP001", lot_id="LOT001"
        Output: reward (float)
        """
        eqp = self.eqps[eqp_id]
        lot = self.lot_pool.get(lot_id)
        if lot is None:
            return -1.0

        cfg = self._reward_cfg
        reward = 0.0

        # Idle 패널티 (배정 전 idle 구간)
        idle_duration = self.current_time - eqp.free_at
        if idle_duration > 0:
            eqp.idle_accum += idle_duration
            self.stats["idle_total"] += idle_duration
            reward += cfg.w_idle_per_min * idle_duration

        # 동일 OPER 연속 보너스
        if eqp.prev_oper == lot.oper_id:
            reward += cfg.w_same_oper
        else:
            if eqp.prev_oper is not None:
                eqp.oper_switches += 1
                self.stats["oper_switches"] += 1

        # 동일 제품 연속 보너스
        if eqp.prev_prod == lot.plan_prod_key:
            reward += cfg.w_same_prod
        else:
            if eqp.prev_prod is not None:
                eqp.prod_switches += 1
                self.stats["prod_switches"] += 1

        # LOT 완료 보상
        reward += cfg.w_completion * lot.wf_qty / 25.0

        # EQP 상태 갱신
        start_time = self.current_time
        end_time   = start_time + lot.processing_time
        eqp.status       = "busy"
        eqp.current_lot  = lot_id
        eqp.current_oper = lot.oper_id
        eqp.current_prod = lot.plan_prod_key
        eqp.free_at      = end_time
        eqp.prev_oper    = lot.oper_id
        eqp.prev_prod    = lot.plan_prod_key

        # 이벤트 큐에 완료 이벤트 추가
        heapq.heappush(self._event_q, (end_time, eqp_id))

        # 큐에서 제거
        if lot_id in self.eqp_queues.get(eqp_id, []):
            self.eqp_queues[eqp_id].remove(lot_id)

        # LOT 풀에서 제거 (다른 EQP 큐에도 있을 수 있으므로 풀만 제거)
        del self.lot_pool[lot_id]

        # 완료 통계
        key = (lot.plan_prod_key, lot.oper_id)
        self.stats["completed_qty"][key] = (
            self.stats["completed_qty"].get(key, 0) + lot.wf_qty
        )

        # 스케줄 기록
        self.schedule.append({
            "EQP_ID":        eqp_id,
            "LOT_ID":        lot_id,
            "CARRIER_ID":    lot.carrier_id,
            "PLAN_PROD_KEY": lot.plan_prod_key,
            "OPER_ID":       lot.oper_id,
            "ST":            "A",
            "SEQ":           lot.seq,
            "START_TM":      start_time,
            "END_TM":        end_time,
        })

        # 다음 결정 포인트로 이동
        self._advance_to_next_decision()
        return reward

    def is_done(self) -> bool:
        """
        목적: 시뮬레이션 종료 여부 확인
        Input:  없음
        Output: True (모든 LOT 배정 완료 or 시간 초과)
        """
        return self._current_eqp is None

    def save_history_step(self):
        """
        목적: UI 단계별 재생을 위해 현재 시뮬레이션 상태 스냅숏 저장
        Input:  없음
        Output: 없음 (self.history에 추가)
        """
        self.history.append({
            "step":        self._step_idx,
            "time":        self.current_time,
            "schedule":    list(self.schedule),
            "completed":   dict(self.stats["completed_qty"]),
            "idle_total":  self.stats["idle_total"],
            "oper_sw":     self.stats["oper_switches"],
            "prod_sw":     self.stats["prod_switches"],
            "eqp_states":  {
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
        self._step_idx += 1

    # ── 관측 벡터 생성 ────────────────────────────────────────────────────────

    def get_observation(self) -> np.ndarray:
        """
        목적: 현재 시뮬레이션 상태를 RL 관측 벡터로 변환 (고정 크기)
        Input:  없음
        Output: np.ndarray shape=(obs_dim,) dtype=float32

        벡터 구조:
          [0]       sim_time_norm
          [1]       remaining_lots_norm
          [2]       global_idle_ratio
          [3]       global_achievement_rate
          [4]       prev_oper_norm (현재 idle EQP)
          [5]       prev_prod_norm
          [6]       eqp_idle_duration_norm
          [7:7+Q]   lot_oper_norm   (MAX_QUEUE)
          [7+Q:7+2Q] lot_prod_norm
          [7+2Q:7+3Q] lot_priority_norm
          [7+3Q:7+4Q] lot_wf_qty_norm
          [7+4Q:7+5Q] lot_same_oper_flag
          [7+5Q:7+6Q] lot_same_prod_flag
          [7+6Q:]   plan_achievement per prod (MAX_PRODS)
        """
        env_cfg = CONFIG.env
        Q  = env_cfg.max_queue_size
        P  = env_cfg.max_prod_count
        data = self._env_data

        obs = np.zeros(7 + 6 * Q + P, dtype=np.float32)

        # 전역 피처
        obs[0] = min(self.current_time / max(self.sim_end, 1), 1.0)
        obs[1] = len(self.lot_pool) / max(len(data["lots"]), 1)
        total_eqp_time = self.current_time * max(len(self.eqps), 1)
        obs[2] = self.stats["idle_total"] / max(total_eqp_time, 1)

        # 전체 계획 달성률
        total_plan = sum(p["d0_plan_qty"] for p in data["plan"])
        total_done = sum(self.stats["completed_qty"].values())
        obs[3] = min(total_done / max(total_plan, 1), 1.0)

        # 현재 idle EQP 피처
        eqp_id = self._current_eqp
        if eqp_id:
            eqp = self.eqps[eqp_id]
            obs[4] = encode_normalized(eqp.prev_oper, data["oper_idx"], env_cfg.max_oper_count)
            obs[5] = encode_normalized(eqp.prev_prod, data["prod_idx"], env_cfg.max_prod_count)
            obs[6] = min((self.current_time - eqp.free_at) / max(self.sim_end, 1), 1.0)

            # 배정 가능 LOT 피처 (최대 Q개 패딩)
            lots = self.available_lots(eqp_id)[:Q]
            for i, lot in enumerate(lots):
                obs[7 + i]         = encode_normalized(lot["oper_id"],        data["oper_idx"], env_cfg.max_oper_count)
                obs[7 + Q + i]     = encode_normalized(lot["plan_prod_key"],   data["prod_idx"], env_cfg.max_prod_count)
                obs[7 + 2*Q + i]   = lot["priority"] / 5.0
                obs[7 + 3*Q + i]   = lot["wf_qty"] / 50.0
                obs[7 + 4*Q + i]   = float(lot["oper_id"] == eqp.prev_oper)
                obs[7 + 5*Q + i]   = float(lot["plan_prod_key"] == eqp.prev_prod)

        # 제품별 계획 달성률 (최대 P개)
        for i, p in enumerate(data["plan"][:P]):
            key  = (p["plan_prod_key"], p["oper_id"])
            done = self.stats["completed_qty"].get(key, 0)
            obs[7 + 6*Q + i] = min(done / max(p["d0_plan_qty"], 1), 1.0)

        return obs
