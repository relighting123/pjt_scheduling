"""
agent/dedication_agent.py – Dedication(전담 배분) 휴리스틱

목표: 장비별로 (PPK, OPER) 버킷을 전담시키고, 전환 없이 같은 버킷을 쭉
연속 생산한다.

동작 원리 (모두 이벤트 시점에 재평가되는 룰 — 사전 고정 계획 없음):

1. 커버 기반 분산 배치 — 재공이 버킷별로 고르게 있으면, 이미 다른 장비가
   전담(commit)한 버킷은 "커버됨"으로 보고 남는 장비가 다음 버킷을 잡는다.
   → 버킷별 작업량에 비례해 장비 대수가 자연히 배분되고, 이후 각 장비는
   자기 버킷만 유지한다(장비 대수 정해두고 쭉 생산).

2. 데드라인 슬랙 기반 cascade 전환 — 재공이 앞 공정에 몰려 있으면 전 장비가
   앞 공정을 진행하다가, 뒤 공정 버킷의 슬랙
   (soft_cutoff까지 남은 시간 − 전환시간 − 미커버 잔여수량×장당ST)
   이 소진되는 시점(= 지금 출발해야 컷오프 안에 끝나는 마지막 순간)에만
   필요한 장비를 전환시킨다. 전환 후 그 버킷이 커버되면 나머지 장비는
   앞 공정을 유지한다.

3. HOLD(결정 보류) — 전환해 봐야 이미 다른 장비가 커버 중인 버킷뿐이거나,
   자기 버킷에 상류 유입이 곧 도착할 예정이면 배정 자체를 보류한다
   (SchedulingEnv의 HOLD_ACTION). 프레임워크의 "유휴 시 무조건 배정" 기본
   동작이 만드는 전환 채터링을 차단한다.

벤치마크의 오라클(BlockDedicationOracle/SafetySwitchOracle)이 손으로 넣던
전담 블록·전환 시점을 재공/계획/커버 정보로 자동 계산하는 일반화판이다.
"""
from typing import Dict, Optional

import numpy as np

from simulation.simulator import SchedulingSimulator

# SchedulingEnv.step()이 인식하는 "이번 결정 보류" 센티널 액션.
HOLD_ACTION = -1

# 슬랙이 이 값(분) 이하로 내려오면 '지금 전환해야 컷오프 안에 끝난다'로 판단.
# 0 = 슬랙 소진 시점 정확히(오라클과 동일). 안전 여유가 필요하면 env_data의
# dedication_switch_slack_minutes 로 키운다.
DEFAULT_SWITCH_SLACK_MINUTES = 0


class DedicationAgent:
    def __init__(self, env_data: dict):
        self._switch_slack_min = int(
            env_data.get("dedication_switch_slack_minutes", DEFAULT_SWITCH_SLACK_MINUTES)
        )
        # eqp_id → 전담 중인 버킷 flat. 에이전트 자체 장부라, 전환 진행 중
        # (simulator의 prev_prod가 아직 안 바뀐 구간)에도 커버 계산에 반영된다.
        self._committed: Dict[str, int] = {}

    # ── 버킷 평가 ─────────────────────────────────────────────────────────

    @staticmethod
    def _deadline(sim: SchedulingSimulator) -> int:
        """생산 마감 시각. soft_cutoff가 시뮬 종료보다 늦게 설정된 환경
        (짧은 벤치마크 시나리오 등)에서는 실제 종료가 마감이다."""
        return min(sim.soft_cutoff, sim.sim_end)

    def _committed_cover_wafers(
        self, sim: SchedulingSimulator, ppk: str, oper_id: str, flat: int, exclude_eqp: str,
    ) -> float:
        """이 버킷에 commit된 '다른' 장비들이 마감까지 생산 가능한 매수."""
        deadline = self._deadline(sim)
        total = 0.0
        for eid, f in self._committed.items():
            if f != flat or eid == exclude_eqp:
                continue
            eqp = sim.eqps.get(eid)
            if eqp is None:
                continue
            st = sim._st_per_wafer_for_eqp(eid, ppk, oper_id)
            if not st or st <= 0:
                continue
            avail = max(deadline - max(sim.current_time, eqp.free_at), 0)
            total += avail / st
        return total

    def _bucket_info(self, sim: SchedulingSimulator, eqp_id: str, flat: int) -> dict:
        ppk, oper_id = sim.ppk_oper_from_flat(flat)
        lot_cd, temp = sim._bucket_lot_cd_temp(ppk, oper_id)
        needs_conv = sim._would_need_conversion(eqp_id, lot_cd, temp)
        st = float(sim._st_per_wafer_for_eqp(eqp_id, ppk, oper_id) or 60.0)
        done = sim.stats["completed_qty"].get((ppk, oper_id), 0)
        need = max(sim._achievable_qty(ppk, oper_id) - done, 0)
        cover = self._committed_cover_wafers(sim, ppk, oper_id, flat, exclude_eqp=eqp_id)
        uncovered = max(need - cover, 0.0)
        time_left = max(self._deadline(sim) - sim.current_time, 0)
        conv_min = sim._conversion_minutes if needs_conv else 0
        # 슬랙: 지금 이 장비 혼자 미커버 잔여를 맡을 때 남는 시간 여유(분).
        slack = (time_left - conv_min) - uncovered * st
        prio = (
            sim._env_data.get("plan_meta", {})
            .get((ppk, oper_id), {})
            .get("priority", 99)
        )
        return {
            "flat": flat, "ppk": ppk, "oper_id": oper_id,
            "needs_conv": needs_conv, "st": st,
            "need": need, "uncovered": uncovered, "slack": slack, "prio": prio,
        }

    def _inflow_coming(self, sim: SchedulingSimulator, flat: int) -> bool:
        """이 버킷으로 상류 재공이 흘러올 예정인가 (inflow 활성 시)."""
        if not sim._enable_wip_inflow:
            return False
        ppk, oper_id = sim.ppk_oper_from_flat(flat)
        prev = sim._flow_prev(ppk, oper_id)
        if prev is None:
            return False
        if sim._ready_wip_qty(ppk, prev) > 0:
            return True
        return sim.get_in_flight_qty(ppk, prev) > 0

    # ── 선택 규칙 ─────────────────────────────────────────────────────────

    @staticmethod
    def _pick_order(b: dict) -> tuple:
        """계획 우선순위 → 미커버 작업량(분) 큰 순 → 빠른 ST → flat."""
        return (b["prio"], -(b["uncovered"] * b["st"]), b["st"], b["flat"])

    def _urgent_pull_target(
        self, sim: SchedulingSimulator, infos: Dict[int, dict], committed: int,
    ) -> Optional[int]:
        """생산 중인 장비를 다른 버킷으로 빼내야 하는가 (cascade 전환).

        조건: 대상 버킷이 미커버 상태로 슬랙이 소진 직전이고, 동시에
        '내 버킷의 잔여 목표를 남은 장비들이 커버 가능'(mine.uncovered ≤ 0 —
        cover 계산이 나를 제외하므로 정확히 이 의미) 또는 대상의 계획
        우선순위가 더 높을 때만. 이 커버 조건이 없으면 재공 과부하 상황
        (모든 버킷 slack이 깊은 음수)에서 슬랙 대소가 계속 뒤집히며 pull이
        난사돼 전환 핑퐁이 생긴다 — 전담을 깨는 건 남이 내 몫을 이어받을 수
        있을 때뿐이다."""
        mine = infos[committed]
        urgent = [
            b for f, b in infos.items()
            if f != committed
            and b["uncovered"] > 0
            and b["slack"] <= self._switch_slack_min
            and (mine["uncovered"] <= 0 or b["prio"] < mine["prio"])
        ]
        if not urgent:
            return None
        return min(urgent, key=lambda b: (b["slack"], self._pick_order(b)))["flat"]

    def _choose_new(
        self, sim: SchedulingSimulator, infos: Dict[int, dict], stale_committed: Optional[int],
    ) -> Optional[int]:
        """전담 버킷이 없거나 소진된 장비의 다음 버킷 선택. None = HOLD."""
        free = [b for b in infos.values() if not b["needs_conv"]]
        if free:
            # 전환 없는 배정은 항상 이득 — 미커버 버킷 우선, 없으면 그냥 유지 생산.
            pool = [b for b in free if b["uncovered"] > 0] or free
            return min(pool, key=self._pick_order)["flat"]

        conv = [b for b in infos.values() if b["uncovered"] > 0]
        if not conv:
            return None  # 남은 건 이미 커버된 버킷으로의 전환뿐 → 순수 채터링, 보류.

        if stale_committed is not None and self._inflow_coming(sim, stale_committed):
            # 내 버킷에 유입이 곧 도착 → 셋업 유지하고 대기. 단, 슬랙이 소진된
            # 버킷이 있으면 그쪽이 우선(컷오프 내 완료가 더 급함).
            conv = [b for b in conv if b["slack"] <= self._switch_slack_min]
            if not conv:
                return None

        # 진짜 소진(유입 없음) → 가장 급한 미커버 버킷으로 1회 전환.
        return min(conv, key=lambda b: (b["slack"], self._pick_order(b)))["flat"]

    # ── 인터페이스 ────────────────────────────────────────────────────────

    def predict(self, sim: SchedulingSimulator) -> np.ndarray:
        eqp_id = sim.current_idle_eqp()
        if eqp_id is None:
            return np.array([0], dtype=np.int64)
        feasible = sim.get_feasible_ppk_oper(eqp_id)
        if not feasible:
            return np.array([0], dtype=np.int64)

        infos = {f: self._bucket_info(sim, eqp_id, f) for f in feasible}
        committed = self._committed.get(eqp_id)

        if committed is not None and committed in infos:
            pull = self._urgent_pull_target(sim, infos, committed)
            target = pull if pull is not None else committed
            self._committed[eqp_id] = target
            return np.array([target], dtype=np.int64)

        choice = self._choose_new(sim, infos, committed)
        if choice is None:
            if sim._has_pending_processing():
                return np.array([HOLD_ACTION], dtype=np.int64)
            # 아무 가공도 진행 중이지 않으면 보류가 시간도 못 흘림 → 강제 선택.
            choice = min(infos.values(), key=self._pick_order)["flat"]
        self._committed[eqp_id] = choice
        return np.array([choice], dtype=np.int64)
