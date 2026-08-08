"""
simulation/eqp_capacity.py – (PPK, OPER)별 적정 장비 대수 산출 + 정원 마스킹

세 단계로 나뉜다.

1. **필요 대수 산출** — 마감(soft_cutoff)까지 남은 시간 안에 잔여량을 처리하려면
   장비가 몇 대 필요한가. 모델별 ST(장당 분)의 역수(1/ST = 분당 처리량)를 우선순위
   순으로 쌓아, 필요 처리율(잔여량 / 남은시간)에 도달하는 시점의 장비 수가 필요 대수다.
   모델마다 ST가 달라 "대수 × 평균 ST"로는 맞지 않으므로 장비를 한 대씩 채운다.

2. **재공 적정성 사전 체크** — 대수만 맞춰도 그 대수를 먹일 재공이 없으면 의미가 없다.
   ready carrier 수(한 대는 최소 1 carrier를 받아야 한다)와 재공 작업량(재공 매수 × ST)을
   기준으로, 한 대당 최소 `capacity_min_run_minutes`는 돌 수 있는 대수까지만 정원으로
   인정한다. 그보다 잘게 쪼개면 전환·셋업 시간이 가공 시간보다 커진다.

3. **마스킹** — 정원(target)이 정해지면 그 버킷을 지금 돌고 있거나(진행 중) 마지막
   셋업이 같은(진행했던) 장비는 항상 통과시키고, 그 외 신규 장비는 점유 수가 정원
   미만일 때만 통과시킨다. 결과적으로 "가급적 하던 장비 그대로, 적정 대수까지만" 운영된다.

정원 계산은 상태(state_version)가 바뀔 때마다 다시 하되 같은 상태에서는 캐시를 쓴다.
재공이 남아 있는 버킷의 정원은 최소 1대로 바닥을 두어(`_floor_one`), 마스킹 때문에
재공이 통째로 묶여 시뮬레이션이 멈추는 일이 없게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import CONFIG

# reason 코드 — 정원이 필요 대수보다 줄었다면 무엇이 묶었는지 남긴다.
REASON_OK = "OK"            # 필요 대수를 그대로 정원으로 인정
REASON_NO_WIP = "NO_WIP"    # 재공 없음 → 구동 불가
REASON_NO_EQP = "NO_EQP"    # 처리 가능 장비 없음
REASON_WIP_LIMIT = "WIP"    # 재공이 필요 대수를 못 먹임 → 정원 축소
REASON_EQP_LIMIT = "EQP"    # 처리 가능 장비 수가 필요 대수보다 적음
REASON_DONE = "DONE"        # 잔여량 0(계획 달성) — 재공이 남아 최소 1대만 유지


@dataclass
class BucketCapacity:
    """(PPK, OPER) 한 버킷의 적정 대수 산출 결과."""
    ppk:            str
    oper_id:        str
    horizon_min:    int                     # 마감까지 남은 시간(분)
    plan_qty:       int                     # D0 계획량
    done_qty:       int                     # 배정 완료 수량
    remain_qty:     int                     # 마감까지 처리해야 할 잔여량(재공 상한 반영)
    wip_qty:        int                     # ready 재공 매수
    wip_carriers:   int                     # ready 재공 carrier 수
    st_avg:         float                   # 처리 가능 장비 평균 ST(장당 분)
    req_cnt:        int                     # ST 기준 필요 대수(재공 미고려)
    target_cnt:     int                     # 재공 체크 후 확정 정원
    capable_cnt:    int                     # 처리 가능(다운 제외) 장비 수
    runnable:       bool                    # 지금 구동 가능한 조건인가
    reason:         str = REASON_OK
    req_eqps:       List[str] = field(default_factory=list)   # 필요 대수만큼의 우선순위 장비
    target_eqps:    List[str] = field(default_factory=list)   # 정원만큼의 우선순위 장비
    st_by_model:    Dict[str, float] = field(default_factory=dict)
    req_by_model:   Dict[str, int] = field(default_factory=dict)
    target_by_model: Dict[str, int] = field(default_factory=dict)


class CapacityPlanner:
    """시뮬레이터에 붙어 버킷별 정원을 산출하고 마스킹 판정을 제공한다."""

    def __init__(self, sim) -> None:
        self._sim = sim
        self._cache: Optional[Dict[Tuple[str, str], BucketCapacity]] = None
        self._cache_version: int = -1

    # ── 설정 ────────────────────────────────────────────────────────────────

    @property
    def mask_enabled(self) -> bool:
        """정원 마스킹 적용 여부. env_data로 시나리오별 override 가능."""
        return bool(self._sim._env_data.get(
            "eqp_capacity_mask_enabled", CONFIG.env.eqp_capacity_mask_enabled,
        ))

    @property
    def _min_run_minutes(self) -> int:
        return max(int(self._sim._env_data.get(
            "capacity_min_run_minutes", CONFIG.env.capacity_min_run_minutes,
        )), 1)

    # ── 정원 산출 ───────────────────────────────────────────────────────────

    def targets(self) -> Dict[Tuple[str, str], BucketCapacity]:
        """현재 상태 기준 버킷별 정원. state_version 동안 캐시."""
        sim = self._sim
        if self._cache is not None and self._cache_version == sim._state_version:
            return self._cache
        result = {key: self._compute_bucket(*key) for key in self._bucket_keys()}
        self._cache = result
        self._cache_version = sim._state_version
        return result

    def target_for(self, ppk: str, oper_id: str) -> Optional[BucketCapacity]:
        return self.targets().get((ppk, oper_id))

    def _bucket_keys(self) -> List[Tuple[str, str]]:
        """재공이 남아 있거나 계획이 있는 (PPK, OPER) 전부."""
        sim = self._sim
        keys = {
            key for key, pool in sim._wip_pool.items()
            if pool.get("wip_qty", 0) > 0
        }
        keys |= {
            key for key, meta in sim._env_data.get("plan_meta", {}).items()
            if meta.get("d0_plan_qty", 0) > 0
        }
        return sorted(keys)

    def _compute_bucket(self, ppk: str, oper_id: str) -> BucketCapacity:
        sim = self._sim
        # 마감시간 = soft_cutoff와 시뮬 종료 중 이른 쪽. 벤치마크처럼 sim_end를
        # 짧게 잡은 데이터에서 soft_cutoff(기본 1320분)를 그대로 쓰면 남은 시간을
        # 과대평가해 필요 대수가 실제보다 적게 나온다.
        deadline = min(sim.soft_cutoff, sim.sim_end)
        horizon = max(deadline - sim.current_time, 0)
        plan_qty = int(
            sim._env_data.get("plan_meta", {}).get((ppk, oper_id), {}).get("d0_plan_qty", 0)
        )
        done_qty = int(sim.stats["completed_qty"].get((ppk, oper_id), 0))
        wip_qty, wip_carriers = self._ready_wip(ppk, oper_id)
        in_flight_qty = self._in_flight_wafers(ppk, oper_id)

        # 잔여량: 계획이 있으면 (계획 − 실적)을 실제로 만들 수 있는 재공까지만 인정하고,
        # 계획이 없는 버킷은 남은 재공 자체가 마감까지 처리해야 할 양이다.
        if plan_qty > 0:
            remain_qty = min(max(plan_qty - done_qty, 0), wip_qty + in_flight_qty)
        else:
            remain_qty = wip_qty

        candidates = self._ranked_candidates(ppk, oper_id)
        st_by_model = self._st_by_model(ppk, oper_id, candidates)
        st_avg = (
            sum(st for _eid, st, _model in candidates) / len(candidates)
            if candidates else 0.0
        )

        req_cnt, fleet_short = self._required_count(candidates, remain_qty, horizon)
        target_cnt, reason = self._apply_wip_check(
            req_cnt, remain_qty, wip_qty, wip_carriers, st_avg, len(candidates),
            fleet_short=fleet_short,
        )

        req_eqps = [eid for eid, _st, _m in candidates[:req_cnt]]
        target_eqps = [eid for eid, _st, _m in candidates[:target_cnt]]
        return BucketCapacity(
            ppk=ppk,
            oper_id=oper_id,
            horizon_min=horizon,
            plan_qty=plan_qty,
            done_qty=done_qty,
            remain_qty=int(remain_qty),
            wip_qty=wip_qty,
            wip_carriers=wip_carriers,
            st_avg=round(st_avg, 4),
            req_cnt=req_cnt,
            target_cnt=target_cnt,
            capable_cnt=len(candidates),
            runnable=target_cnt > 0 and wip_carriers > 0,
            reason=reason,
            req_eqps=req_eqps,
            target_eqps=target_eqps,
            st_by_model=st_by_model,
            req_by_model=self._count_by_model(candidates[:req_cnt]),
            target_by_model=self._count_by_model(candidates[:target_cnt]),
        )

    def _required_count(
        self, candidates: List[Tuple[str, float, str]], remain_qty: int, horizon: int,
    ) -> Tuple[int, bool]:
        """필요 처리율(잔여량/남은시간)에 도달할 때까지 우선순위 순으로 장비를 채운다.

        Returns: (필요 대수, 장비를 다 써도 필요 처리율에 못 미치는지)
        """
        if remain_qty <= 0 or horizon <= 0 or not candidates:
            return 0, False
        need_rate = remain_qty / horizon          # 매/분
        rate = 0.0
        for n, (_eid, st, _model) in enumerate(candidates, start=1):
            rate += 1.0 / st
            if rate >= need_rate:
                return n, False
        return len(candidates), True

    def _apply_wip_check(
        self,
        req_cnt: int,
        remain_qty: int,
        wip_qty: int,
        wip_carriers: int,
        st_avg: float,
        capable_cnt: int,
        *,
        fleet_short: bool = False,
    ) -> Tuple[int, str]:
        """재공이 그 대수를 먹일 수 있는지 사전 체크 → 구동 가능한 정원으로 축소."""
        if capable_cnt <= 0:
            return 0, REASON_NO_EQP
        if wip_carriers <= 0:
            return 0, REASON_NO_WIP

        # 재공 작업량(분) / 한 대당 최소 가동시간 = 재공이 지탱 가능한 대수
        wip_minutes = wip_qty * st_avg if st_avg > 0 else 0.0
        by_workload = int(wip_minutes // self._min_run_minutes) if wip_minutes > 0 else 0
        wip_cap = min(wip_carriers, by_workload) if by_workload > 0 else min(wip_carriers, 1)

        target = min(req_cnt, wip_cap, capable_cnt)
        # 재공이 남아 있으면 최소 1대는 열어 둔다 — 계획을 이미 채웠거나(잔여 0)
        # 산출 대수가 0이라는 이유로 남은 재공이 통째로 묶이면 안 된다.
        target = max(target, 1)

        if remain_qty <= 0:
            return target, REASON_DONE
        if target < req_cnt:
            # req_cnt는 이미 처리 가능 장비 수를 넘지 않으므로, 정원이 그보다
            # 줄었다면 원인은 항상 재공이다.
            return target, REASON_WIP_LIMIT
        if fleet_short:
            # 있는 장비를 다 붙여도 마감까지 잔여량을 못 끝낸다(장비 부족).
            return target, REASON_EQP_LIMIT
        return target, REASON_OK

    # ── 후보 장비 우선순위 ──────────────────────────────────────────────────

    def _ranked_candidates(self, ppk: str, oper_id: str) -> List[Tuple[str, float, str]]:
        """이 버킷을 처리 가능한 장비를 배치 우선순위대로 [(eqp_id, st, model)] 반환.

        지금 이 버킷을 돌고 있는 장비 → 마지막 셋업이 같은 장비 → 전환이 필요 없는
        장비 순으로 앞세운다("가급적 지금 진행 중인 혹은 진행했던 장비를 그대로 배치").
        동순위는 ST가 짧은 장비, 그다음 범용성이 낮은(전용) 장비를 먼저 써서 범용
        장비를 다른 버킷 몫으로 남긴다.
        """
        sim = self._sim
        lot_cd, temp = sim._bucket_lot_cd_temp(ppk, oper_id)
        rows: List[Tuple[tuple, str, float, str]] = []
        for eqp_id in sim._env_data.get("eqp_ids", []):
            eqp = sim.eqps.get(eqp_id)
            if eqp is None or eqp.status == "down":
                continue
            # 모델이 없는 EQP(eqp_queue_init에만 등장 — EQP_MODEL_CD는
            # discrete_arrange에서만 나온다)는 ST를 알 수 없어 정원 산출 대상이
            # 아니다. _eqp_can_process()는 모델을 전제로 하므로 먼저 걸러낸다.
            model = sim._eqp_model_map.get(eqp_id)
            if not model:
                continue
            if not sim._eqp_can_process(eqp_id, ppk, oper_id):
                continue
            st = sim._st_per_wafer_for_eqp(eqp_id, ppk, oper_id)
            if st is None or st <= 0:
                continue
            running = eqp.status == "busy" and (eqp.current_prod, eqp.current_oper) == (ppk, oper_id)
            same_setup = (eqp.prev_prod, eqp.prev_oper) == (ppk, oper_id)
            needs_conv = sim._would_need_conversion(eqp_id, lot_cd, temp)
            key = (
                0 if running else 1,
                0 if same_setup else 1,
                1 if needs_conv else 0,
                float(st),
                sim.model_breadth(eqp_id),
                eqp_id,
            )
            rows.append((key, eqp_id, float(st), model))
        rows.sort(key=lambda r: r[0])
        return [(eqp_id, st, model) for _key, eqp_id, st, model in rows]

    @staticmethod
    def _count_by_model(candidates: List[Tuple[str, float, str]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _eid, _st, model in candidates:
            counts[model] = counts.get(model, 0) + 1
        return counts

    def _st_by_model(
        self, ppk: str, oper_id: str, candidates: List[Tuple[str, float, str]],
    ) -> Dict[str, float]:
        """모델별 ST(장당 분). 후보에 없는 모델도 arrange에 ST가 있으면 포함한다."""
        st_map: Dict[str, float] = {}
        for _eid, st, model in candidates:
            st_map.setdefault(model, round(float(st), 4))
        arrange_map = self._sim._env_data.get("abstract_arrange_map", {})
        for (a_ppk, a_oper, model), st in arrange_map.items():
            if a_ppk == ppk and a_oper == oper_id and st:
                st_map.setdefault(model, round(float(st), 4))
        return st_map

    # ── 재공 집계 ───────────────────────────────────────────────────────────

    def _ready_wip(self, ppk: str, oper_id: str) -> Tuple[int, int]:
        """(ready 재공 매수, ready carrier 수)."""
        sim = self._sim
        wip = sim._wip_for(ppk, oper_id)
        if not wip or wip.get("wip_qty", 0) <= 0:
            return 0, 0
        wafers = 0
        carriers = 0
        for lid in wip.get("lot_ids", []):
            meta = sim._wip_lot_meta.get(lid, {})
            lot = sim.lot_pool.get(lid)
            oper_in_time = meta.get("oper_in_time", wip.get("oper_in_time", 0))
            if not sim._lot_ready(lid, oper_in_time):
                continue
            wafers += int(meta.get("wf_qty", lot.wf_qty if lot else 25))
            carriers += 1
        return wafers, carriers

    def _in_flight_wafers(self, ppk: str, oper_id: str) -> int:
        return sum(
            int(meta.get("wf_qty", 0))
            for meta in self._sim._in_flight.values()
            if meta.get("PLAN_PROD_ATTR_VAL") == ppk and meta.get("oper_id") == oper_id
        )

    # ── 마스킹 판정 ─────────────────────────────────────────────────────────

    def occupancy(self, ppk: str, oper_id: str) -> List[str]:
        """지금 이 버킷의 정원을 실제로 점유 중인 장비 목록.

        - 가공 중(busy)이거나 전환 후 이 버킷 투입이 예약된(pending) 장비
        - 마지막 셋업이 이 버킷이면서 **지금 이 버킷을 실제로 받을 수 있는** idle 장비
          (셋업만 같고 투입 자격이 없는 장비까지 정원을 차지하면, 정작 받을 수 있는
          장비가 밀려 재공이 묶인다 — 그래서 정원 미적용 feasible 집합으로 확인한다)
        """
        sim = self._sim
        occupied: List[str] = []
        for eqp_id in sim._env_data.get("eqp_ids", []):
            eqp = sim.eqps.get(eqp_id)
            if eqp is None or eqp.status == "down":
                continue
            pending = sim._eqp_pending_assign.get(eqp_id)
            if pending is not None:
                if (pending.get("ppk"), pending.get("oper_id")) == (ppk, oper_id):
                    occupied.append(eqp_id)
                continue
            if eqp.status == "busy":
                if (eqp.current_prod, eqp.current_oper) == (ppk, oper_id):
                    occupied.append(eqp_id)
                continue
            if eqp.status != "idle":
                continue
            if (eqp.prev_prod, eqp.prev_oper) != (ppk, oper_id):
                continue
            if (ppk, oper_id) in sim._uncapped_feasible_bucket_keys(eqp_id):
                occupied.append(eqp_id)
        return occupied

    def blocks(self, eqp_id: str, ppk: str, oper_id: str) -> bool:
        """정원 초과로 이 장비의 이 버킷 투입을 막아야 하면 True."""
        if not self.mask_enabled:
            return False
        cap = self.target_for(ppk, oper_id)
        if cap is None or cap.target_cnt <= 0:
            return False
        occupied = self.occupancy(ppk, oper_id)
        if eqp_id in occupied:
            return False          # 이미 정원 안에 있는 장비 — 하던 대로 계속
        return len(occupied) >= cap.target_cnt

    # ── 출력 스냅샷 ─────────────────────────────────────────────────────────

    def snapshot_rows(self) -> List[dict]:
        """(PPK, OPER, MODEL)별 산출 결과 행. RULE_TIMEKEY 시점 스냅샷 저장용."""
        rows: List[dict] = []
        for (ppk, oper_id), cap in sorted(self.targets().items()):
            models = set(cap.st_by_model) | set(cap.req_by_model) | set(cap.target_by_model)
            eqp_model_map = self._sim._eqp_model_map
            for model in sorted(models):
                req = cap.req_by_model.get(model, 0)
                target = cap.target_by_model.get(model, 0)
                if req <= 0 and target <= 0 and not cap.runnable:
                    continue
                plan_eqps = [e for e in cap.target_eqps if eqp_model_map.get(e) == model]
                rows.append({
                    "PLAN_PROD_ATTR_VAL": ppk,
                    "OPER_ID":       oper_id,
                    "EQP_MODEL_CD":  model,
                    "ST":            cap.st_by_model.get(model, 0.0),
                    "HORIZON_MIN":   cap.horizon_min,
                    "PLAN_QTY":      cap.plan_qty,
                    "DONE_QTY":      cap.done_qty,
                    "REMAIN_QTY":    cap.remain_qty,
                    "WIP_QTY":       cap.wip_qty,
                    "WIP_CARRIER_CNT": cap.wip_carriers,
                    "REQ_EQP_CNT":   req,
                    "PLAN_EQP_CNT":  target,
                    "PLAN_EQP_LVAL": ",".join(plan_eqps),
                    "BUCKET_REQ_EQP_CNT":  cap.req_cnt,
                    "BUCKET_PLAN_EQP_CNT": cap.target_cnt,
                    "CAPABLE_EQP_CNT": cap.capable_cnt,
                    "RUNNABLE_YN":   "Y" if cap.runnable else "N",
                    "MASK_APPLY_YN": "Y" if self.mask_enabled else "N",
                    "REASON_CD":     cap.reason,
                })
        return rows


def summarize_allocation(schedule: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """스케줄 결과 → (PPK, OPER, MODEL)별 실제 배치 장비 집계.

    같은 장비가 같은 버킷을 여러 번 돌아도 1대로 센다(대수 비교가 목적).
    """
    alloc: Dict[Tuple[str, str, str], dict] = {}
    for rec in schedule:
        key = (
            rec.get("PLAN_PROD_ATTR_VAL", ""),
            rec.get("OPER_ID", ""),
            rec.get("EQP_MODEL", "") or rec.get("EQP_MODEL_CD", ""),
        )
        entry = alloc.setdefault(key, {"eqp_ids": set(), "run_qty": 0})
        if rec.get("EQP_ID"):
            entry["eqp_ids"].add(rec["EQP_ID"])
        entry["run_qty"] += int(rec.get("WF_QTY", 0) or 0)
    return alloc
