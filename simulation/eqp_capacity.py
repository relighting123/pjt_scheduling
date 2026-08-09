"""
simulation/eqp_capacity.py – (PPK, OPER)별 적정 장비 대수 산출 + 정원 마스킹

세 단계로 나뉜다.

1. **필요 대수 산출**(`_required_count`) — 마감(soft_cutoff와 sim_end 중 이른 쪽)까지 남은
   시간 안에 잔여량을 처리하려면 장비가 몇 대 필요한가. 분당 처리량(1/ST)을 우선순위 순으로
   쌓아, 필요 처리율(잔여량 / 남은시간)에 도달하는 시점의 장비 수가 필요 대수다.

2. **재공 적정성 사전 체크**(`_wip_sustainable_count`) — 대수만 맞춰도 그 대수를 먹일 재공이
   없으면 의미가 없다. n대를 붙이면 재공은 `재공 매수 / Σ(1/ST)`분 만에 마르므로, 그 소진
   시간이 한 대당 최소 가동시간(`capacity_min_run_minutes`) 이상인 대수와 ready carrier 수
   중 작은 쪽까지만 정원으로 인정한다. 그보다 잘게 쪼개면 전환·셋업 시간이 가공 시간보다 커진다.

3. **마스킹**(`blocks`) — 정원(target)이 정해지면 그 버킷을 지금 돌고 있거나(진행 중) 마지막
   셋업이 같은(진행했던) 장비는 항상 통과시키고, 그 외 신규 장비는 점유 수가 정원
   미만일 때만 통과시킨다. 결과적으로 "가급적 하던 장비 그대로, 적정 대수까지만" 운영된다.

**ST는 `(PPK, OPER, EQP_MODEL)`로 정해진다**(`abstract_arrange_map`). 같은 버킷 안에서도
모델이 다르면 처리량이 다르고 `mean(1/ST) ≠ 1/mean(ST)`이므로, 1·2단계 모두 평균 ST로 뭉치지
않고 `_ranked_candidates()`가 만든 **같은 우선순위 목록**을 장비 한 대씩 누적한다. 목록 원소가
`(EQP_ID, ST, EQP_MODEL_CD)`라 대수가 정해지면 앞에서부터 잘라 모델별 대수로 바로 나뉜다.

정원 계산은 상태(state_version)가 바뀔 때마다 다시 하되 같은 상태에서는 캐시를 쓴다.
재공이 남아 있는 버킷의 정원은 최소 1대로 바닥을 두어(`_apply_wip_check`의 `max(target, 1)`),
마스킹 때문에 재공이 통째로 묶여 시뮬레이션이 멈추는 일이 없게 한다.
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
    req_cnt:        int                     # ST 기준 필요 대수(재공 미고려)
    target_cnt:     int                     # 재공 체크 후 확정 정원
    wip_cap:        int                     # 재공만으로 본 상한(req_cnt와 min 취하기 전)
    capable_cnt:    int                     # 처리 가능(다운 제외) 장비 수
    runnable:       bool                    # 지금 구동 가능한 조건인가
    mandatory_cnt:  int = 0                 # 그 장비에만 투입 가능한 carrier가 있는 장비 수
    reason:         str = REASON_OK
    # 우선순위 순 후보 [(EQP_ID, ST, EQP_MODEL_CD)] — 배분이 그대로 재사용한다
    candidates:     List[Tuple[str, float, str]] = field(default_factory=list)
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
        self._alloc_cache = None
        self._alloc_version: int = -1
        self.alloc_trace: List[dict] = []
        self._discrete_index: Optional[Dict[Tuple[str, str], set]] = None

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

    # ── 라인 밸런스 (최종 out 기준) ─────────────────────────────────────────

    @property
    def line_balance(self) -> bool:
        return bool(self._sim._env_data.get(
            "capacity_line_balance", CONFIG.env.capacity_line_balance,
        ))

    def _line_need_out(self) -> Dict[Tuple[str, str], int]:
        """PPK별 flow를 역방향으로 훑어 각 공정이 마감까지 '내보내야 할 양'을 구한다.

        같은 PPK의 공정들은 직렬이라 최종 공정의 out이 곧 그 제품의 실적이다.
        앞 공정에 장비를 몰아줘 봐야 뒤 공정이 못 받으면 재공만 쌓이고 최종 out은
        늘지 않으므로, **최종 공정의 잔여 계획**을 기준으로 잡고 상류로 거슬러
        올라가며 각 공정이 내보내야 할 양을 정한다.

            최종 공정:  내보낼 양 = 계획 − 실적
            앞 공정:    내보낼 양 = 뒤 공정이 내보낼 양 − 뒤 공정이 이미 들고 있는 재공
                                   (대기 + 가공 중)

        뒤 공정이 이미 재공을 들고 있으면 그만큼 앞 공정은 덜 만들어도 된다(pull).
        각 공정의 목표 out이 정해지면 필요 대수는 `목표 out / 남은시간`이라는 **같은
        처리율**을 내는 대수가 되므로, ST가 공정마다 달라도 결과적으로 공정 간
        capa(매/분)가 맞춰진다 — 대수가 아니라 처리율이 맞는 것이 라인 밸런스다.
        """
        sim = self._sim
        need: Dict[Tuple[str, str], int] = {}
        flow = sim._env_data.get("flow", {})
        plan_meta = sim._env_data.get("plan_meta", {})
        for ppk, steps in flow.items():
            ordered = [s["oper_id"] for s in sorted(steps, key=lambda x: x["seq_id"])]
            if not ordered:
                continue
            downstream_need = None
            for oper_id in reversed(ordered):
                plan_qty = int(plan_meta.get((ppk, oper_id), {}).get("d0_plan_qty", 0))
                done = int(sim.stats["completed_qty"].get((ppk, oper_id), 0))
                own = max(plan_qty - done, 0)
                if downstream_need is None:
                    out = own                       # 최종 공정 — 계획이 곧 목표 out
                else:
                    out = max(downstream_need, 0)
                need[(ppk, oper_id)] = out
                # 이 공정이 이미 들고 있는 재공만큼은 앞 공정이 안 만들어도 된다
                held = self._ready_wip(ppk, oper_id)[0] + self._in_flight_wafers(ppk, oper_id)
                downstream_need = out - held
        return need

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

        # 잔여량 = 마감까지 이 공정이 내보내야 할 양.
        #  - 라인 밸런스: 최종 공정 계획에서 역산한 값(_line_need_out) — PPK 안의 공정들이
        #    같은 처리율을 목표로 하게 되어 공정 간 capa가 맞는다.
        #  - 그 외: 자기 공정 계획 − 실적.
        # 어느 쪽이든 "실제로 만들 수 있는 양"(이 공정 재공 + 가공 중 + 상류 재공)까지만
        # 인정한다 — 재공이 안 흘러온 만큼은 아무리 장비를 붙여도 못 만든다.
        if self.line_balance:
            need_out = self._line_need_out().get((ppk, oper_id))
        else:
            need_out = None
        if need_out is None:
            need_out = max(plan_qty - done_qty, 0) if plan_qty > 0 else wip_qty
        reachable = wip_qty + in_flight_qty + self._upstream_wip(ppk, oper_id)
        remain_qty = min(need_out, reachable) if plan_qty > 0 or need_out > 0 else wip_qty

        candidates = self._ranked_candidates(ppk, oper_id)
        st_by_model = self._st_by_model(ppk, oper_id, candidates)

        # 필요 대수는 처리율만으로는 부족하다 — carrier가 특정 장비에만 투입 가능하게
        # 선언돼 있으면(discrete_arrange), 그 장비 없이는 아무리 대수를 늘려도 그 재공을
        # 못 만진다. 그런 '유일 장비'는 반드시 정원에 들어가야 한다.
        mandatory = self._mandatory_eqps(ppk, oper_id, candidates)
        candidates = self._promote_mandatory(candidates, mandatory)
        req_by_rate, fleet_short = self._required_count(candidates, remain_qty, horizon)
        req_cnt = max(req_by_rate, len(mandatory))
        wip_cap = self._wip_cap(candidates, wip_qty, wip_carriers)
        target_cnt, reason = self._apply_wip_check(
            req_cnt, remain_qty, wip_cap, wip_carriers, len(candidates),
            fleet_short=fleet_short, mandatory_cnt=len(mandatory),
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
            req_cnt=req_cnt,
            target_cnt=target_cnt,
            wip_cap=wip_cap,
            mandatory_cnt=len(mandatory),
            capable_cnt=len(candidates),
            runnable=target_cnt > 0 and wip_carriers > 0,
            reason=reason,
            candidates=candidates,
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

    def _wip_sustainable_count(
        self, candidates: List[Tuple[str, float, str]], wip_qty: int,
    ) -> int:
        """재공이 최소 가동시간 이상 먹일 수 있는 최대 대수.

        n대를 병렬로 붙이면 재공은 `재공 매수 / Σ(1/ST)`분 만에 마르므로, 그 시간이
        한 대당 최소 가동시간(capacity_min_run_minutes) 이상인 최대 n을 찾는다.
        ST가 모델마다 다르면 어느 장비를 붙이느냐로 소진 시간이 달라지므로
        평균 ST로 뭉치지 않고 _required_count()와 같은 우선순위 목록을 그대로
        한 대씩 누적한다(같은 목록 = 실제로 붙게 될 장비 순서).
        """
        if wip_qty <= 0 or not candidates:
            return 0
        min_run = self._min_run_minutes
        rate = 0.0
        count = 0
        for _eid, st, _model in candidates:
            rate += 1.0 / st
            if wip_qty / rate < min_run:
                break
            count += 1
        return count

    def _discrete_eqp_index(self) -> Dict[Tuple[str, str], set]:
        """(LOT_ID, OPER_ID) → 그 carrier가 투입 가능하다고 선언된 EQP 집합.

        proc_time_matrix(discrete_arrange)는 시뮬 내내 불변이라 한 번만 만든다.
        """
        if self._discrete_index is None:
            index: Dict[Tuple[str, str], set] = {}
            for (lot_id, eqp_id, oper_id) in self._sim._env_data.get("proc_time_matrix", {}):
                index.setdefault((lot_id, oper_id), set()).add(eqp_id)
            self._discrete_index = index
        return self._discrete_index

    def _mandatory_eqps(
        self, ppk: str, oper_id: str, candidates: List[Tuple[str, float, str]],
    ) -> List[str]:
        """이 버킷의 남은 재공 중 '그 장비에만' 투입 가능한 carrier가 있는 장비들.

        carrier는 쪼갤 수 없고 discrete_arrange가 투입 가능 장비를 좁게 선언할 수 있어서,
        처리율만으로 대수를 정하면 그 carrier를 만질 수 있는 유일한 장비가 정원 밖으로
        밀려 재공이 통째로 남는다. 그런 장비는 필요 대수의 하한이 되어야 한다.

        선언이 없는(순수 abstract) 재공은 모델만 맞으면 아무 장비나 가능하므로 해당 없음.
        """
        sim = self._sim
        wip = sim._wip_for(ppk, oper_id)
        if not wip or wip.get("wip_qty", 0) <= 0:
            return []
        index = self._discrete_eqp_index()
        eligible = {eid for eid, _st, _m in candidates}
        mandatory: List[str] = []
        seen = set()
        for lid in wip.get("lot_ids", []):
            if not sim._is_current_wip_lot(lid, ppk, oper_id):
                continue
            meta = sim._wip_lot_meta.get(lid, {})
            if not sim._lot_ready(lid, meta.get("oper_in_time", wip.get("oper_in_time", 0))):
                continue
            declared = index.get((lid, oper_id))
            if not declared:
                continue                       # 선언 없음 = abstract, 아무 장비나 가능
            usable = declared & eligible
            if len(usable) == 1:
                only = next(iter(usable))
                if only not in seen:
                    seen.add(only)
                    mandatory.append(only)
        return mandatory

    @staticmethod
    def _promote_mandatory(
        candidates: List[Tuple[str, float, str]], mandatory: List[str],
    ) -> List[Tuple[str, float, str]]:
        """유일 장비를 후보 목록 앞으로 — 정원/배분이 반드시 이들을 포함하게 한다."""
        if not mandatory:
            return candidates
        order = {eid: i for i, eid in enumerate(mandatory)}
        head = sorted(
            (c for c in candidates if c[0] in order), key=lambda c: order[c[0]],
        )
        tail = [c for c in candidates if c[0] not in order]
        return head + tail

    def _wip_cap(
        self, candidates: List[Tuple[str, float, str]], wip_qty: int, wip_carriers: int,
    ) -> int:
        """재공만으로 본 상한. 한 대는 최소 1 carrier를 받아야 하고(carrier 수 상한),
        최소 가동시간도 채워야 한다(작업량 상한) — 둘 중 작은 쪽.

        필요 대수와 min을 취하기 전의 값이라, 배분(allocate 모드)에서 "이 버킷에
        최대 몇 대까지 줘도 되는가"의 한도로 쓸 수 있다.
        """
        if wip_carriers <= 0:
            return 0
        by_workload = self._wip_sustainable_count(candidates, wip_qty)
        return min(wip_carriers, by_workload) if by_workload > 0 else min(wip_carriers, 1)

    def _apply_wip_check(
        self,
        req_cnt: int,
        remain_qty: int,
        wip_cap: int,
        wip_carriers: int,
        capable_cnt: int,
        *,
        fleet_short: bool = False,
        mandatory_cnt: int = 0,
    ) -> Tuple[int, str]:
        """재공이 그 대수를 먹일 수 있는지 사전 체크 → 구동 가능한 정원으로 축소."""
        if capable_cnt <= 0:
            return 0, REASON_NO_EQP
        if wip_carriers <= 0:
            return 0, REASON_NO_WIP

        target = min(req_cnt, wip_cap, capable_cnt)
        # 유일 장비(그 장비에만 투입 가능한 carrier가 있는 장비)는 재공 상한보다 우선한다
        # — 최소 가동시간이 안 나온다는 이유로 빼면 그 carrier를 만질 방법이 아예 없다.
        target = max(target, min(mandatory_cnt, capable_cnt))
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
        arrange_map = sim._env_data.get("abstract_arrange_map", {})
        rows: List[Tuple[tuple, str, float, str]] = []
        for eqp_id in sim._env_data.get("eqp_ids", []):
            eqp = sim.eqps.get(eqp_id)
            if eqp is None or eqp.status == "down":
                continue
            # 모델이 없는 EQP(eqp_queue_init에만 등장 — EQP_MODEL_CD는
            # discrete_arrange에서만 나온다)는 ST를 알 수 없어 정원 산출 대상이 아니다.
            model = sim._eqp_model_map.get(eqp_id)
            if not model:
                continue
            # 이 버킷을 처리 가능한지는 (PPK, OPER, MODEL) arrange 행의 존재로 본다.
            # _eqp_can_process()는 OPER 단위인 eqp_oper_cap(discrete)도 OR로 받아주는데,
            # 그건 "이 장비가 이 공정을 한다"는 뜻일 뿐 "이 제품을 이 모델로 한다"는
            # 보장이 아니다. 실제 배정 경로(_abstract_assignable_on_eqp)는 템플릿을
            # eqp_model로 먼저 거르므로, 그 기준을 그대로 맞추지 않으면 실제로는 한 대도
            # 못 돌리는 장비를 정원에 세고 ST도 모델 평균 폴백으로 잘못 잡게 된다.
            if (ppk, oper_id, model) not in arrange_map:
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

    def _upstream_wip(self, ppk: str, oper_id: str) -> int:
        """같은 PPK의 선행 공정들이 들고 있는 재공(대기 + 가공 중) 합.

        이 공정으로 흘러들어올 수 있는 양이라, "마감까지 실제로 만들 수 있는 상한"에
        포함된다. flow_prev를 따라 올라가며 순환은 seen으로 끊는다.
        """
        sim = self._sim
        total = 0
        seen = {oper_id}
        prev = sim._env_data.get("flow_prev", {}).get(ppk, {}).get(oper_id)
        while prev and prev not in seen:
            seen.add(prev)
            total += self._ready_wip(ppk, prev)[0] + self._in_flight_wafers(ppk, prev)
            prev = sim._env_data.get("flow_prev", {}).get(ppk, {}).get(prev)
        return total

    def _in_flight_wafers(self, ppk: str, oper_id: str) -> int:
        return sum(
            int(meta.get("wf_qty", 0))
            for meta in self._sim._in_flight.values()
            if meta.get("PLAN_PROD_ATTR_VAL") == ppk and meta.get("oper_id") == oper_id
        )

    # ── 배분 (allocate 모드) ────────────────────────────────────────────────

    @property
    def alloc_mode(self) -> str:
        """'cap'(버킷별 상한) | 'allocate'(보유 장비를 버킷에 나눠줌)."""
        mode = str(self._sim._env_data.get(
            "capacity_alloc_mode", CONFIG.env.capacity_alloc_mode,
        )).lower()
        return mode if mode in ("cap", "allocate") else "cap"

    def allocation(self) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], float]]:
        """보유 장비를 버킷에 배분한 표와 배분 후 부족률. state_version 동안 캐시.

        긴급도(부족률) water-filling — 한 대 줄 때마다 부족률을 다시 계산해
        가장 아쉬운 버킷에 다음 한 대를 준다. 부족률(비율)을 쓰는 이유는 절대
        부족분을 쓰면 잔여량이 큰 버킷 하나가 전부 독식하기 때문이다.
        """
        sim = self._sim
        if self._alloc_cache is not None and self._alloc_version == sim._state_version:
            return self._alloc_cache
        result = self._compute_allocation()
        self._alloc_cache = result
        self._alloc_version = sim._state_version
        return result

    def _compute_allocation(self):
        sim = self._sim
        caps = self.targets()
        horizon = max(min(sim.soft_cutoff, sim.sim_end) - sim.current_time, 0)
        alloc: Dict[Tuple[str, str], List[str]] = {b: [] for b in caps}
        # 배분 근거 추적 — "왜 이 장비가 이 버킷에 갔나"를 사후에 설명하기 위한 기록.
        # 마지막 배분 계산의 부여 순서만 남긴다(가벼운 리스트).
        trace: List[dict] = []
        self.alloc_trace = trace
        free = {
            eid for eid in sim._env_data.get("eqp_ids", [])
            if sim.eqps.get(eid) is not None and sim.eqps[eid].status != "down"
        }
        st_of = {
            (b, eid): st for b, c in caps.items() for eid, st, _m in c.candidates
        }

        def shortfall_ratio(b):
            remain = caps[b].remain_qty
            if remain <= 0:
                return 0.0
            # horizon이 0이면(마감 경과) 커버가 0이라 부족률은 항상 1 — 잔여량 기준
            # 균등 분배로 수렴한다.
            covered = sum(horizon / st_of[(b, e)] for e in alloc[b])
            return max(remain - covered, 0) / remain

        def critical_for_other(e, b):
            """e가 '아직 한 대도 못 받은 다른 버킷'의 유일하게 남은 후보인가."""
            for b2, c2 in caps.items():
                if b2 == b or alloc[b2] or c2.remain_qty <= 0 or c2.wip_cap <= 0:
                    continue
                rest = [x for x, _s, _m in c2.candidates if x in free]
                if rest == [e]:
                    return True
            return False

        def next_eqp(b):
            avail = [e for e, _s, _m in caps[b].candidates if e in free]
            if not avail:
                return None
            # 남 몫의 유일한 장비는 건너뛴다 — 범용 버킷이 전용 버킷의 유일한 장비를
            # 가져가 그 제품이 통째로 굶는 것을 막는다.
            for e in avail:
                if not critical_for_other(e, b):
                    return e
            return avail[0]

        def fill(require_shortfall: bool) -> None:
            while free:
                best, best_key, best_eqp = None, None, None
                for b, c in caps.items():
                    # 유일 장비는 재공 상한과 무관하게 반드시 배분돼야 한다
                    # (그 carrier를 만질 다른 장비가 없다).
                    limit = max(c.wip_cap, min(c.mandatory_cnt, c.capable_cnt))
                    if len(alloc[b]) >= limit:
                        continue
                    sr = shortfall_ratio(b)
                    if require_shortfall and sr <= 0:
                        continue
                    e = next_eqp(b)
                    if e is None:
                        continue
                    lot_cd, temp = sim._bucket_lot_cd_temp(*b)
                    key = (
                        self._plan_priority(*b),      # ① 계획 우선순위(작을수록 먼저)
                        -sr,                          # ② 부족률 큰 순
                        c.capable_cnt,                # ③ 전용(처리 가능 장비 적은) 버킷 먼저
                        int(sim._would_need_conversion(e, lot_cd, temp)),  # ④ 전환 불필요 먼저
                        -c.remain_qty,                # ⑤ 잔여량 큰 순(첫 바퀴의 실질 기준)
                        b,                            # ⑥ 결정성
                    )
                    if best_key is None or key < best_key:
                        best, best_key, best_eqp = b, key, e
                if best is None:
                    return
                alloc[best].append(best_eqp)
                free.discard(best_eqp)
                trace.append({
                    "round": "부족 해소" if require_shortfall else "유휴 방지",
                    "bucket": best,
                    "eqp": best_eqp,
                    "model": sim._eqp_model_map.get(best_eqp, ""),
                    "shortfall_before": round(float(-best_key[1]), 4),
                    "shortfall_after": round(shortfall_ratio(best), 4),
                })

        # 1차 — 계획 부족을 메운다(부족률 > 0인 버킷만)
        fill(require_shortfall=True)
        # 2차 — 그러고도 남는 장비는 재공이 남은 버킷에 더 준다. 정원의 목적은
        # "한 버킷에 과도하게 몰리는 것"을 막는 것이지 장비를 놀리는 것이 아니다.
        # (계획 초과분은 다음 회차 재공이 되므로 버리는 것보다 낫다.)
        if self._alloc_fill_idle:
            fill(require_shortfall=False)

        shortfalls = {b: round(shortfall_ratio(b), 4) for b in caps}
        return alloc, shortfalls

    @property
    def _alloc_fill_idle(self) -> bool:
        return bool(self._sim._env_data.get(
            "capacity_alloc_fill_idle", CONFIG.env.capacity_alloc_fill_idle,
        ))

    def _plan_priority(self, ppk: str, oper_id: str) -> int:
        """PLAN_PRIORITY (작을수록 우선, 미지정 99) — MinProgressAgent와 같은 규약."""
        return int(
            self._sim._env_data.get("plan_meta", {})
            .get((ppk, oper_id), {}).get("priority", 99)
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
        """정원(또는 배분)을 넘어 이 장비의 이 버킷 투입을 막아야 하면 True."""
        if not self.mask_enabled:
            return False
        cap = self.target_for(ppk, oper_id)
        if cap is None:
            return False
        if self.alloc_mode == "allocate":
            return self._alloc_blocks(eqp_id, (ppk, oper_id))
        if cap.target_cnt <= 0:
            return False
        occupied = self.occupancy(ppk, oper_id)
        if eqp_id in occupied:
            return False          # 이미 정원 안에 있는 장비 — 하던 대로 계속
        return len(occupied) >= cap.target_cnt

    def _alloc_blocks(self, eqp_id: str, bucket: Tuple[str, str]) -> bool:
        """배분표가 곧 마스크 — 배분받은 장비만 그 버킷에 들어간다."""
        alloc, _short = self.allocation()
        assigned = alloc.get(bucket, [])
        if not assigned:
            # 이번 회차 배분에서 한 대도 못 받은 버킷. 장비가 풀리면 재배분이
            # 가져가므로(배분은 state_version마다 재계산) 지금은 열지 않는다.
            return True
        if eqp_id in assigned:
            return False
        # 안전 밸브: 배분받은 장비가 지금 아무도 이 버킷을 실제로 받을 수 없는 상태면
        # (전부 다른 버킷에서 가공 중이거나 투입 자격 미달) 재공이 묶이므로 임시 허용한다.
        return any(self._can_serve_now(a, bucket) for a in assigned)

    def _can_serve_now(self, eqp_id: str, bucket: Tuple[str, str]) -> bool:
        """이 장비가 지금 이 버킷을 실제로 돌리고 있거나 받을 수 있는가."""
        sim = self._sim
        eqp = sim.eqps.get(eqp_id)
        if eqp is None or eqp.status == "down":
            return False
        pending = sim._eqp_pending_assign.get(eqp_id)
        if pending is not None:
            return (pending.get("ppk"), pending.get("oper_id")) == bucket
        if eqp.status == "busy":
            return (eqp.current_prod, eqp.current_oper) == bucket
        if eqp.status != "idle":
            return False
        return bucket in sim._uncapped_feasible_bucket_keys(eqp_id)

    # ── 출력 스냅샷 ─────────────────────────────────────────────────────────

    def snapshot_rows(self) -> List[dict]:
        """(PPK, OPER, MODEL)별 산출 결과 행. RULE_TIMEKEY 시점 스냅샷 저장용.

        allocate 모드에서는 PLAN_EQP_CNT/LVAL이 '정원'이 아니라 '배분 결과'가 된다
        (그 모드에서는 배분표가 곧 이번 회차에 그 버킷에 주기로 한 대수다).
        어느 쪽인지는 ALLOC_MODE 컬럼으로 구분한다.
        """
        allocating = self.alloc_mode == "allocate"
        alloc_map, shortfalls = self.allocation() if allocating else ({}, {})
        rows: List[dict] = []
        for (ppk, oper_id), cap in sorted(self.targets().items()):
            models = set(cap.st_by_model) | set(cap.req_by_model) | set(cap.target_by_model)
            eqp_model_map = self._sim._eqp_model_map
            if allocating:
                assigned = alloc_map.get((ppk, oper_id), [])
                plan_by_model = self._count_by_model(
                    [(e, 0.0, eqp_model_map.get(e, "")) for e in assigned]
                )
                plan_eqps_all = list(assigned)
                bucket_plan_cnt = len(assigned)
                shortfall = shortfalls.get((ppk, oper_id), 0.0)
                models |= set(plan_by_model)
            else:
                plan_by_model = cap.target_by_model
                plan_eqps_all = cap.target_eqps
                bucket_plan_cnt = cap.target_cnt
                shortfall = 0.0
            for model in sorted(models):
                req = cap.req_by_model.get(model, 0)
                target = plan_by_model.get(model, 0)
                if req <= 0 and target <= 0 and not cap.runnable:
                    continue
                plan_eqps = [e for e in plan_eqps_all if eqp_model_map.get(e) == model]
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
                    "BUCKET_PLAN_EQP_CNT": bucket_plan_cnt,
                    "CAPABLE_EQP_CNT": cap.capable_cnt,
                    "RUNNABLE_YN":   "Y" if cap.runnable else "N",
                    "MASK_APPLY_YN": "Y" if self.mask_enabled else "N",
                    "ALLOC_MODE":    "ALLOC" if allocating else "CAP",
                    "SHORTFALL_RATIO": round(float(shortfall), 4),
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
