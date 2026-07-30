"""
agent/experts.py – 모방학습(BC) 시연용 전문가 정책 모음 + 시나리오별 최적 전문가 선택

왜 필요한가
-----------
BC 시연을 `DedicationAgent` 하나로만 뽑으면, 정책의 출발점은 아무리 잘해야
"전담 휴리스틱"이다. 그런데 벤치마크를 보면 전담이 항상 최선은 아니다 —
처리시간이 이질적인 시나리오(LOAD_stmix)에서는 Earliest-ST 계열이
전담보다 점수가 2배 높았다(12 vs 6). 이런 시나리오에서 전담을 모방시키면
RL은 나쁜 출발점에서 시작해 그 격차를 스스로 메워야 한다.

그래서 **시나리오마다 후보 전문가들을 실제로 굴려보고 KPI가 가장 좋은
전문가를 그 시나리오의 시연자로 삼는다**. 라벨은 시나리오 안에서 일관되고
(한 시나리오엔 전문가 하나), 시나리오마다 다른 전문가를 쓰므로 정책은
"이런 상황에선 전담, 저런 상황에선 빠른 ST 우선"을 관측으로부터 배운다.

여기 있는 전문가는 모두 `SchedulingRLEnv`의 버킷 인덱스를 직접 내놓는다
(`predict(sim) -> np.ndarray([flat])`, HOLD 없음) — RL env에는 진짜 보류가
없기 때문이다.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agent.dedication_agent import DedicationAgent, HOLD_ACTION
from agent.minprogress_agent import MinProgressAgent

# size_level 라벨 정책:
#   "max"  = 그 시점 마스크에서 허용된 가장 큰 레벨 (전담: 긴 블록 선호)
#   "min"  = 항상 0 (매 carrier 재결정: 빠른 ST 추종형)
SizeLevelPolicy = str


class ExpertPolicy:
    """BC 시연자 공통 인터페이스."""

    name: str = "expert"
    size_level_policy: SizeLevelPolicy = "max"

    def bucket(self, sim, eqp_id: Optional[str], mask: np.ndarray, n_bucket: int) -> int:
        raise NotImplementedError

    @staticmethod
    def _fallback(mask: np.ndarray, n_bucket: int) -> int:
        feasible = np.flatnonzero(mask[:n_bucket])
        return int(feasible[0]) if feasible.size else 0

    def resolve(self, sim, eqp_id: Optional[str], mask: np.ndarray, n_bucket: int) -> int:
        """마스크 안의 유효한 버킷을 보장해서 반환."""
        if eqp_id is None:
            return self._fallback(mask, n_bucket)
        try:
            choice = int(self.bucket(sim, eqp_id, mask, n_bucket))
        except Exception:
            return self._fallback(mask, n_bucket)
        if not (0 <= choice < n_bucket) or not mask[choice]:
            return self._fallback(mask, n_bucket)
        return choice

    def size_level(self, mask: np.ndarray, n_bucket: int) -> int:
        allowed = np.flatnonzero(mask[n_bucket:])
        if not allowed.size:
            return 0
        return int(allowed[-1]) if self.size_level_policy == "max" else int(allowed[0])


class DedicationExpert(ExpertPolicy):
    """전담 배분 — 장비별로 한 버킷을 붙잡고 유지 (긴 블록)."""

    name = "dedication"
    size_level_policy = "max"

    def __init__(self, env_data: dict):
        self._agent = DedicationAgent(env_data)

    def bucket(self, sim, eqp_id, mask, n_bucket):
        choice = int(self._agent.predict(sim)[0])
        if choice == HOLD_ACTION:
            # RL env엔 진짜 보류가 없다 — 직전 커밋 버킷으로 대체.
            committed = self._agent._committed.get(eqp_id)
            if committed is not None:
                return committed
            return self._fallback(mask, n_bucket)
        return choice


class MinProgressExpert(ExpertPolicy):
    """계획 진행률이 가장 뒤처진 버킷 우선."""

    name = "minprogress"
    size_level_policy = "max"

    def __init__(self, env_data: dict):
        self._agent = MinProgressAgent(env_data)

    def bucket(self, sim, eqp_id, mask, n_bucket):
        return int(self._agent.predict(sim)[0])


class EarliestSTExpert(ExpertPolicy):
    """예상 종료 시각이 가장 이른 carrier의 버킷 (Earliest-ST의 버킷 버전).

    `EarliestSTAgent`는 simulator의 min_st 경로(`_earliest_st_pick` →
    `assign_earliest_st_pending`)에 의존해서 EQP×LOT을 직접 고르므로,
    `assign_ppk_oper`만 쓰는 RL env에서는 그대로 쓸 수 없다. 여기서는 같은
    점수 함수(`earliest_st_combo_score`)를 **현재 결정 EQP에 한정해서** 적용해
    버킷을 고른다 — 처리시간이 이질적인 시나리오에서 전담보다 강한 시연자다.
    """

    name = "earliest_st"
    size_level_policy = "min"

    def __init__(self, env_data: dict):
        self._env_data = env_data

    def bucket(self, sim, eqp_id, mask, n_bucket):
        feasible = set(sim.get_feasible_ppk_oper(eqp_id))
        if not feasible:
            return self._fallback(mask, n_bucket)

        best_flat: Optional[int] = None
        best_score: Optional[Tuple] = None
        for lot in sim.available_lots(eqp_id):
            if not sim._lot_conv_discrete_eligible(eqp_id, lot):
                continue
            flat = sim.ppk_oper_flat_index(lot["oper_id"], lot["PLAN_PROD_ATTR_VAL"])
            if flat not in feasible:
                continue
            score = sim.earliest_st_combo_score(eqp_id, lot)
            if best_score is None or score < best_score:
                best_score, best_flat = score, flat

        if best_flat is None:
            return self._fallback(mask, n_bucket)
        return best_flat


EXPERT_FACTORIES: Dict[str, Callable[[dict], ExpertPolicy]] = {
    "dedication": DedicationExpert,
    "minprogress": MinProgressExpert,
    "earliest_st": EarliestSTExpert,
}

DEFAULT_EXPERT_CANDIDATES: List[str] = ["dedication", "earliest_st"]


def make_expert(name: str, env_data: dict) -> ExpertPolicy:
    factory = EXPERT_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"알 수 없는 전문가: {name}. 사용 가능: {', '.join(sorted(EXPERT_FACTORIES))}"
        )
    return factory(env_data)


def run_expert_episode(env, expert: ExpertPolicy, max_steps: int) -> dict:
    """전문가로 1 에피소드를 굴려 KPI를 반환 (시연자 선택용).

    env는 이미 생성된 SchedulingRLEnv 계열 인스턴스여야 한다.
    """
    env.reset()
    n_bucket = env._n_bucket
    steps = 0
    for _ in range(max_steps):
        mask = env.action_masks()
        eqp_id = env.sim.current_idle_eqp()
        flat = expert.resolve(env.sim, eqp_id, mask, n_bucket)
        level = expert.size_level(mask, n_bucket)
        _obs, _r, terminated, truncated, _info = env.step(
            np.array([flat, level], dtype=np.int64)
        )
        steps += 1
        if terminated or truncated:
            break

    sim = env.sim
    produced = sum(1 for r in sim.schedule if r.get("END_TM", 0) <= sim.sim_end)
    conversions = int(sim.stats.get("conversions", 0))
    return {
        "produced": produced,
        "conversions": conversions,
        "score": float(produced - conversions),
        "steps": steps,
    }


def select_best_expert(
    data: dict,
    env_factory: Callable[[dict], object],
    candidates: Optional[List[str]] = None,
    max_steps: int = 4000,
) -> Tuple[str, dict]:
    """시나리오 하나에 대해 후보 전문가를 모두 굴려 KPI 최고를 고른다.

    Returns: (전문가 이름, 후보별 KPI dict)
    """
    names = list(candidates or DEFAULT_EXPERT_CANDIDATES)
    scores: Dict[str, dict] = {}
    for name in names:
        env = env_factory(data)
        try:
            scores[name] = run_expert_episode(env, make_expert(name, data), max_steps)
        except Exception as exc:                # 한 전문가가 실패해도 나머지로 진행
            scores[name] = {"produced": 0, "conversions": 10 ** 6,
                            "score": -float("inf"), "error": str(exc)}
    # 동점이면 후보 순서(= DEFAULT_EXPERT_CANDIDATES 우선순위)를 따른다.
    best = max(names, key=lambda n: (scores[n]["score"], -names.index(n)))
    return best, scores
