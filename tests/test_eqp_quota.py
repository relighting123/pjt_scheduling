"""
tests/test_eqp_quota.py

버킷(=PPK×OPER)별 '필요 장비 대수' 쿼터 검증.

운영 사용자가 손으로 하는 계산
    필요대수 = 계획량 ÷ (IPH × 가동시간),  IPH = 60 / ST(분/매)
과 시뮬레이터 산출이 정확히 일치하는지, 그리고 그 쿼터가 실제로 버킷 결정
(action mask / 리워드)에 반영돼 과점유를 막는지 확인한다.
"""
import math

import numpy as np
import pytest

from config import CONFIG, EqpQuotaConfig
from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


@pytest.fixture(scope="module")
def env_data(tmp_path_factory):
    input_dir = tmp_path_factory.mktemp("input")
    generate_sample_data(scenario="default", output_dir=input_dir)
    raw = load_data(input_dir)
    return preprocess(raw, period_key=RULE_TIMEKEY)


@pytest.fixture
def quota_defaults():
    """테스트가 CONFIG.quota를 건드려도 다른 테스트에 새지 않게 복원."""
    saved = CONFIG.quota
    CONFIG.quota = EqpQuotaConfig()
    yield CONFIG.quota
    CONFIG.quota = saved


def _planned_bucket(sim: SchedulingSimulator):
    """계획량과 ST가 모두 있는 버킷 하나."""
    for (ppk, oper_id), pm in sim._env_data["plan_meta"].items():
        if pm.get("d0_plan_qty", 0) > 0 and sim.bucket_iph(ppk, oper_id):
            return ppk, oper_id
    pytest.skip("계획+ST가 있는 버킷이 없는 데이터셋")


# ── 산출식: 사용자 수기 계산과 동일한가 ────────────────────────────────────

def test_required_matches_manual_formula(env_data, quota_defaults):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)

    info = sim.bucket_required_eqp_breakdown(ppk, oper_id)
    plan_qty = sim._env_data["plan_meta"][(ppk, oper_id)]["d0_plan_qty"]
    st = sim._bucket_st_per_wafer(ppk, oper_id)

    # 사용자가 엑셀에서 하는 계산 그대로
    iph = 60.0 / st
    hours = sim.soft_cutoff / 60.0
    expected = min(
        max(math.ceil(plan_qty / (iph * hours)), CONFIG.quota.min_eqp),
        sim._capable_eqp_count(ppk, oper_id),
    )

    assert info["iph"] == pytest.approx(iph, rel=1e-6)
    assert info["hours"] == pytest.approx(hours, rel=1e-6)
    assert info["required"] == expected
    assert sim.bucket_required_eqp(ppk, oper_id) == expected


def test_static_quota_does_not_drift_over_time(env_data, quota_defaults):
    """하루 고정 쿼터는 시간이 흘러도 값이 바뀌지 않는다(플립플롭 억제의 근거)."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    before = sim.bucket_required_eqp(ppk, oper_id)

    sim.current_time = sim.soft_cutoff // 2
    sim.stats["completed_qty"][(ppk, oper_id)] = 10
    assert sim.bucket_required_eqp(ppk, oper_id) == before


def test_rolling_quota_tracks_remaining(env_data, quota_defaults):
    """static_daily=False면 잔여계획/잔여시간 기준으로 재계산된다."""
    CONFIG.quota.static_daily = False
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)

    at_start = sim.bucket_required_eqp_breakdown(ppk, oper_id)
    sim.stats["completed_qty"][(ppk, oper_id)] = (
        sim._env_data["plan_meta"][(ppk, oper_id)]["d0_plan_qty"]
    )
    after_done = sim.bucket_required_eqp_breakdown(ppk, oper_id)

    assert after_done["basis_qty"] == 0
    assert after_done["required"] <= at_start["required"]


def test_deadline_uses_shorter_of_soft_cutoff_and_sim_end(env_data, quota_defaults):
    """가동시간은 min(soft_cutoff, sim_end) — 벤치마크처럼 sim_end을 짧게
    덮어써 soft_cutoff보다 먼저 끝나는 환경에서 '가동시간'을 실제보다
    길게(=필요대수를 적게) 잡으면 안 된다.

    DedicationAgent._deadline()과 동일 정의라야 한다 — 안 맞으면 이
    쿼터가 dedication 자신의 판단보다 적은 대수로 마스크를 씌워, 계획을
    못 따라가는 버킷이 생기고 그걸 만회하려 전환만 늘어난다(실측: 벤치마크
    LOAD_skew에서 dedication 생산 16/30 → 20/30, 이 값이 quota 없을 때와
    동일하게 맞아떨어짐).
    """
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    assert sim.soft_cutoff > 0

    # soft_cutoff보다 훨씬 짧은 sim_end (벤치마크가 실제로 하는 오버라이드)
    sim.sim_end = max(sim.soft_cutoff // 4, 1)
    sim._quota_cache.clear()

    info = sim.bucket_required_eqp_breakdown(ppk, oper_id)
    assert info["hours"] == pytest.approx(sim.sim_end / 60.0, rel=1e-6)

    plan_qty = sim._env_data["plan_meta"][(ppk, oper_id)]["d0_plan_qty"]
    iph = sim.bucket_iph(ppk, oper_id)
    expected_raw = plan_qty / (iph * (sim.sim_end / 60.0))
    assert info["raw"] == pytest.approx(expected_raw, rel=1e-4)


def test_quota_released_after_plan_met(env_data, quota_defaults):
    """계획을 채운 버킷은 쿼터가 풀린다 — 남은 재공 소진에 장비를 다시 붙일 수 있다."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    assert sim.bucket_required_eqp(ppk, oper_id) is not None

    sim.stats["completed_qty"][(ppk, oper_id)] = (
        sim._env_data["plan_meta"][(ppk, oper_id)]["d0_plan_qty"]
    )
    assert sim.bucket_required_eqp(ppk, oper_id) is None
    assert sim.bucket_over_quota(None, ppk, oper_id) is False
    assert sim.bucket_quota_state(None, ppk, oper_id)["plan_met_released"] is True
    # 대조표에는 '원래 몇 대가 필요했는지'가 남아야 검증이 된다
    assert sim.bucket_required_eqp_breakdown(ppk, oper_id)["required"] is not None

    CONFIG.quota.release_when_plan_met = False
    sim._quota_cache.clear()
    assert sim.bucket_required_eqp(ppk, oper_id) is not None


def test_disabled_quota_is_unlimited(env_data, quota_defaults):
    CONFIG.quota.enabled = False
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    assert sim.bucket_required_eqp(ppk, oper_id) is None
    assert sim.bucket_over_quota(None, ppk, oper_id) is False
    # 쿼터가 없으면 allowed == feasible
    for eqp_id in sim._env_data["eqp_ids"]:
        assert sim.get_allowed_ppk_oper(eqp_id) == sim.get_feasible_ppk_oper(eqp_id)


def test_planless_bucket_is_unlimited_by_default(env_data, quota_defaults):
    """계획이 없는 버킷(재공 소진용)에는 쿼터를 걸지 않는다."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    info = sim.bucket_required_eqp_breakdown("__NO_SUCH_PPK__", "__NO_SUCH_OPER__")
    assert info["required"] is None
    assert info["skip_reason"] == "no_plan"


# ── 점유 계산: 자기 자신 때문에 자기 버킷이 막히면 안 된다 ─────────────────

def test_committed_count_excludes_self(env_data, quota_defaults):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    eqp_ids = [
        e for e in sim._env_data["eqp_ids"] if sim._eqp_can_process(e, ppk, oper_id)
    ]
    if len(eqp_ids) < 2:
        pytest.skip("이 버킷을 처리 가능한 장비가 2대 미만")

    for eid in eqp_ids:
        sim.eqps[eid].prev_prod = ppk
        sim.eqps[eid].prev_oper = oper_id

    mine = eqp_ids[0]
    assert sim.bucket_committed_eqp_count(ppk, oper_id) == len(eqp_ids)
    assert sim.bucket_committed_eqp_count(
        ppk, oper_id, exclude_eqp=mine,
    ) == len(eqp_ids) - 1


def test_staying_on_own_bucket_is_never_blocked(env_data, quota_defaults):
    """이미 그 버킷에 셋업된 장비가 계속 도는 것은 쿼터가 막지 않는다.

    (막으면 매 결정마다 공정을 옮겨다니게 되어 플립플롭이 오히려 심해진다.)
    """
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    required = sim.bucket_required_eqp(ppk, oper_id)
    assert required is not None

    capable = [
        e for e in sim._env_data["eqp_ids"] if sim._eqp_can_process(e, ppk, oper_id)
    ]
    # 필요 대수만큼 이미 붙여 둔다 → 정확히 포화 상태
    committed = capable[:required]
    for eid in committed:
        sim.eqps[eid].prev_prod = ppk
        sim.eqps[eid].prev_oper = oper_id

    # 이미 붙어 있는 장비: 계속 같은 버킷 OK
    assert sim.bucket_over_quota(committed[0], ppk, oper_id) is False
    # 새로 들어오려는 다른 장비: 초과 → 차단
    newcomer = next((e for e in capable if e not in committed), None)
    if newcomer is not None:
        assert sim.bucket_over_quota(newcomer, ppk, oper_id) is True


# ── 마스크 반영 ────────────────────────────────────────────────────────────

def test_mask_drops_over_quota_bucket(env_data, quota_defaults):
    """포화된 버킷은 action mask에서 빠진다 (다른 선택지가 있을 때)."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id = sim.current_idle_eqp()
    feasible = sim.get_feasible_ppk_oper(eqp_id)
    if len(feasible) < 2:
        pytest.skip("선택지가 2개 미만이라 마스크 차이를 볼 수 없음")

    ppk, oper_id = sim.ppk_oper_from_flat(feasible[0])
    required = sim.bucket_required_eqp(ppk, oper_id)
    if required is None:
        pytest.skip("쿼터 없는 버킷")

    # 이 버킷을 다른 장비들로 정확히 포화시킨다
    others = [
        e for e in sim._env_data["eqp_ids"]
        if e != eqp_id and sim._eqp_can_process(e, ppk, oper_id)
    ]
    if len(others) < required:
        pytest.skip("포화시킬 다른 장비가 부족")
    for eid in others[:required]:
        sim.eqps[eid].prev_prod = ppk
        sim.eqps[eid].prev_oper = oper_id
    sim._invalidate_caches()

    allowed = sim.get_allowed_ppk_oper(eqp_id)
    assert feasible[0] not in allowed
    assert set(allowed) < set(feasible)


def test_mask_falls_back_when_all_over_quota(env_data, quota_defaults):
    """선택지가 전부 초과면 원래 feasible로 폴백 — 장비를 놀리지 않는다."""
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    eqp_id = sim.current_idle_eqp()
    feasible = set(sim.get_feasible_ppk_oper(eqp_id))
    assert feasible

    # 모든 버킷을 min_eqp=0 쿼터로 만들어 무조건 초과 상태로
    CONFIG.quota.min_eqp = 0
    CONFIG.quota.headroom = -10.0
    CONFIG.quota.rounding = "floor"
    sim._quota_cache.clear()
    sim._allowed_cache.clear()

    assert set(sim.get_allowed_ppk_oper(eqp_id)) == feasible

    CONFIG.quota.allow_overflow_when_idle = False
    sim._allowed_cache.clear()
    assert sim.get_allowed_ppk_oper(eqp_id) == []


def test_env_mask_matches_allowed_set(env_data, quota_defaults):
    env = SchedulingRLEnv(env_data, record_history=False, record_event_log=False)
    env.reset()
    eqp_id = env.sim.current_idle_eqp()
    if eqp_id is None:
        pytest.skip("reset 직후 결정 EQP 없음")

    mask = env.action_masks()
    bucket_mask = mask[: env._n_bucket]
    allowed = [f for f in env.sim.get_allowed_ppk_oper(eqp_id) if f < env._n_bucket]
    if allowed:
        assert set(np.flatnonzero(bucket_mask)) == set(allowed)


# ── 리워드 ─────────────────────────────────────────────────────────────────

def test_over_quota_penalty_applied(env_data, quota_defaults):
    sim = SchedulingSimulator(env_data, record_history=False, record_event_log=False)
    ppk, oper_id = _planned_bucket(sim)
    required = sim.bucket_required_eqp(ppk, oper_id)
    capable = [
        e for e in sim._env_data["eqp_ids"] if sim._eqp_can_process(e, ppk, oper_id)
    ]
    if required is None or len(capable) <= required:
        pytest.skip("초과 상태를 만들 장비가 부족")

    mine = capable[-1]
    for eid in capable[:required]:
        sim.eqps[eid].prev_prod = ppk
        sim.eqps[eid].prev_oper = oper_id

    penalty, state = sim._eqp_quota_reward(mine, ppk, oper_id)
    assert state["over_quota"] is True
    assert penalty < 0
    assert penalty == pytest.approx(
        sim._reward_cfg.w_eqp_over_quota * min(state["overflow"] / max(required, 1), 1.0)
    )

    # 포화 전이라면 페널티 0
    sim.eqps[capable[0]].prev_prod = None
    sim.eqps[capable[0]].prev_oper = None
    penalty_ok, state_ok = sim._eqp_quota_reward(mine, ppk, oper_id)
    assert state_ok["over_quota"] is False
    assert penalty_ok == 0.0


# ── 대조표 ─────────────────────────────────────────────────────────────────

def test_quota_report_shape(env_data, quota_defaults):
    env = SchedulingRLEnv(env_data, record_history=False, record_event_log=False)
    env.reset()
    for _ in range(50):
        mask = env.action_masks()
        flat = int(np.flatnonzero(mask[: env._n_bucket])[0])
        _o, _r, terminated, truncated, _i = env.step(np.array([flat, 0], dtype=np.int64))
        if terminated or truncated:
            break

    report = env.sim.bucket_quota_report()
    assert report
    for row in report:
        assert {"ppk", "oper_id", "required", "used_eqp", "peak_eqp"} <= set(row)
        assert row["peak_eqp"] <= row["used_eqp"]
        if row["required"] is not None:
            assert row["over_used"] == row["used_eqp"] - row["required"]


def test_peak_concurrent_counts_distinct_eqps():
    peak = SchedulingSimulator._peak_concurrent_eqp
    # 같은 장비의 연속 carrier는 1대
    assert peak([(0, 10, "E1"), (10, 20, "E1")]) == 1
    # 겹치는 두 장비는 2대
    assert peak([(0, 10, "E1"), (5, 15, "E2")]) == 2
    # 끝나고 시작하면 동시 아님
    assert peak([(0, 10, "E1"), (10, 20, "E2")]) == 1
    assert peak([]) == 0
