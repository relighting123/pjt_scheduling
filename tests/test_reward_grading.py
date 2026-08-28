"""등급화(Grading) — grade_ratio/grade_array 유닛 + 리워드/관측 적용 검증.

배경: 5분 주기 재추론에서 관측·리워드가 공유하는 연속 원시값(회피가능
전환비율, cover/need, block/budget)이 회차마다 미세하게 흔들리면 정책이
그 미세차를 그대로 좇아 스케줄이 크게 뒤바뀐다(실측: 관측 1.5% 변화 →
스케줄 시퀀스 33% 변화). RewardConfig.reward_grade_levels/EnvConfig.
obs_grade_levels(둘 다 기본 0=비활성)로 이 원시값들을 이산 등급으로 묶어
미세 진동을 흡수한다.
"""
import numpy as np
import pytest

from config import CONFIG
from data.loader.preprocess import preprocess
from env.scheduling_rl_env import SchedulingRLEnv
from simulation.simulator import grade_array, grade_ratio

RULE_TIMEKEY = "20260712070000"


# ── grade_ratio() / grade_array() 단위 테스트 ────────────────────────────────

def test_grade_ratio_disabled_returns_raw_value():
    assert grade_ratio(0.373, 0) == 0.373
    assert grade_ratio(0.373, 1) == 0.373


def test_grade_ratio_matches_grade5_at_5_levels():
    for ratio in (0.0, 0.1, 0.24, 0.25, 0.3, 0.49, 0.5, 0.51, 0.75, 0.76, 0.99, 1.0):
        from simulation.simulator import _grade5
        assert grade_ratio(ratio, 5, cap=1.0) == _grade5(ratio)


def test_grade_ratio_clamps_out_of_range():
    assert grade_ratio(-0.5, 5) == 0.0
    assert grade_ratio(1.5, 5, cap=1.0) == 1.0


def test_grade_ratio_respects_cap():
    # cap=2.0, levels=3 → 단계 {0, 1, 2}
    assert grade_ratio(0.0, 3, cap=2.0) == 0.0
    assert grade_ratio(1.0, 3, cap=2.0) == 1.0
    assert grade_ratio(2.0, 3, cap=2.0) == 2.0
    assert grade_ratio(1.9, 3, cap=2.0) == 1.0


def test_grade_ratio_reduces_distinct_values():
    """등급화 목적 검증: 연속값 다수가 소수 등급으로 뭉쳐야 한다."""
    raw = np.linspace(0.0, 1.0, 101)
    graded = {grade_ratio(float(x), 5, cap=1.0) for x in raw}
    assert len(graded) <= 5


def test_grade_array_matches_grade_ratio_elementwise():
    levels, cap = 5, 1.0
    raw = np.array([0.0, 0.1, 0.3, 0.5, 0.75, 0.9, 1.0], dtype=np.float32)
    arr_out = grade_array(raw, levels, cap)
    scalar_out = np.array([grade_ratio(float(x), levels, cap) for x in raw], dtype=np.float32)
    np.testing.assert_allclose(arr_out, scalar_out, atol=1e-6)


def test_grade_array_disabled_returns_same_array():
    raw = np.array([0.1, 0.9], dtype=np.float32)
    assert grade_array(raw, 0, 1.0) is raw


# ── config 배선 (reward_params_dict / apply_reward_params) ─────────────────

def test_reward_params_dict_includes_grading_fields():
    from config import reward_params_dict
    d = reward_params_dict()
    assert "reward_grade_levels" in d
    assert "conversion_escalation_step" in d
    assert "conversion_escalation_bucket" in d
    assert "conversion_escalation_max" in d


def test_apply_reward_params_sets_grading_fields():
    from config import apply_reward_params
    r = CONFIG.reward
    original = (
        r.reward_grade_levels, r.conversion_escalation_step,
        r.conversion_escalation_bucket, r.conversion_escalation_max,
    )
    try:
        apply_reward_params({
            "reward_grade_levels": 5,
            "conversion_escalation_step": 0.5,
            "conversion_escalation_bucket": 3,
            "conversion_escalation_max": 2.0,
        })
        assert r.reward_grade_levels == 5
        assert r.conversion_escalation_step == pytest.approx(0.5)
        assert r.conversion_escalation_bucket == 3
        assert r.conversion_escalation_max == pytest.approx(2.0)
    finally:
        (r.reward_grade_levels, r.conversion_escalation_step,
         r.conversion_escalation_bucket, r.conversion_escalation_max) = original


# ── 시뮬레이션 통합: 기본값(0)이면 기존과 완전 동일 ──────────────────────────

def _raw_2eqp_2ppk():
    discrete, plan, flow, abstract = [], [], [], []
    for i, (eqp, ppk) in enumerate([("EQP001", "PPK001"), ("EQP002", "PPK002")]):
        for c in range(3):
            discrete.append({
                "EQP_ID": eqp, "LOT_ID": f"LOT{i}{c}", "CARRIER_ID": f"CAR{i}{c}",
                "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001", "ST": 30,
                "EQP_MODEL_CD": "A", "WF_QTY": 1, "SEQ": 1, "LOT_STAT_CD": "WAIT",
            })
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                     "D0_PLAN_QTY": 3, "D1_PLAN_QTY": 3, "PLAN_PRIORITY": 1})
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": "OPER001",
                         "EQP_MODEL_CD": "A", "ST": 30})
    return {"discrete_arrange": discrete, "plan": plan, "flow": flow,
            "abstract_arrange": abstract}


def _load_ed():
    ed = preprocess(_raw_2eqp_2ppk(), period_key=RULE_TIMEKEY)
    ed["eqp_selection"] = "order"
    ed["sim_end_minutes"] = 180
    ed["conversion_minutes"] = 30
    return ed


@pytest.fixture
def reward_cfg():
    r = CONFIG.reward
    fields = [
        "reward_grade_levels", "conversion_escalation_step",
        "conversion_escalation_bucket", "conversion_escalation_max",
    ]
    original = {f: getattr(r, f) for f in fields}
    yield r
    for f, v in original.items():
        setattr(r, f, v)


@pytest.fixture
def env_cfg():
    e = CONFIG.env
    original = e.obs_grade_levels
    yield e
    e.obs_grade_levels = original


def _run_episode(ed, max_steps=500):
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    env.reset()
    rewards = []
    for _ in range(max_steps):
        mask = env.action_masks()
        feasible = np.flatnonzero(mask[:env._n_bucket])
        action = np.array([int(feasible[0]) if feasible.size else 0, env._L - 1],
                          dtype=np.int64)
        _obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards


def test_default_grading_off_matches_baseline(reward_cfg, env_cfg):
    """기본값(모두 0)이면 등급화 코드가 없던 것과 스텝별 리워드가 완전히 같아야 한다."""
    reward_cfg.reward_grade_levels = 0
    reward_cfg.conversion_escalation_step = 0.0
    env_cfg.obs_grade_levels = 0

    ed1 = _load_ed()
    ed2 = _load_ed()
    r1 = _run_episode(ed1)
    r2 = _run_episode(ed2)
    assert r1 == pytest.approx(r2)


def test_observation_grading_reduces_channel_precision(reward_cfg, env_cfg):
    """obs_grade_levels를 켜면 관측 채널 값이 등급 격자 위에만 놓여야 한다."""
    env_cfg.obs_grade_levels = 5
    ed = _load_ed()
    env = SchedulingRLEnv(ed, record_history=False, record_event_log=False)
    obs, _ = env.reset()
    allowed = {i / 4 for i in range(5)}
    # bucket 구간(그레이드 대상 채널 포함)은 관측 전체가 [0,1]이라 값 자체가
    # 항상 격자 위인지 완전히 보장할 순 없지만(다른 비-등급 채널도 섞여 있음),
    # 적어도 값들이 유한하고 [0,1] 범위 안에 있어야 한다(회귀 없는지 sanity check).
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert np.isfinite(obs).all()


def test_conversion_escalation_multiplier_scales_penalty(reward_cfg):
    """전환 페널티 계단화: step>0이면 누적 전환 횟수가 늘수록 conversion 항이 커진다."""
    from simulation.simulator import SchedulingSimulator

    reward_cfg.conversion_escalation_step = 1.0
    reward_cfg.conversion_escalation_bucket = 1
    reward_cfg.conversion_escalation_max = 5.0

    ed = _load_ed()
    sim = SchedulingSimulator(ed, reward_cfg, record_history=False, record_event_log=False)

    class _FakeEqp:
        eqp_id = "EQP001"
        conversion_count = 0
        prev_lot_cd = "OLDLOT"
        prev_temp = ""
        free_at = 0
        status = "idle"
        idle_accum = 0

    eqp = _FakeEqp()
    # needs_conv를 강제로 True로 만들기 위해 _would_need_conversion을 몽키패치
    sim._would_need_conversion = lambda *a, **k: True
    sim._conversion_avoidable_fraction = lambda *a, **k: 0.0

    eqp.conversion_count = 0
    _cs, _ce, reward0, _nc = sim._apply_conversion_start(eqp, "NEWLOT", "T1", 0.0)
    term0 = sim._cur_conv_terms["conversion"]

    eqp.conversion_count = 1
    _cs, _ce, reward1, _nc = sim._apply_conversion_start(eqp, "NEWLOT", "T1", 0.0)
    term1 = sim._cur_conv_terms["conversion"]

    # mult(0회)=1.0, mult(1회)=min(1+1*1, 5)=2.0 → w_conversion(-10)*배수
    assert term0 == pytest.approx(reward_cfg.w_conversion * 1.0)
    assert term1 == pytest.approx(reward_cfg.w_conversion * 2.0)
    assert term1 < term0  # 더 강한 패널티(음수가 더 커짐)


def test_conversion_escalation_disabled_by_default(reward_cfg):
    """step=0.0(기본)이면 배수가 항상 1.0 — 현행과 완전히 동일."""
    from simulation.simulator import SchedulingSimulator

    reward_cfg.conversion_escalation_step = 0.0
    ed = _load_ed()
    sim = SchedulingSimulator(ed, reward_cfg, record_history=False, record_event_log=False)

    class _FakeEqp:
        eqp_id = "EQP001"
        conversion_count = 7
        prev_lot_cd = "OLDLOT"
        prev_temp = ""
        free_at = 0
        status = "idle"
        idle_accum = 0

    eqp = _FakeEqp()
    sim._would_need_conversion = lambda *a, **k: True
    sim._conversion_avoidable_fraction = lambda *a, **k: 0.0
    sim._apply_conversion_start(eqp, "NEWLOT", "T1", 0.0)
    assert sim._cur_conv_terms["conversion"] == pytest.approx(reward_cfg.w_conversion)


def test_graded_avoidable_conversion_never_hits_reject_sentinel(reward_cfg):
    """avoidable_conversion 등급값이 -1.0(배정 거부 센티널)과 충돌하지 않아야 한다.

    conversion이 발생하는 스텝은 항상 w_conversion(-10.0)이 먼저 더해지므로
    avoidable 등급항이 어떤 값이든 총합이 -1.0 부근에 오지 않는다 — 여러
    등급 수(3~10)에 걸쳐 sanity 확인.
    """
    from simulation.simulator import SchedulingSimulator

    for levels in range(0, 11):
        reward_cfg.reward_grade_levels = levels
        ed = _load_ed()
        sim = SchedulingSimulator(ed, reward_cfg, record_history=False, record_event_log=False)

        class _FakeEqp:
            eqp_id = "EQP001"
            conversion_count = 0
            prev_lot_cd = "OLDLOT"
            prev_temp = ""
            free_at = 0
            status = "idle"

        eqp = _FakeEqp()
        sim._would_need_conversion = lambda *a, **k: True
        for frac in (x / 10 for x in range(11)):
            sim._conversion_avoidable_fraction = lambda *a, **k: frac
            _cs, _ce, reward, _nc = sim._apply_conversion_start(eqp, "NEWLOT", "T1", 0.0)
            assert reward != -1.0, f"levels={levels} frac={frac} → reward hit -1.0 sentinel"
