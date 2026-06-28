"""리워드 파라미터 API 연동 테스트."""
from config import CONFIG, apply_reward_params, reward_params_dict


def test_reward_params_dict_has_all_keys():
    params = reward_params_dict()
    assert set(params) == {
        "w_same_setup",
        "w_same_oper", "w_same_prod", "w_prod_switch", "w_idle_per_min",
        "w_completion", "w_plan_hit", "w_pacing", "w_conversion",
        "w_avoidable_conversion", "conversion_amortize_factor",
        "w_late_finish", "w_flow_balance", "reward_clip",
        "flow_balance_starving_cover_min",
        "use_achievable_target", "same_oper_conditional",
    }


def test_apply_reward_params_updates_config():
    original = reward_params_dict()
    try:
        apply_reward_params({
            "w_plan_hit": 4.5,
            "w_pacing": 2.5,
            "w_conversion": -8.0,
            "use_achievable_target": False,
            "same_oper_conditional": False,
        })
        assert CONFIG.reward.w_plan_hit == 4.5
        assert CONFIG.reward.w_pacing == 2.5
        assert CONFIG.reward.w_conversion == -8.0
        assert CONFIG.reward.use_achievable_target is False
        assert CONFIG.reward.same_oper_conditional is False
    finally:
        apply_reward_params(original)
