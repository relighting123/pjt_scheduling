"""
tests/test_sametool_setup.py

sametool_setup(BATCHID는 동일·PPK/OPER만 전환)을 실제 TOOL 전환과 구분해
카운트/리워드하는 기능을 검증한다. 설계서:
docs/superpowers/specs/2026-08-14-sametool-setup-dedication-design.md
"""
from pathlib import Path

import pytest

from config import CONFIG, apply_reward_params, reward_params_dict


def test_reward_params_round_trip_includes_w_sametool_setup():
    original = CONFIG.reward.w_sametool_setup
    try:
        d = reward_params_dict()
        assert "w_sametool_setup" in d
        assert d["w_sametool_setup"] == pytest.approx(original)

        apply_reward_params({"w_sametool_setup": 0.75})
        assert CONFIG.reward.w_sametool_setup == pytest.approx(0.75)
    finally:
        CONFIG.reward.w_sametool_setup = original
