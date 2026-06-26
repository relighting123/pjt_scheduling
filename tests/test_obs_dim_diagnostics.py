"""obs_dim 진단 메시지 단위 테스트."""
import numpy as np
import pytest

from config import CONFIG
from env.scheduling_env import (
    compute_obs_dim,
    obs_dim_components,
    format_obs_dim_mismatch,
    validate_obs_shape,
    _opk_product_from_obs_dim,
    _factor_opk_triples,
    _OBS_GLOBAL_DIM,
    _OBS_EQP_LOCAL_DIM,
    _OBS_CONTEXT_DIM,
)
from simulation.simulator import SchedulingSimulator


def test_obs_dim_components_formula():
    comp = obs_dim_components()
    F = SchedulingSimulator.BUCKET_FEATURES
    assert comp["total"] == (
        _OBS_GLOBAL_DIM
        + comp["O"] * comp["P"] * comp["K"] * F
        + _OBS_EQP_LOCAL_DIM
        + _OBS_CONTEXT_DIM
    )
    assert comp["total"] == compute_obs_dim()


def test_opk_reverse_from_default_dim():
    dim = compute_obs_dim()
    opk = _opk_product_from_obs_dim(dim)
    comp = obs_dim_components()
    assert opk == comp["O"] * comp["P"] * comp["K"]


def test_factor_opk_triples_same_product_as_default():
    comp = obs_dim_components()
    opk = comp["O"] * comp["P"] * comp["K"]
    triples = _factor_opk_triples(opk)
    assert triples
    for o, p, k in triples:
        assert o * p * k == opk


def test_format_obs_dim_mismatch_contains_config_and_formula():
    expected = compute_obs_dim()
    actual = expected - 750  # 다른 O×P×K 조합을 시뮬레이션
    env_data = {
        "oper_ids": ["OP01", "OP02"],
        "prod_keys": ["PPK_A", "PPK_B"],
        "eqp_models": ["M1", "M2"],
        "eqp_ids": ["EQP001"],
    }
    msg = format_obs_dim_mismatch(
        expected,
        actual,
        env_data=env_data,
        source="테스트",
        model_files=["old_model.zip (obs_dim=1516)"],
    )
    assert "관측 차원(obs_dim) 불일치" in msg
    assert "max_oper_count" in msg
    assert "max_prod_count" in msg
    assert "max_model_count" in msg
    assert "OP01" in msg
    assert "PPK_A" in msg
    assert "old_model.zip" in msg
    assert "O×P×K×15" in msg


def test_validate_obs_shape_raises_detailed_error():
    dim = compute_obs_dim()
    obs = np.zeros(dim - 1, dtype=np.float32)
    with pytest.raises(ValueError) as exc:
        validate_obs_shape(obs, expected_dim=dim, source="단위테스트")
    assert "관측 차원(obs_dim) 불일치" in str(exc.value)
    assert "config" in str(exc.value)


def test_validate_obs_shape_passes_matching_obs():
    dim = compute_obs_dim()
    obs = np.zeros(dim, dtype=np.float32)
    validate_obs_shape(obs, expected_dim=dim)
