"""tool_capacity.json 필드명(EQP_MODEL_CD) 검증 테스트."""
from data.loader import validate_data, preprocess
from utils.helpers import (
    normalize_tool_capacity_rows,
    validate_tool_capacity_records,
)


def _minimal_raw(**overrides):
    base = {
        "discrete_arrange": [
            {
                "EQP_ID": "EQP001",
                "LOT_ID": "LOT001",
                "PLAN_PROD_KEY": "PPK001",
                "OPER_ID": "OPER001",
                "ST": 10,
                "EQP_MODEL_CD": "A",
                "WF_QTY": 25,
            },
        ],
        "abstract_arrange": [
            {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 10},
        ],
        "plan": [
            {
                "PLAN_PROD_KEY": "PPK001",
                "OPER_ID": "OPER001",
                "D0_PLAN_QTY": 25,
                "D1_PLAN_QTY": 25,
                "PLAN_PRIORITY": 1,
            },
        ],
        "flow": [
            {"PLAN_PROD_KEY": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"},
        ],
    }
    base.update(overrides)
    return base


def test_tool_capacity_accepts_eqp_model_cd():
    raw = _minimal_raw(
        tool_capacity=[{"LOT_CD": "LC01", "EQP_MODEL_CD": "A", "MAX_TOOL": 2}],
    )
    assert not validate_data(raw), validate_data(raw)
    env = preprocess(raw)
    assert env["tool_capacity"][("LC01", "A")] == 2


def test_tool_capacity_accepts_legacy_eqp_model_key():
    raw = _minimal_raw(
        tool_capacity=[{"LOT_CD": "LC01", "EQP_MODEL": "A", "MAX_TOOL": 2}],
    )
    assert not validate_data(raw), validate_data(raw)
    env = preprocess(raw)
    assert env["tool_capacity"][("LC01", "A")] == 2


def test_normalize_tool_capacity_uppercases_model_cd():
    rows = normalize_tool_capacity_rows(
        [{"LOT_CD": " lc01 ", "EQP_MODEL_CD": " a ", "MAX_TOOL": 3}],
    )
    assert rows == [{"LOT_CD": "lc01", "EQP_MODEL_CD": "A", "MAX_TOOL": 3}]


def test_normalize_tool_capacity_legacy_eqp_model_maps_to_cd():
    rows = normalize_tool_capacity_rows(
        [{"LOT_CD": "LC01", "EQP_MODEL": "B", "MAX_TOOL": 1}],
    )
    assert rows == [{"LOT_CD": "LC01", "EQP_MODEL_CD": "B", "MAX_TOOL": 1}]


def test_validate_tool_capacity_records_empty():
    assert validate_tool_capacity_records([]) == ["tool_capacity: 데이터가 비어 있습니다."]
