"""tool_capacity EQP_MODEL_CD 보정·검증."""
from utils.helpers import (
    backfill_tool_capacity_rows,
    normalize_tool_capacity_rows,
    validate_tool_capacity_records,
)


def _sample_discrete():
    return [
        {
            "EQP_ID": "EQP01",
            "LOT_ID": "LOT_A_1",
            "PLAN_PROD_ATTR_VAL": "PPK1",
            "OPER_ID": "OP1",
            "ST": 10,
            "EQP_MODEL_CD": "MODEL_A",
            "WF_QTY": 25,
        },
        {
            "EQP_ID": "EQP02",
            "LOT_ID": "LOT_B_1",
            "PLAN_PROD_ATTR_VAL": "PPK2",
            "OPER_ID": "OP1",
            "ST": 12,
            "EQP_MODEL_CD": "MODEL_B",
            "WF_QTY": 25,
        },
    ]


def _sample_lot_master():
    return [
        {"LOT_ID": "LOT_A_1", "LOT_CD": "LC_A", "TEMP": "25"},
        {"LOT_ID": "LOT_B_1", "LOT_CD": "LC_B", "TEMP": "25"},
    ]


def test_backfill_tool_capacity_from_discrete():
    raw = [
        {"LOT_CD": "LC_A", "EQP_MODEL": None, "MAX_TOOL": 2},
        {"LOT_CD": "LC_B", "EQP_MODEL_CD": "", "MAX_TOOL": 3},
    ]
    filled, warnings = backfill_tool_capacity_rows(
        raw, _sample_discrete(), _sample_lot_master(),
    )
    assert len(warnings) == 2
    normalized = normalize_tool_capacity_rows(filled)
    assert normalized == [
        {"LOT_CD": "LC_A", "EQP_MODEL_CD": "MODEL_A", "MAX_TOOL": 2},
        {"LOT_CD": "LC_B", "EQP_MODEL_CD": "MODEL_B", "MAX_TOOL": 3},
    ]
    assert validate_tool_capacity_records(filled) == []


def test_validate_tool_capacity_allows_empty_after_drop():
    raw = [{"LOT_CD": "LC_X", "EQP_MODEL_CD": "", "MAX_TOOL": 2}]
    filled, _ = backfill_tool_capacity_rows(raw, _sample_discrete(), _sample_lot_master())
    assert filled == []
    assert validate_tool_capacity_records(raw) == []
