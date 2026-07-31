"""dataset tool_capacity.json 직접 수리."""
import json

import pytest

from data.dataset_fix import repair_tool_capacity_folder, repair_tool_capacity_rows


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
    ]


def _sample_lot_master():
    return [{"LOT_ID": "LOT_A_1", "LOT_CD": "LC_A", "TEMP": "25"}]


def test_repair_tool_capacity_rows_fills_missing_model():
    raw = [{"LOT_CD": "LC_A", "EQP_MODEL": None, "MAX_TOOL": 2}]
    repaired, notes = repair_tool_capacity_rows(raw, _sample_discrete(), _sample_lot_master())
    assert repaired == [{"LOT_CD": "LC_A", "EQP_MODEL_CD": "MODEL_A", "MAX_TOOL": 2}]
    assert notes


def test_repair_tool_capacity_folder_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("data.dataset_fix.DATASET_DIR", tmp_path)
    folder = "FAC001/train/20260101070000"
    input_dir = tmp_path / folder / "input"
    input_dir.mkdir(parents=True)

    (input_dir / "discrete_arrange.json").write_text(
        json.dumps(_sample_discrete(), ensure_ascii=False), encoding="utf-8",
    )
    (input_dir / "lot_master.json").write_text(
        json.dumps(_sample_lot_master(), ensure_ascii=False), encoding="utf-8",
    )
    (input_dir / "tool_capacity.json").write_text(
        json.dumps([{"LOT_CD": "LC_A", "EQP_MODEL_CD": "", "MAX_TOOL": 2}], ensure_ascii=False),
        encoding="utf-8",
    )

    result = repair_tool_capacity_folder(folder)
    assert result["status"] == "fixed"

    saved = json.loads((input_dir / "tool_capacity.json").read_text(encoding="utf-8"))
    assert saved == [{"LOT_CD": "LC_A", "EQP_MODEL_CD": "MODEL_A", "MAX_TOOL": 2}]
