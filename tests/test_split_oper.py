"""split 규칙 PPK×OPER×EQP_MODEL_CD 키 테스트"""
from data.generator import build_split_rules
from data.loader.preprocess import _build_split_lookup, _resolve_split_qty


def test_build_split_rules_includes_oper_id():
    flow = [
        {"PLAN_PROD_KEY": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK001", "OPER_SEQ": 2, "OPER_ID": "OPER002"},
    ]
    rules = build_split_rules(flow, split_qty=3, models=("A",))
    assert all("OPER_ID" in r for r in rules)
    assert all("EQP_MODEL_CD" in r for r in rules)
    assert all("EQP_MODEL" not in r for r in rules)
    assert {"PPK001", "OPER001", "OPER002"} <= {
        r["PLAN_PROD_KEY"] if k == "PLAN_PROD_KEY" else r["OPER_ID"]
        for r in rules
        for k in ("PLAN_PROD_KEY", "OPER_ID")
    }
    keys = {(r["PLAN_PROD_KEY"], r["OPER_ID"]) for r in rules}
    assert keys == {("PPK001", "OPER001"), ("PPK001", "OPER002")}


def test_resolve_split_qty_by_oper():
    lookup = _build_split_lookup([
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "SPLIT_QTY": 3},
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER002", "EQP_MODEL_CD": "A", "SPLIT_QTY": 5},
    ])
    assert _resolve_split_qty("PPK001", "OPER001", "A", lookup) == 3
    assert _resolve_split_qty("PPK001", "OPER002", "A", lookup) == 5
    assert _resolve_split_qty("PPK001", "OPER003", "A", lookup) is None


def test_sql_example_split_has_oper_id():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "data/sql.example/split.sql").read_text()
    assert "OPER_ID" in text
    assert "PLAN_PROD_KEY" in text
    assert "EQP_MODEL_CD" in text
