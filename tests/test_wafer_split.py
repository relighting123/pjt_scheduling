"""
tests/test_wafer_split_forced_lot_stat.py

_apply_wafer_lot_split()는 discrete_arrange 유래 LOT(전건 자유배정 후보, WAIT)만
다룬다 — SPLIT_QTY 규칙에 따라 분할된다. 이미 확정된 재공(eqp_queue_init)은
lot_info/_apply_wafer_lot_split에 아예 들어오지 않으므로(별도 큐 구조로 관리)
분할 대상이 될 수 없다 — 그 커버리지는 tests/test_eqp_queue_init.py 참고.

회귀 테스트: discrete_raw를 분할 자식으로 확장하는 단계가 r["LOT_ID"](논리
LOT_ID)로 split_children(내부 carrier 키로 색인)을 조회해서, CARRIER_ID가
LOT_ID와 다른 정상적인 1:N 케이스에서는 discrete_raw가 전혀 확장되지 않고
옛 부모 행이 그대로 남아있던 버그가 있었다.
"""
from data.loader.preprocess import _apply_wafer_lot_split, preprocess


def _lot(lot_id, eqp_id, wf_qty):
    return {
        "lot_id": lot_id,
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "oper_id": "OPER001",
        "wf_qty": wf_qty,
        "original_eqp": eqp_id,
        "carrier_id": f"CAR_{lot_id}",
        "logical_lot_id": lot_id,
    }


def _discrete_row(lot_id, eqp_id, wf_qty, carrier_id=None):
    row = {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "PLAN_PROD_ATTR_VAL": "PPK001",
        "OPER_ID": "OPER001", "WF_QTY": wf_qty,
    }
    if carrier_id:
        row["CARRIER_ID"] = carrier_id
    return row


def _run():
    lot_info = {"LOT001": _lot("LOT001", "EQP001", 25)}
    eqp_lot_map = {"EQP001": ["LOT001"]}
    proc_time_matrix = {("LOT001", "EQP001", "OPER001"): 60}
    discrete_raw = [_discrete_row("LOT001", "EQP001", 25)]
    eqp_model_map = {"EQP001": "A"}
    split_lookup = {("PPK001", "OPER001", "A"): 10}

    _apply_wafer_lot_split(
        lot_info, eqp_lot_map, proc_time_matrix, discrete_raw,
        eqp_model_map, split_lookup,
    )
    return lot_info, eqp_lot_map, proc_time_matrix, discrete_raw


def test_wait_lot_is_split_per_split_qty_rule():
    lot_info, eqp_lot_map, proc_time_matrix, discrete_raw = _run()

    assert "LOT001" not in lot_info
    assert len(lot_info) > 1
    assert sum(v["wf_qty"] for v in lot_info.values()) == 25
    assert eqp_lot_map["EQP001"] != ["LOT001"]
    assert len(discrete_raw) > 1


def test_split_expands_discrete_raw_when_carrier_id_differs_from_lot_id():
    discrete = [
        {"EQP_ID": "EQP001", "LOT_ID": "LOT001", "CARRIER_ID": "CAR001",
         "PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "ST": 60,
         "EQP_MODEL_CD": "A", "WF_QTY": 30, "SEQ": 1},
    ]
    plan = [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
             "D0_PLAN_QTY": 30, "D1_PLAN_QTY": 30, "PLAN_PRIORITY": 1}]
    flow = [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}]
    split = [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
              "EQP_MODEL_CD": "A", "SPLIT_QTY": 10}]

    raw = {
        "discrete_arrange": discrete, "plan": plan, "flow": flow,
        "abstract_arrange": [], "split": split,
    }
    data = preprocess(raw, period_key="20260712070000")

    lot_ids = {lot["lot_id"] for lot in data["lots"]}
    assert lot_ids == {"CAR001__S01", "CAR001__S02", "CAR001__S03"}

    # eqp_oper_cap/proc_time_matrix는 _rebuild_eqp_oper_cap()이 확장된
    # discrete_raw를 봐야 채워진다 — 확장이 안 되면 빈 채로 남는다.
    assert data["eqp_oper_cap"] == {"EQP001": ["OPER001"]}
    assert set(data["proc_time_matrix"]) == {
        ("CAR001__S01", "EQP001", "OPER001"),
        ("CAR001__S02", "EQP001", "OPER001"),
        ("CAR001__S03", "EQP001", "OPER001"),
    }
