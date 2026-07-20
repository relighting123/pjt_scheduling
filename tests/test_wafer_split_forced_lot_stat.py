"""
tests/test_wafer_split_forced_lot_stat.py

_apply_wafer_lot_split()는 LOT_STAT_CD가 PROC/LOAD/RESV/SELE(이미 확정된 재공)인
LOT은 분할하지 않고 지정된 EQP_ID에 원래 수량 그대로 유지해야 한다.
WAIT LOT은 기존대로 SPLIT_QTY 규칙에 따라 분할된다.
"""
from data.loader.preprocess import _apply_wafer_lot_split


def _lot(lot_id, eqp_id, wf_qty, lot_stat_cd="WAIT"):
    return {
        "lot_id": lot_id,
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "oper_id": "OPER001",
        "wf_qty": wf_qty,
        "original_eqp": eqp_id,
        "carrier_id": f"CAR_{lot_id}",
        "logical_lot_id": lot_id,
        "lot_stat_cd": lot_stat_cd,
    }


def _discrete_row(lot_id, eqp_id, wf_qty, carrier_id=None):
    row = {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "PLAN_PROD_ATTR_VAL": "PPK001",
        "OPER_ID": "OPER001", "WF_QTY": wf_qty,
    }
    if carrier_id:
        row["CARRIER_ID"] = carrier_id
    return row


def _run(lot_stat_cd):
    lot_info = {"LOT001": _lot("LOT001", "EQP001", 25, lot_stat_cd=lot_stat_cd)}
    eqp_lot_map = {"EQP001": ["LOT001"]}
    proc_time_matrix = {("LOT001", "EQP001", "OPER001"): 60}
    discrete_raw = [_discrete_row("LOT001", "EQP001", 25)]
    eqp_model_map = {"EQP001": "A"}
    split_lookup = {("PPK001", "OPER001", "A"): 10}
    eqp_forced_queue = {"EQP001": ["LOT001"]} if lot_stat_cd != "WAIT" else {}

    _apply_wafer_lot_split(
        lot_info, eqp_lot_map, proc_time_matrix, discrete_raw,
        eqp_model_map, split_lookup, eqp_forced_queue,
    )
    return lot_info, eqp_lot_map, proc_time_matrix, discrete_raw, eqp_forced_queue


def test_wait_lot_is_split_per_split_qty_rule():
    lot_info, eqp_lot_map, proc_time_matrix, discrete_raw, _ = _run("WAIT")

    assert "LOT001" not in lot_info
    assert len(lot_info) > 1
    assert sum(v["wf_qty"] for v in lot_info.values()) == 25
    assert eqp_lot_map["EQP001"] != ["LOT001"]
    assert len(discrete_raw) > 1


def test_forced_lot_stat_cd_lots_are_not_split():
    for stat in ("PROC", "LOAD", "RESV", "SELE"):
        lot_info, eqp_lot_map, proc_time_matrix, discrete_raw, eqp_forced_queue = _run(stat)

        assert list(lot_info.keys()) == ["LOT001"]
        assert lot_info["LOT001"]["wf_qty"] == 25
        assert eqp_lot_map["EQP001"] == ["LOT001"]
        assert eqp_forced_queue["EQP001"] == ["LOT001"]
        assert proc_time_matrix == {("LOT001", "EQP001", "OPER001"): 60}
        assert len(discrete_raw) == 1
        assert discrete_raw[0]["LOT_ID"] == "LOT001"
        assert discrete_raw[0]["WF_QTY"] == 25
