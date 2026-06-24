"""data.writer – RTS output.json / 적재 SQL 테스트."""
from datetime import datetime

from data.writer import (
    build_rts_output,
    build_writer_sql_scripts,
    write_sql,
)
from data.writer.rts_json import minutes_to_timekey


def _sample_result():
    base = datetime(2026, 6, 20, 7, 0, 0)
    return {
        "algorithm": "minprogress",
        "schedule": [
            {
                "EQP_ID": "EQP001",
                "LOT_ID": "LOT001",
                "CARRIER_ID": "C1",
                "PLAN_PROD_KEY": "PPK001",
                "OPER_ID": "OP001",
                "EQP_MODEL": "A",
                "SEQ": 1,
                "START_TM": 60,
                "END_TM": 120,
                "WF_QTY": 25,
                "LOT_CD": "LC001",
                "TEMP": "T650",
            },
            {
                "EQP_ID": "EQP001",
                "LOT_ID": "LOT002",
                "CARRIER_ID": "C2",
                "PLAN_PROD_KEY": "PPK002",
                "OPER_ID": "OP001",
                "EQP_MODEL": "A",
                "SEQ": 1,
                "START_TM": 180,
                "END_TM": 240,
                "WF_QTY": 25,
                "LOT_CD": "LC002",
                "TEMP": "T700",
                "CONVERSION": True,
            },
        ],
        "conversion_plans": [
            {
                "eqp_id": "EQP001",
                "eqp_model_cd": "A",
                "oper_id": "OP001",
                "plan_prod_key": "PPK002",
                "from_lot_cd": "LC001",
                "from_temp": "T650",
                "to_lot_cd": "LC002",
                "to_temp": "T700",
                "conv_start_min": 120,
                "conv_end_min": 180,
                "conv_time": 60,
            },
        ],
    }, {
        "sim_base_time": base,
        "prod_keys": ["PPK001"],
        "oper_ids": ["OP001"],
        "eqp_ids": ["EQP001"],
    }


def test_build_rts_output_fields():
    result, env_data = _sample_result()
    out = build_rts_output(
        result, env_data, fac_id="FAC001", rule_timekey="20260620070000",
    )
    assert out["meta"]["FAC_ID"] == "FAC001"
    assert out["meta"]["RULE_TIMEKEY"] == "20260620070000"
    assert len(out["RTS_RSLT_INF"]) == 2
    r0 = out["RTS_RSLT_INF"][0]
    assert r0["SEQ_NO"] == 1
    assert r0["LOT_CD"] == "LC001"
    assert r0["TEMPER_VAL"] == "T650"
    assert r0["PRODUCE_QTY"] == 25
    assert r0["START_TIME"] == minutes_to_timekey(60, env_data["sim_base_time"])
    assert len(out["RTS_EQPCONVPLAN_INF"]) == 1
    conv = out["RTS_EQPCONVPLAN_INF"][0]
    assert conv["LOT_CD"] == "LC001"
    assert conv["TO_LOT_CD"] == "LC002"
    assert conv["PRB_CARD_NO"] == "-"
    assert conv["TO_PRB_CARD_NO"] == "-"
    assert conv["PRB_CARD_NO_LVAL"] == "-"
    assert conv["CONV_TIME"] == 60


def test_write_sql_contains_tables(tmp_path):
    result, env_data = _sample_result()
    payload = build_rts_output(result, env_data, fac_id="FAC001", rule_timekey="20260620070000")
    scripts = build_writer_sql_scripts(payload)
    assert "rts_rslt_inf.sql" in scripts
    assert "DELETE FROM RTS_RSLT_INF" in scripts["rts_rslt_inf.sql"]
    assert "INSERT INTO RTS_RSLT_INF" in scripts["rts_rslt_inf.sql"]
    assert "INSERT INTO RTS_EQPCONVPLAN_INF" in scripts["rts_eqpconvplan_inf.sql"]
    paths = write_sql(payload, tmp_path)
    assert len(paths) >= 2
    assert (tmp_path / "rts_rslt_inf.sql").is_file()
