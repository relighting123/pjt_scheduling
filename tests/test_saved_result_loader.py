"""저장된 RTS output.json → UI 간트용 result 복원 테스트."""

from data.generator import _build_dataset_bundle, build_abstract_arrange
from data.loader import preprocess, validate_data
from data.writer.rts_json import build_rts_output
from api.server import _result_from_rts_output


def _env_data():
    discrete, plan, flow = _build_dataset_bundle("default")
    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": build_abstract_arrange(discrete, flow),
        "plan": plan,
        "flow": flow,
        "split": [],
        "lot_master": [],
        "tool_capacity": [],
    }
    assert not validate_data(raw), validate_data(raw)
    return preprocess(raw)


def test_rts_output_restores_schedule_for_gantt():
    env_data = _env_data()
    schedule = [
        {
            "EQP_ID": "EQP001",
            "LOT_ID": "LOT001",
            "CARRIER_ID": "CAR001",
            "PLAN_PROD_KEY": "PPK001",
            "OPER_ID": "OPER001",
            "ST": 4,
            "EQP_MODEL": "A",
            "SEQ": 1,
            "START_TM": 0,
            "END_TM": 100,
            "PROC_TIME": 100,
            "WF_QTY": 25,
            "LOT_CD": "LC01",
            "TEMP": "T650",
            "CONVERSION": False,
            "ABSTRACT": False,
            "OPER_IN_TIME": 0,
        },
    ]
    payload = build_rts_output(
        {"schedule": schedule, "conversion_plans": [], "algorithm": "rl"},
        env_data,
    )
    result = _result_from_rts_output(payload, env_data)

    assert result["stats"]["source_file"] == "output.json"
    assert result["schedule"][0]["EQP_ID"] == "EQP001"
    assert result["schedule"][0]["START_TM"] == 0
    assert result["schedule"][0]["END_TM"] == 100
    assert result["stats"]["completed_qty"]["PPK001|OPER001"] == 25
