"""전환(컨버전) 횟수 상한 테스트."""
from data.loader.preprocess import preprocess
from inference.runner import run_inference
from simulation.simulator import SchedulingSimulator

TEMP = "T600"
LC = {"PPK001": "LC_A", "PPK002": "LC_B"}


def _env_data():
    ppks = ["PPK001", "PPK002"]
    eqps = ["EQP001", "EQP002"]
    discrete, lot_master = [], []
    n = 0
    for ppk in ppks:
        for ci in range(4):
            n += 1
            lid = f"LOT{n:03d}"
            discrete.append({
                "EQP_ID": eqps[ci % 2], "LOT_ID": lid, "PLAN_PROD_KEY": ppk,
                "OPER_ID": "OPER001", "ST": 60, "EQP_MODEL_CD": "A",
                "WF_QTY": 1, "SEQ": 1, "CARRIER_ID": f"CAR{n:03d}",
            })
            lot_master.append({"LOT_ID": lid, "LOT_CD": LC[ppk], "TEMP": TEMP})
    abstract = [{"EQP_MODEL_CD": "A", "PLAN_PROD_KEY": p, "OPER_ID": "OPER001", "ST": 60}
                for p in ppks]
    plan = [{"PLAN_PROD_KEY": p, "OPER_ID": "OPER001", "D0_PLAN_QTY": 4,
             "D1_PLAN_QTY": 4, "PLAN_PRIORITY": 1} for p in ppks]
    flow = [{"PLAN_PROD_KEY": p, "OPER_SEQ": 1, "OPER_ID": "OPER001"} for p in ppks]
    batch = [{"PLAN_PROD_KEY": p, "OPER_ID": "OPER001", "LOT_CD": LC[p], "TEMP": TEMP}
             for p in ppks]
    tool = [{"LOT_CD": lc, "EQP_MODEL_CD": "A", "MAX_TOOL": 99} for lc in LC.values()]
    eqp_init = [{
        "EQP_ID": "EQP001",
        "LOT_CD": LC["PPK001"],
        "TEMP": TEMP,
        "PLAN_PROD_KEY": "PPK001",
        "OPER_ID": "OPER001",
    }]
    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": [],
        "lot_master": lot_master,
        "batch_info": batch,
        "tool_capacity": tool,
        "eqp_initial_state": eqp_init,
    }
    return preprocess(raw)


def test_conversion_limit_blocks_when_global_cap_reached():
    env_data = dict(_env_data())
    env_data["max_conversions"] = 0
    sim = SchedulingSimulator(env_data, record_history=False)
    assert sim._conversion_limit_blocks("EQP001", LC["PPK002"], TEMP) is True
    assert sim._conversion_limit_blocks("EQP001", LC["PPK001"], TEMP) is False


def test_conversion_limit_per_eqp():
    env_data = dict(_env_data())
    env_data["max_conversions_per_eqp"] = 0
    sim = SchedulingSimulator(env_data, record_history=False)
    assert sim._conversion_limit_blocks("EQP001", LC["PPK002"], TEMP) is True


def test_run_inference_respects_global_conversion_cap():
    env_data = dict(_env_data())
    result = run_inference(
        env_data,
        algorithm="earliest_st",
        record_history=False,
        max_conversions=0,
    )
    assert result["stats"]["conversions"] == 0
    assert not any(r.get("CONVERSION") for r in result["schedule"])


def test_run_inference_applies_conversion_minutes():
    env_data = dict(_env_data())
    result = run_inference(
        env_data,
        algorithm="earliest_st",
        record_history=False,
        conversion_minutes=15,
    )
    assert result["stats"]["conversions"] >= 1
    conv_plans = result.get("conversion_plans", [])
    assert conv_plans, "전환이 발생해야 conversion_minutes 검증 가능"
    for plan in conv_plans:
        assert plan.get("conv_time") == 15


def test_run_inference_allows_one_conversion_with_cap_one():
    env_data = dict(_env_data())
    unlimited = run_inference(env_data, algorithm="earliest_st", record_history=False)
    capped = run_inference(
        env_data,
        algorithm="earliest_st",
        record_history=False,
        max_conversions=1,
    )
    assert unlimited["stats"]["conversions"] >= 1
    assert capped["stats"]["conversions"] == 1
    assert capped["stats"]["conversions"] <= unlimited["stats"]["conversions"]
