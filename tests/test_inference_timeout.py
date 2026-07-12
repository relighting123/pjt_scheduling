"""run_inference – timeout_seconds 조기 종료 테스트."""
from data.loader.fetch import validate_data
from data.loader.preprocess import preprocess
from inference.runner import run_inference


def _env_with_many_lots(n_lots: int = 200):
    discrete = [
        {
            "EQP_ID": "EQP001",
            "LOT_ID": f"LOT{i:04d}",
            "PLAN_PROD_ATTR_VAL": "PPK001",
            "OPER_ID": "OPER001",
            "ST": 10,
            "EQP_MODEL_CD": "A",
            "WF_QTY": 5,
        }
        for i in range(n_lots)
    ]
    raw = {
        "discrete_arrange": discrete,
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 10},
        ],
        "plan": [
            {
                "PLAN_PROD_ATTR_VAL": "PPK001",
                "OPER_ID": "OPER001",
                "D0_PLAN_QTY": 5 * n_lots,
                "D1_PLAN_QTY": 5 * n_lots,
                "PLAN_PRIORITY": 1,
            },
        ],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "split": [
            {
                "PLAN_PROD_ATTR_VAL": "PPK001",
                "OPER_ID": "OPER001",
                "EQP_MODEL_CD": "A",
                "SPLIT_QTY": 10,
            },
        ],
        "lot_master": [{"LOT_ID": "LC001", "LOT_CD": "LC001", "TEMP": "T700"}],
        "batch_info": [
            {
                "PLAN_PROD_ATTR_VAL": "PPK001",
                "OPER_ID": "OPER001",
                "LOT_CD": "LC001",
                "TEMP": "T700",
            },
        ],
        "tool_capacity": [],
        "eqp_initial_state": [],
    }
    errors = validate_data(raw)
    assert not errors, errors
    return preprocess(raw)


def test_timeout_seconds_truncates_early():
    env = _env_with_many_lots(200)
    result = run_inference(
        env, algorithm="earliest_st", record_history=False, timeout_seconds=1e-9,
    )
    assert result["stats"]["timed_out"] is True
    assert result["stats"]["truncated"] is True


def test_no_timeout_completes_normally():
    env = _env_with_many_lots(5)
    result = run_inference(env, algorithm="earliest_st", record_history=False)
    assert result["stats"]["timed_out"] is False
