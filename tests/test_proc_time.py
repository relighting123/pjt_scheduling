"""ST × wf_qty 실제 소요시간 테스트."""
from data.generator import build_abstract_arrange
from data.loader.fetch import validate_data
from data.loader.preprocess import preprocess
from utils.helpers import effective_proc_time, split_wf_qty


def test_effective_proc_time():
    assert effective_proc_time(5, 25) == 125
    assert effective_proc_time(3, 0) == 0


def test_preprocess_proc_time_after_split():
    """split 이후 자식 LOT wf_qty 기준으로 proc_time 계산."""
    discrete = [
        {
            "EQP_ID": "EQP001",
            "LOT_ID": "LOT001",
            "PLAN_PROD_KEY": "PPK001",
            "OPER_ID": "OPER001",
            "ST": 10,
            "EQP_MODEL": "A",
            "WF_QTY": 25,
        },
    ]
    raw = {
        "discrete_arrange": discrete,
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
        "split": [
            {
                "PLAN_PROD_KEY": "PPK001",
                "OPER_ID": "OPER001",
                "EQP_MODEL_CD": "A",
                "SPLIT_QTY": 3,
            },
        ],
    }
    assert validate_data(raw) == []

    env = preprocess(raw)

    child_qty = split_wf_qty(25, 3)[0]
    child_id = "LOT001__S01"
    child_lot = next(lot for lot in env["lots"] if lot["lot_id"] == child_id)
    assert child_lot["wf_qty"] == child_qty

    st_pw = env["proc_time_matrix"][(child_id, "EQP001", "OPER001")]
    assert st_pw == 10

    row = next(
        r for r in env["arrange_actual_table"]
        if r["lot_id"] == child_id and r["eqp_id"] == "EQP001"
    )
    assert row["st"] == 10
    assert row["proc_time"] == effective_proc_time(10, child_qty)
    assert env["max_proc_time"] >= row["proc_time"]


def test_build_abstract_arrange_uses_eqp_model_cd():
    """abstract_arrange 자동 생성은 입력 스키마에 맞춰 EQP_MODEL_CD를 사용."""
    rows = build_abstract_arrange([
        {
            "EQP_ID": "EQP001",
            "LOT_ID": "LOT001",
            "PLAN_PROD_KEY": "PPK001",
            "OPER_ID": "OPER001",
            "ST": 10,
            "EQP_MODEL": "A",
            "WF_QTY": 25,
        },
    ])

    assert rows == [
        {
            "PLAN_PROD_KEY": "PPK001",
            "OPER_ID": "OPER001",
            "EQP_MODEL_CD": "A",
            "ST": 10,
        },
    ]
