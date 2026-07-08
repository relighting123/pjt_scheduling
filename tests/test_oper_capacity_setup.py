"""setup 커밋 + idle 균등분배(방법 B) 기반 oper capacity 테스트."""
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

ST = 60.0


def _raw():
    ppks = ["PPK001", "PPK002"]
    discrete, lot_master = [], []
    n = 0
    for ppk in ppks:
        for _ in range(2):
            n += 1
            lid = f"LOT{n:03d}"
            discrete.append({
                "EQP_ID": "EQP001",
                "LOT_ID": lid,
                "PLAN_PROD_ATTR_VAL": ppk,
                "OPER_ID": "OPER001",
                "ST": ST,
                "EQP_MODEL_CD": "A",
                "WF_QTY": 1,
                "SEQ": 1,
                "CARRIER_ID": f"CAR{n:03d}",
            })
            lot_master.append({"LOT_ID": lid, "LOT_CD": f"LC_{ppk}", "TEMP": "T600"})
    abstract = [
        {"EQP_MODEL_CD": "A", "PLAN_PROD_ATTR_VAL": p, "OPER_ID": "OPER001", "ST": ST}
        for p in ppks
    ]
    plan = [
        {
            "PLAN_PROD_ATTR_VAL": p,
            "OPER_ID": "OPER001",
            "D0_PLAN_QTY": 4,
            "D1_PLAN_QTY": 4,
            "PLAN_PRIORITY": 1,
        }
        for p in ppks
    ]
    flow = [{"PLAN_PROD_ATTR_VAL": p, "OPER_SEQ": 1, "OPER_ID": "OPER001"} for p in ppks]
    batch = [
        {"PLAN_PROD_ATTR_VAL": p, "OPER_ID": "OPER001", "LOT_CD": f"LC_{p}", "TEMP": "T600"}
        for p in ppks
    ]
    return {
        "discrete_arrange": discrete,
        "abstract_arrange": abstract,
        "plan": plan,
        "flow": flow,
        "split": [],
        "lot_master": lot_master,
        "batch_info": batch,
        "tool_capacity": [],
        "eqp_initial_state": [],
    }


def _sim() -> SchedulingSimulator:
    return SchedulingSimulator(preprocess(_raw()), record_history=False)


def test_committed_capacity_uses_prev_setup_only():
    """prev_prod/prev_oper 일치 장비만 전액 capa에 포함."""
    sim = _sim()
    eqp = sim.eqps["EQP001"]
    eqp.status = "idle"
    eqp.prev_prod = "PPK001"
    eqp.prev_oper = "OPER001"

    assert sim._oper_committed_capacity_per_min("PPK001", "OPER001") == 1.0 / ST
    assert sim._oper_committed_capacity_per_min("PPK002", "OPER001") == 0.0
    assert sim._oper_capacity_per_min("PPK001", "OPER001") == 1.0 / ST


def test_idle_split_divides_by_feasible_bucket_count():
    """커밋 0일 때 idle 장비 capa는 feasible 버킷 수(K)로 균등 분배."""
    sim = _sim()
    eqp = sim.eqps["EQP001"]
    eqp.status = "idle"
    eqp.prev_prod = None
    eqp.prev_oper = None

    buckets = sim._eqp_feasible_bucket_keys("EQP001")
    assert len(buckets) == 2

    expected = (1.0 / ST) / 2
    for ppk in ("PPK001", "PPK002"):
        assert sim._oper_committed_capacity_per_min(ppk, "OPER001") == 0.0
        assert abs(sim._oper_idle_split_capacity_per_min(ppk, "OPER001") - expected) < 1e-9
        assert abs(sim._oper_capacity_per_min(ppk, "OPER001") - expected) < 1e-9


def test_committed_takes_priority_over_idle_split():
    """커밋 capa가 있으면 idle 분배 fallback을 쓰지 않음."""
    sim = _sim()
    eqp = sim.eqps["EQP001"]
    eqp.prev_prod = "PPK001"
    eqp.prev_oper = "OPER001"
    eqp.status = "busy"

    committed = 1.0 / ST
    idle_split = sim._oper_idle_split_capacity_per_min("PPK001", "OPER001")

    assert sim._oper_capacity_per_min("PPK001", "OPER001") == committed
    assert idle_split < committed
