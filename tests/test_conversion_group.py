"""전환 그룹(conversion_group) 제약 테스트.

conversion_group.json으로 (LOT_CD, TEMP) 그룹을 지정하면, 같은 그룹 안에서만
전환이 허용되고 다른 그룹으로의 전환은 배정 후보에서 제외된다.
"""
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

TEMP = "T600"
LC = {"PPK001": "LC_A", "PPK002": "LC_B"}


def _raw(conversion_group):
    ppks = ["PPK001", "PPK002"]
    eqps = ["EQP001", "EQP002"]
    discrete, lots, lot_master = [], [], []
    n = 0
    for pi, ppk in enumerate(ppks):
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
    tool = [{"LOT_CD": lc, "TEMP": TEMP, "MAX_TOOL": 99} for lc in LC.values()]
    return {
        "discrete_arrange": discrete, "abstract_arrange": abstract,
        "plan": plan, "flow": flow, "split": [], "lot_master": lot_master,
        "batch_info": batch, "tool_capacity": tool, "eqp_initial_state": [],
        "conversion_group": conversion_group,
    }


def _sim(conversion_group):
    ed = preprocess(_raw(conversion_group))
    ed["sim_end_minutes"] = 480
    sim = SchedulingSimulator(ed, record_history=False, record_event_log=False)
    return ed, sim


def _ppk002_flat(ed, sim):
    return sim.ppk_oper_flat_index("OPER001", "PPK002")


def test_no_group_is_inactive():
    ed, sim = _sim([])
    assert sim._conv_groups_active is False
    sim.eqps["EQP001"].prev_lot_cd = "LC_A"
    sim.eqps["EQP001"].prev_temp = TEMP
    # 그룹 제약 없으면 전환 필요한 PPK002도 막히지 않음
    assert sim._conversion_group_blocks("EQP001", "LC_B", TEMP) is False


def test_cross_group_conversion_blocked():
    groups = [
        {"GROUP_ID": "G1", "LOT_CD": "LC_A", "TEMP": TEMP},
        {"GROUP_ID": "G2", "LOT_CD": "LC_B", "TEMP": TEMP},
    ]
    ed, sim = _sim(groups)
    assert sim._conv_groups_active is True
    sim.eqps["EQP001"].prev_lot_cd = "LC_A"
    sim.eqps["EQP001"].prev_temp = TEMP
    sim._invalidate_caches()
    # 다른 그룹(G2)으로의 전환은 차단
    assert sim._conversion_group_blocks("EQP001", "LC_B", TEMP) is True
    # 동일 셋업(전환 없음)은 허용
    assert sim._conversion_group_blocks("EQP001", "LC_A", TEMP) is False
    # feasible 후보에서 PPK002(LC_B) 제외
    feasible = sim.get_feasible_ppk_oper("EQP001")
    assert _ppk002_flat(ed, sim) not in feasible


def test_same_group_conversion_allowed():
    groups = [
        {"GROUP_ID": "G1", "LOT_CD": "LC_A", "TEMP": TEMP},
        {"GROUP_ID": "G1", "LOT_CD": "LC_B", "TEMP": TEMP},
    ]
    ed, sim = _sim(groups)
    sim.eqps["EQP001"].prev_lot_cd = "LC_A"
    sim.eqps["EQP001"].prev_temp = TEMP
    sim._invalidate_caches()
    # 같은 그룹(G1) 안의 전환은 허용
    assert sim._conversion_group_blocks("EQP001", "LC_B", TEMP) is False
    feasible = sim.get_feasible_ppk_oper("EQP001")
    assert _ppk002_flat(ed, sim) in feasible


def test_first_assignment_not_blocked():
    groups = [
        {"GROUP_ID": "G1", "LOT_CD": "LC_A", "TEMP": TEMP},
        {"GROUP_ID": "G2", "LOT_CD": "LC_B", "TEMP": TEMP},
    ]
    ed, sim = _sim(groups)
    # prev_lot_cd=None(첫 배정) → 어떤 그룹도 차단하지 않음
    assert sim.eqps["EQP002"].prev_lot_cd is None
    assert sim._conversion_group_blocks("EQP002", "LC_B", TEMP) is False
