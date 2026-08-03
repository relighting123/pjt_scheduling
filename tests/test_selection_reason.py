"""
tests/test_selection_reason.py

simulation/selection_reason.py: carrier 배정 시 (PPK,OPER) 버킷 선택 사유 /
carrier(LOT) 선택 사유 산출 로직을 검증한다.

- carrier_select_reason(): _auto_select_lot()과 동일한 우선순위(초기재공 →
  discrete실적 → 동일LOT진행중 → earliest ST → 전환불필요 → oper_in_time →
  tie-break)로 승자를 가른 결정적 기준을 코드로 되짚는다.
- ppk_oper_select_reason(): 배정에 실제 반영된 리워드 항목 + 계획/WIP 현황을
  근거 문구로 만든다.
- 끝에서 SchedulingSimulator를 실제 preprocess() 산출물로 굴려, assign_ppk_oper()
  경로가 schedule 행에 4개 컬럼(PPK_OPER_REASON_CD/CTN, CARRIER_REASON_CD/CTN)을
  실제로 채우는지 통합 검증한다.
"""
from simulation.selection_reason import (
    CARRIER_REASON_LABELS,
    carrier_select_reason,
    ppk_oper_select_reason,
)


class _FakeSim:
    """carrier_select_reason()이 호출하는 4개 메서드만 흉내낸 최소 더블."""

    def __init__(self, active_logical_lots=None):
        self._active = active_logical_lots or set()

    def _lot_conv_discrete_eligible(self, eqp_id, lot):
        return lot.get("_eligible", True)

    def _active_logical_lot_ids(self, ppk, oper_id):
        return self._active

    def earliest_st_combo_score(self, eqp_id, lot):
        return (
            lot.get("_end", 0),
            int(lot.get("_needs_conv", False)),
            lot.get("carrier_id", lot.get("lot_id", "")),
            lot.get("lot_id", ""),
        )

    def _logical_lot_id(self, lot):
        return lot.get("logical_lot_id") or lot.get("lot_id", "")


def _lot(lot_id, **overrides):
    row = {
        "lot_id": lot_id,
        "carrier_id": lot_id,
        "PLAN_PROD_ATTR_VAL": "PPK001",
        "oper_id": "OPER001",
        "is_initial_wip": False,
        "is_abstract": True,
        "oper_in_time": 0,
        "_end": 100,
        "_needs_conv": False,
        "_eligible": True,
    }
    row.update(overrides)
    return row


def test_empty_candidates_yields_none_reason():
    sim = _FakeSim()
    code, text = carrier_select_reason(sim, "EQP001", [])
    assert code == "NONE"
    assert text == CARRIER_REASON_LABELS["NONE"]


def test_single_candidate_reason():
    sim = _FakeSim()
    code, _ = carrier_select_reason(sim, "EQP001", [_lot("CAR001")])
    assert code == "SINGLE_CANDIDATE"


def test_initial_wip_decides_winner():
    sim = _FakeSim()
    candidates = [
        _lot("CAR001", is_initial_wip=True, _end=500),
        _lot("CAR002", is_initial_wip=False, _end=10),
    ]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "INITIAL_WIP"


def test_discrete_actual_decides_winner_when_initial_wip_tied():
    sim = _FakeSim()
    candidates = [
        _lot("CAR001", is_abstract=False, _end=500),
        _lot("CAR002", is_abstract=True, _end=10),
    ]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "DISCRETE_ACTUAL"


def test_lot_continuation_decides_winner():
    sim = _FakeSim(active_logical_lots={"LOTX"})
    candidates = [
        _lot("CAR001", logical_lot_id="LOTX", _end=500),
        _lot("CAR002", logical_lot_id="LOTY", _end=10),
    ]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "LOT_CONTINUATION"


def test_earliest_st_decides_winner_when_others_tied():
    sim = _FakeSim()
    candidates = [
        _lot("CAR001", _end=200),
        _lot("CAR002", _end=50),
    ]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "EARLIEST_ST"


def test_no_conversion_decides_winner_when_end_tied():
    sim = _FakeSim()
    candidates = [
        _lot("CAR001", _end=100, _needs_conv=True),
        _lot("CAR002", _end=100, _needs_conv=False),
    ]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "NO_CONVERSION"


def test_oper_in_time_decides_winner_when_everything_else_tied():
    sim = _FakeSim()
    candidates = [
        _lot("CAR001", oper_in_time=50),
        _lot("CAR002", oper_in_time=5),
    ]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "OPER_IN_TIME"


def test_tie_break_when_all_named_fields_identical():
    sim = _FakeSim()
    candidates = [_lot("CAR001"), _lot("CAR002")]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "TIE_BREAK"


def test_ineligible_candidates_filtered_out_first():
    sim = _FakeSim()
    candidates = [_lot("CAR001", _eligible=False)]
    code, _ = carrier_select_reason(sim, "EQP001", candidates)
    assert code == "NONE"


class _FakeSimForPpkOper:
    def __init__(self, plan_meta=None, completed_qty=None):
        self._env_data = {"plan_meta": plan_meta or {}}
        self.stats = {"completed_qty": completed_qty or {}}


def test_ppk_oper_reason_reports_dominant_term_and_plan_progress():
    sim = _FakeSimForPpkOper(
        plan_meta={("PPK001", "OPER001"): {"d0_plan_qty": 100}},
        completed_qty={("PPK001", "OPER001"): 40},
    )
    terms = {"same_setup": 1.0, "flow_balance": 0.1}
    code, text = ppk_oper_select_reason(
        sim, "EQP001", "PPK001", "OPER001", terms, wip_qty=5,
    )
    assert code == "SAME_SETUP"
    assert "PPK001/OPER001" in text
    assert "D0계획 100" in text
    assert "완료 40" in text


def test_ppk_oper_reason_falls_back_to_default_when_no_terms():
    sim = _FakeSimForPpkOper()
    code, text = ppk_oper_select_reason(
        sim, "EQP001", "PPK001", "OPER001", {}, wip_qty=3,
    )
    assert code == "DEFAULT"
    assert "계획 미정" in text
    assert "WIP 3건" in text


def test_ppk_oper_reason_global_earliest_st_mode_ignores_terms():
    sim = _FakeSimForPpkOper()
    code, text = ppk_oper_select_reason(
        sim, "EQP001", "PPK001", "OPER001", {"same_setup": 1.0}, wip_qty=1,
        mode="GLOBAL_EARLIEST_ST",
    )
    assert code == "GLOBAL_EARLIEST_ST"
    assert "earliest_st" in text


# ── 통합: 실제 SchedulingSimulator 경로가 schedule 행에 사유를 채우는지 ──────

RULE_TIMEKEY = "20260712070000"


def _base_raw(**overrides):
    raw = {
        "discrete_arrange": [],
        "plan": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1}],
        "flow": [{"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
        "abstract_arrange": [
            {"PLAN_PROD_ATTR_VAL": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 60},
        ],
    }
    raw.update(overrides)
    return raw


def _discrete_row(eqp_id, lot_id, carrier_id, st=45, wf_qty=25,
                   ppk="PPK001", oper_id="OPER001"):
    return {
        "EQP_ID": eqp_id, "LOT_ID": lot_id, "CARRIER_ID": carrier_id,
        "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper_id, "ST": st,
        "EQP_MODEL_CD": "A", "WF_QTY": wf_qty,
    }


def test_assign_ppk_oper_populates_reason_columns_on_schedule_row():
    from data.loader.preprocess import preprocess
    from simulation.simulator import SchedulingSimulator

    raw = _base_raw(discrete_arrange=[
        _discrete_row("EQP001", "LOT_A", "CAR_A"),
    ])
    data = preprocess(raw, period_key=RULE_TIMEKEY)
    sim = SchedulingSimulator(data, record_history=False, record_event_log=False)

    eqp_id = sim.current_idle_eqp()
    assert eqp_id == "EQP001"
    reward = sim.assign_ppk_oper(eqp_id, "PPK001", "OPER001")
    assert reward >= 0

    assert len(sim.schedule) == 1
    row = sim.schedule[0]
    assert row["CARRIER_REASON_CD"]
    assert row["CARRIER_REASON_CTN"]
    assert row["PPK_OPER_REASON_CD"]
    assert "PPK001/OPER001" in row["PPK_OPER_REASON_CTN"] or row["PPK_OPER_REASON_CD"] == "DEFAULT"
