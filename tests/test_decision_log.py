"""추론 decision_log 옵션 테스트."""
from data.generator import _build_dataset_bundle, build_abstract_arrange
from data.loader import preprocess, validate_data
from env.scheduling_env import SchedulingEnv
from inference.runner import run_inference


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


def test_decision_log_records_assignments():
    env_data = _env_data()
    result = run_inference(
        env_data,
        algorithm="earliest_st",
        record_history=False,
        record_decision_log=True,
    )
    log = result["decision_log"]
    assert log, "decision_log should not be empty"
    assert all("step" in row and "status" in row and "reason" in row for row in log)
    assigned = [row for row in log if row["status"] in ("assigned", "action_corrected")]
    assert assigned, "at least one assigned step expected"
    first = assigned[0]
    assert first.get("eqp_id")
    assert first.get("resolved_ppk")
    assert first.get("resolved_oper")
    assert first.get("selected_eqp_id")
    assert first.get("selected_ppk")
    assert first.get("selected_oper_id")
    assert first.get("selected_lot_id")
    assert first.get("selection_reason")
    assert result["stats"]["terminated"] is True
    assert result["stats"]["truncated"] is False
    assert result["stats"]["remaining_wip"] == {}


def test_decision_log_disabled_by_default():
    env_data = _env_data()
    result = run_inference(env_data, algorithm="earliest_st", record_history=False)
    assert result.get("decision_log", []) == []


def test_scheduling_env_decision_log_statuses():
    env_data = _env_data()
    env = SchedulingEnv(env_data, record_history=False, record_decision_log=True)
    env.reset()
    steps = 0
    done = False
    while not done and steps < 5000:
        mask = env.action_masks()
        flat = int(mask.argmax())
        _, _, term, trunc, _ = env.step(flat)
        done = term or trunc
        steps += 1
    log = env.get_decision_log()
    assert log
    statuses = {row["status"] for row in log}
    assert "assigned" in statuses or "action_corrected" in statuses
