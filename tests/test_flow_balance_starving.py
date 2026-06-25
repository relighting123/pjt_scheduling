"""flow-balance starving 조건 (ready WIP / capa cover) 테스트."""
from config import CONFIG
from data.loader import load_data, preprocess
from simulation.simulator import SchedulingSimulator


def _sim() -> SchedulingSimulator:
    env_data = preprocess(load_data())
    return SchedulingSimulator(env_data, record_history=False)


def test_downstream_starving_when_cover_below_threshold():
    sim = _sim()
    ppk = sim._env_data["prod_keys"][0]
    oper = sim._env_data["oper_ids"][0]
    nxt = sim._flow_post(ppk, oper)
    if nxt is None or not sim._has_plan(ppk, nxt):
        return

    sim._reward_cfg.flow_balance_starving_cover_min = 120.0

    calls = {"ready": 50, "capa": 1.0}  # cover = 50 min

    sim._ready_wip_qty = lambda _p, _o: calls["ready"]  # type: ignore[method-assign]
    sim._oper_capacity_per_min = lambda _p, _o: calls["capa"]  # type: ignore[method-assign]

    assert sim._downstream_is_starving(ppk, oper) is True

    calls["ready"] = 200  # cover = 200 min
    assert sim._downstream_is_starving(ppk, oper) is False


def test_flow_balance_feeding_bonus_only_when_starving():
    sim = _sim()
    ppk = sim._env_data["prod_keys"][0]
    oper = sim._env_data["oper_ids"][0]
    if sim._flow_post(ppk, oper) is None:
        return

    sim._reward_cfg.w_flow_balance = 2.0
    wips = sim.get_wip_waiting()
    if not wips or f"{ppk}|{oper}" not in wips:
        return

    base = sim._flow_balance_reward(ppk, oper)

    sim._downstream_is_starving = lambda _p, _o: True  # type: ignore[method-assign]
    with_bonus = sim._flow_balance_reward(ppk, oper)
    assert with_bonus == base + 1.0  # w_flow_balance * 0.5

    sim._downstream_is_starving = lambda _p, _o: False  # type: ignore[method-assign]
    without_bonus = sim._flow_balance_reward(ppk, oper)
    assert without_bonus == base


def test_reward_params_include_starving_cover_min():
    from config import reward_params_dict

    assert "flow_balance_starving_cover_min" in reward_params_dict()
    assert reward_params_dict()["flow_balance_starving_cover_min"] == CONFIG.reward.flow_balance_starving_cover_min
