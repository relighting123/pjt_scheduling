"""flow-balance starving 조건 (net-rate: 후속 WIP / (소비−공급) cover) 테스트."""
from config import CONFIG
from data.loader import load_data, preprocess
from simulation.simulator import SchedulingSimulator


def _sim() -> SchedulingSimulator:
    env_data = preprocess(load_data())
    return SchedulingSimulator(env_data, record_history=False)


def test_downstream_starving_uses_net_rate():
    """소비−공급 순감소율 기준 cover로 starving 판정."""
    sim = _sim()
    ppk = sim._env_data["prod_keys"][0]
    oper = sim._env_data["oper_ids"][0]
    nxt = sim._flow_post(ppk, oper)
    if nxt is None or not sim._has_plan(ppk, nxt):
        return

    sim._reward_cfg.flow_balance_starving_cover_min = 120.0
    state = {"ready": 50, "consume": 1.0, "supply": 0.0}
    sim._ready_wip_qty = lambda _p, _o: state["ready"]  # type: ignore[method-assign]
    sim._oper_capacity_per_min = lambda _p, _o: state["consume"]  # type: ignore[method-assign]
    sim._oper_supply_rate = lambda _p, _o: state["supply"]  # type: ignore[method-assign]

    # 공급 0 → net=1, cover=50 ≤ 120 → starving
    assert sim._downstream_is_starving(ppk, oper) is True

    # 버퍼 충분 → cover=200 > 120 → 안 굶음
    state["ready"] = 200
    assert sim._downstream_is_starving(ppk, oper) is False

    # 공급이 소비를 따라잡음(net≤0, 적체) → 안 굶음 (net-rate 신규 동작)
    state["ready"] = 50
    state["supply"] = 1.0
    assert sim._downstream_is_starving(ppk, oper) is False

    # 부분 공급(consume=2, supply=1 → net=1, cover=50) → starving
    state["consume"] = 2.0
    assert sim._downstream_is_starving(ppk, oper) is True


def test_oper_supply_rate_gated_by_prev_wip():
    """공급율 = 선행 capa, 단 선행 ready 재공이 있을 때만. 첫 공정은 0."""
    sim = _sim()
    ppk = sim._env_data["prod_keys"][0]

    oper_with_prev = next(
        (o for o in sim._env_data["oper_ids"] if sim._flow_prev(ppk, o) is not None),
        None,
    )
    if oper_with_prev is not None:
        sim._oper_capacity_per_min = lambda _p, _o, **_kw: 0.7  # type: ignore[method-assign]
        # 선행 ready 재공 0 → 공급 0
        sim._ready_wip_qty = lambda _p, _o: 0  # type: ignore[method-assign]
        assert sim._oper_supply_rate(ppk, oper_with_prev) == 0.0
        # 선행 ready 재공 > 0 → 공급 = 선행 capa
        sim._ready_wip_qty = lambda _p, _o: 10  # type: ignore[method-assign]
        assert sim._oper_supply_rate(ppk, oper_with_prev) == 0.7

    first_oper = next(
        (o for o in sim._env_data["oper_ids"] if sim._flow_prev(ppk, o) is None),
        None,
    )
    if first_oper is not None:
        assert sim._oper_supply_rate(ppk, first_oper) == 0.0


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
