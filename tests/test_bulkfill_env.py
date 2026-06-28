"""BulkFillEnv(벌크 점유 MDP) 스캐폴딩 단위 테스트."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from data.generator import (
    write_json_bundle,
    build_lot_master_from_discrete,
    build_tool_capacity_from_lots,
)
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from env.bulkfill_env import BulkFillEnv

ST, WF = 5, 25


def _disc(eqp, lot, ppk, oper, seq, carrier):
    return {"EQP_ID": eqp, "LOT_ID": lot, "PLAN_PROD_KEY": ppk, "OPER_ID": oper,
            "ST": ST, "EQP_MODEL_CD": "A", "WF_QTY": WF, "SEQ": seq, "CARRIER_ID": carrier}


def _build_s1(tmp_path: Path, n_lots=6, plan=150, max_tool=3):
    prods = ("PPK001", "PPK002", "PPK003")
    eqps = ("EQP001", "EQP002", "EQP003")
    disc, pl, flow = [], [], []
    lc = 0
    for ppk in prods:
        flow.append({"PLAN_PROD_KEY": ppk, "OPER_SEQ": 1, "OPER_ID": "OPER001"})
        pl.append({"PLAN_PROD_KEY": ppk, "OPER_ID": "OPER001",
                   "D0_PLAN_QTY": plan, "D1_PLAN_QTY": plan, "PLAN_PRIORITY": 1})
        for _ in range(n_lots):
            lc += 1
            for e in eqps:
                disc.append(_disc(e, f"L{lc:03d}", ppk, "OPER001", 1, f"C{lc:03d}"))
    tc = build_tool_capacity_from_lots(build_lot_master_from_discrete(disc), max_tool=max_tool)
    out = tmp_path / "s1"
    write_json_bundle(out, disc, pl, flow, tool_capacity=tc)
    ed = preprocess(load_data(out))
    ed["eqp_selection"] = "order"
    return ed


def _greedy_masked_action(env):
    mask = env.action_masks()
    nb = env._n_bucket
    bmask, smask = mask[:nb], mask[nb:]
    b = int(np.flatnonzero(bmask)[0]) if bmask.any() else 0
    # 가장 큰 블록 레벨 선호 (벌크 검증)
    s = int(np.flatnonzero(smask)[-1]) if smask.any() else 0
    return np.array([b, s], dtype=np.int64)


def _rollout(env, max_steps=60000):
    obs, _ = env.reset()
    info = {}
    for i in range(max_steps):
        a = _greedy_masked_action(env)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return info, env.get_schedule()


def _runs(seq):
    out = []
    for x in seq:
        if not out or out[-1][0] != x:
            out.append([x, 0])
        out[-1][1] += 1
    return [n for _, n in out]


def test_bulkfill_action_space_multidiscrete(tmp_path):
    ed = _build_s1(tmp_path)
    env = BulkFillEnv(ed, record_history=False, record_event_log=False)
    # MultiDiscrete([O*P, L])
    assert tuple(env.action_space.nvec) == (env._n_bucket, env._L)
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (env._n_bucket + env._L,)
    assert mask.dtype == bool


def test_bulkfill_completes_and_produces(tmp_path):
    ed = _build_s1(tmp_path)
    env = BulkFillEnv(ed, record_history=False, record_event_log=False)
    info, sched = _rollout(env)
    assert len(sched) > 0
    produced = sum(info.get("completed_qty", {}).values())
    assert produced > 0


def test_bulkfill_forms_blocks(tmp_path):
    """벌크 점유 → 장비별 동일 제품 연속 run이 1보다 길게 형성."""
    ed = _build_s1(tmp_path)
    env = BulkFillEnv(ed, record_history=False, record_event_log=False)
    _info, sched = _rollout(env)
    by_eqp = defaultdict(list)
    for row in sorted(sched, key=lambda r: (r["EQP_ID"], r["START_TM"])):
        by_eqp[row["EQP_ID"]].append(row["PLAN_PROD_KEY"])
    max_run = max((max(_runs(seq)) for seq in by_eqp.values() if seq), default=0)
    assert max_run >= 2, f"블록이 형성되지 않음 (최대 run={max_run})"


def test_bulkfill_respects_tool_cap(tmp_path):
    """동시 가동 장비 수가 MAX_TOOL을 넘지 않음 (블록 크기 = tool 잔여로 클램프)."""
    ed = _build_s1(tmp_path, max_tool=2)
    env = BulkFillEnv(ed, record_history=False, record_event_log=False)
    _info, sched = _rollout(env)
    events = []
    for row in sched:
        events.append((row["START_TM"], +1, row.get("LOT_CD")))
        events.append((row["END_TM"], -1, row.get("LOT_CD")))
    by_lc = defaultdict(list)
    for t, d, lc in events:
        by_lc[lc].append((t, d))
    max_tool = max(ed["tool_capacity"].values())
    for lc, evs in by_lc.items():
        cur = 0
        for t, d in sorted(evs, key=lambda x: (x[0], x[1])):
            cur += d
            assert cur <= max_tool, f"tool cap 위반: {lc} 동시 {cur} > {max_tool}"


def test_bulk_block_size_clamped_by_tool_remaining(tmp_path):
    """bulk_block_size는 tool '잔여'로 묶인다 (사용 중 제외)."""
    ed = _build_s1(tmp_path, max_tool=2)
    env = BulkFillEnv(ed, record_history=False, record_event_log=False)
    env.reset()
    sim = env.sim
    eqp = sim.current_idle_eqp()
    assert eqp is not None
    feasible = sim.get_feasible_ppk_oper(eqp)
    ppk, oper = sim.ppk_oper_from_flat(feasible[0])
    # 최대 레벨이라도 잔여(=2)와 WIP/계획 상한을 넘지 않음
    n = sim.bulk_block_size(eqp, ppk, oper, level=env._L - 1, n_levels=env._L)
    lot_id = sim._auto_select_lot(eqp, [
        l for l in sim.available_lots(eqp)
        if l["plan_prod_key"] == ppk and l["oper_id"] == oper
    ])
    lot_cd, _ = sim._lot_cd_temp(lot_id, sim.lot_pool.get(lot_id), ppk=ppk, oper_id=oper)
    tool_rem = sim._tool_tracker.remaining(lot_cd, eqp)
    assert 1 <= n <= tool_rem


def test_model_breadth_and_dedication(tmp_path):
    ed = _build_s1(tmp_path)
    env = BulkFillEnv(ed, record_history=False, record_event_log=False)
    env.reset()
    sim = env.sim
    # 단일 모델 A — 3제품×1공정 가공 가능 → breadth=3
    assert sim.model_breadth("EQP001") == 3
    # 모든 버킷이 모델 A만 가능 → 전용도 1.0
    assert sim.bucket_dedication("PPK001", "OPER001") == pytest.approx(1.0)
