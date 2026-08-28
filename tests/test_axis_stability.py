"""
tests/test_axis_stability.py

축(공정/제품/설비/모델) 인덱스 고정 검증 — `data/loader/axis_map.py`.

배경: `preprocess()`는 축 목록을 매 회차 그 회차 데이터로 재산출해서, 앞선 키가
빠지면 뒤가 전부 한 칸씩 밀렸다. 정책 action 버킷이 `oper_idx*P + prod_idx`라
이 시프트는 학습된 슬롯 대응을 통째로 어긋나게 한다(실측: 물리 상태가 동일한데
제품 축만 한 칸 회전 → 스케줄 일치율 16.9%).
"""
import copy
import json

import pytest

from config import CONFIG
from data.generator import generate_sample_data
from data.loader.axis_map import (
    build_axis_map, load_axis_map, order_axis, positional_index_map, save_axis_map,
)
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess

RULE_TIMEKEY = "20260712070000"


# ── order_axis 단위 ─────────────────────────────────────────────────────────

def test_no_stored_map_falls_back_to_sorted():
    assert order_axis(["C", "A", "B"], None, "prod_keys") == ["A", "B", "C"]
    assert order_axis(["C", "A"], {}, "prod_keys") == ["A", "C"]


def test_stored_order_is_preserved():
    stored = {"prod_keys": {"order": ["PPK003", "PPK001", "PPK002"], "last_seen": {}}}
    assert order_axis(["PPK001", "PPK002", "PPK003"], stored, "prod_keys") == [
        "PPK003", "PPK001", "PPK002",
    ]


def test_new_keys_are_appended_sorted_not_renumbered():
    stored = {"prod_keys": {"order": ["PPK003", "PPK001"], "last_seen": {}}}
    got = order_axis(["PPK001", "PPK003", "PPK009", "PPK005"], stored, "prod_keys")
    assert got == ["PPK003", "PPK001", "PPK005", "PPK009"]
    # 기존 키의 슬롯이 그대로여야 한다
    assert got.index("PPK003") == 0 and got.index("PPK001") == 1


def test_absent_key_keeps_its_slot_so_later_keys_do_not_shift():
    """이번 회차에 PPK001이 없어도 자리를 지켜, 뒤 키가 밀리지 않아야 한다."""
    stored = {"prod_keys": {"order": ["PPK001", "PPK002", "PPK003"], "last_seen": {}}}
    got = order_axis(["PPK002", "PPK003"], stored, "prod_keys")
    assert got == ["PPK001", "PPK002", "PPK003"]
    idx = positional_index_map(got)
    assert idx["PPK002"] == 1 and idx["PPK003"] == 2      # 시프트 없음


def test_cap_overflow_evicts_inactive_lru_first_and_never_active(monkeypatch):
    monkeypatch.setattr(CONFIG.env, "max_prod_count", 3)
    stored = {"prod_keys": {
        "order": ["OLD1", "OLD2", "ACT1", "ACT2"],
        "last_seen": {"OLD1": "20260101000000", "OLD2": "20260601000000",
                      "ACT1": "20260801000000", "ACT2": "20260801000000"},
    }}
    got = order_axis(["ACT1", "ACT2"], stored, "prod_keys")
    assert len(got) == 3
    # 활성 키는 절대 밀려나지 않는다
    assert "ACT1" in got and "ACT2" in got
    # 비활성 중 last_seen 이 가장 오래된 OLD1 이 먼저 빠진다
    assert "OLD1" not in got and "OLD2" in got


def test_positional_index_map_matches_list_positions():
    items = ["B", "A", "C"]
    idx = positional_index_map(items)
    assert all(idx[k] == i for i, k in enumerate(items))


# ── 저장/로드 왕복 ───────────────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    ed = {"oper_ids": ["O2", "O1"], "prod_keys": ["P1"],
          "eqp_ids": ["E1"], "eqp_models": ["M1"]}
    path = tmp_path / "axis_map.json"
    save_axis_map(ed, timestamp=RULE_TIMEKEY, path=path)
    loaded = load_axis_map(path)
    assert loaded["oper_ids"]["order"] == ["O2", "O1"]     # 정렬하지 않고 그대로
    assert loaded["oper_ids"]["last_seen"]["O1"] == RULE_TIMEKEY


def test_save_merges_previous_last_seen(tmp_path):
    path = tmp_path / "axis_map.json"
    save_axis_map({"prod_keys": ["P1", "P2"]}, timestamp="20260101000000", path=path)
    save_axis_map({"prod_keys": ["P2"]}, timestamp="20260801000000", path=path)
    loaded = load_axis_map(path)
    # 이번에 없던 P1 의 과거 last_seen 이 남아 있어야 LRU 판단이 가능하다
    assert loaded["prod_keys"]["last_seen"]["P1"] == "20260101000000"
    assert loaded["prod_keys"]["last_seen"]["P2"] == "20260801000000"


def test_missing_file_returns_none(tmp_path):
    assert load_axis_map(tmp_path / "nope.json") is None


# ── preprocess 통합 ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_input(tmp_path_factory):
    d = tmp_path_factory.mktemp("axis_input")
    generate_sample_data(scenario="default", output_dir=d)
    return load_data(d)


def _drop_ppk(raw: dict, ppk: str) -> dict:
    r = copy.deepcopy(raw)
    for k in ("discrete_arrange", "abstract_arrange", "plan", "flow", "split", "lot_master"):
        if isinstance(r.get(k), list):
            r[k] = [x for x in r[k] if x.get("PLAN_PROD_ATTR_VAL") != ppk]
    return r


def test_without_map_removing_first_ppk_shifts_indices(raw_input, tmp_path, monkeypatch):
    """축 고정을 끄면(파일 없음) 기존 동작 그대로 — 시프트가 발생한다."""
    monkeypatch.setenv("AXIS_MAP_CONFIG", str(tmp_path / "absent.json"))
    base = preprocess(raw_input, period_key=RULE_TIMEKEY)
    first = base["prod_keys"][0]
    after = preprocess(_drop_ppk(raw_input, first), period_key=RULE_TIMEKEY)
    assert first not in after["prod_keys"]
    # 남은 키가 한 칸씩 앞으로 밀린다 = 이 변경 전의 문제 상황
    assert after["prod_idx"][base["prod_keys"][1]] == 0


def test_with_map_removing_first_ppk_keeps_remaining_indices(raw_input, tmp_path, monkeypatch):
    """축 맵이 있으면 앞 키가 빠져도 남은 키의 인덱스가 그대로여야 한다."""
    path = tmp_path / "axis_map.json"
    monkeypatch.setenv("AXIS_MAP_CONFIG", str(path))

    base = preprocess(raw_input, period_key=RULE_TIMEKEY)
    save_axis_map(base, timestamp=RULE_TIMEKEY, path=path)
    first, second, third = base["prod_keys"][0], base["prod_keys"][1], base["prod_keys"][2]

    after = preprocess(_drop_ppk(raw_input, first), period_key=RULE_TIMEKEY)
    assert after["prod_idx"][second] == base["prod_idx"][second]
    assert after["prod_idx"][third] == base["prod_idx"][third]
    # 빠진 키는 자리를 지켜 뒤를 밀지 않는다
    assert after["prod_keys"][0] == first


def test_axis_list_and_index_map_stay_consistent(raw_input, tmp_path, monkeypatch):
    """axis_list[i] == key  ⇔  idx_map[key] == i 불변식."""
    path = tmp_path / "axis_map.json"
    monkeypatch.setenv("AXIS_MAP_CONFIG", str(path))
    base = preprocess(raw_input, period_key=RULE_TIMEKEY)
    save_axis_map(base, timestamp=RULE_TIMEKEY, path=path)
    ed = preprocess(_drop_ppk(raw_input, base["prod_keys"][0]), period_key=RULE_TIMEKEY)
    for lst, idx in (("oper_ids", "oper_idx"), ("prod_keys", "prod_idx")):
        for i, key in enumerate(ed[lst]):
            assert ed[idx][key] == i, f"{lst}[{i}]={key} 인덱스 불일치"


def test_corrupt_map_is_rejected_loudly(tmp_path, monkeypatch):
    path = tmp_path / "axis_map.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    monkeypatch.setenv("AXIS_MAP_CONFIG", str(path))
    with pytest.raises(ValueError):
        load_axis_map()
