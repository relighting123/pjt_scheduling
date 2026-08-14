# sametool_setup 지표/리워드 확장 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BATCHID(LOT_CD/TEMP)는 유지한 채 PPK/OPER만 바뀌는 전환("sametool_setup")을 실제 TOOL
전환과 구분해 카운트하고, RL 리워드에 소액 페널티로 반영한다.

**Architecture:** `simulator.py::_execute_assignment()`은 어느 에이전트(RL/휴리스틱/DedicationAgent)가
액션을 골랐든 공통으로 거치는 배정 실행 경로다. 이미 그 안에서 새 BATCHID(`lot_cd, temp`)를 계산한
직후, 아직 갱신되지 않은 `eqp.prev_lot_cd/prev_temp`(이전 BATCHID)를 가진 채로
`_same_setup_reward()`가 호출된다 — 이 시점에 배치 비교에 필요한 값이 모두 있으므로, 새 함수를
만들지 않고 이 함수 하나만 확장한다. 버킷 선택(어느 PPK/OPER를 고를지) 로직 자체는 건드리지
않는다 — `agent/dedication_agent.py`가 이미 이 설계가 의도한 STAY/SWITCH 판단(커버리지 기반 유지,
슬랙 기반 전환)을 구현하고 있기 때문이다.

**Tech Stack:** Python 3.13, pytest, 기존 `config.py`/`simulation/simulator.py`/`agent/kpi_eval.py` 그대로.

## Global Constraints

- 기존 `w_same_setup` 필드/동작(직전과 완전 동일할 때 보너스)은 변경하지 않는다 — 설계서
  `docs/superpowers/specs/2026-08-14-sametool-setup-dedication-design.md` 3절/7절.
- `agent/dedication_agent.py`는 수정하지 않는다 — 설계서 2절("중요 — 계획 단계에서 발견").
- 실제 TOOL 전환(BATCHID 자체가 다름) 카운트는 기존 `stats["conversions"]` 경로를 그대로 쓴다.
  `sametool_setup_count`와 별개 카운터이며, 이 계획에서 `stats["conversions"]` 계산 로직은
  건드리지 않는다.
- 신규 config 필드는 기존 `RewardConfig` dataclass 패턴(필드 추가 + `reward_params_dict()` +
  `apply_reward_params()` 3곳 동시 수정)을 그대로 따른다 — `config.py:709-814`.
- 새 stats 키(`sametool_setup_count`)는 `self.stats` 딕셔너리 초기화 시점(`simulator.py:294`)에
  다른 키와 함께 선언해, 나머지 코드에서 `.get()` 없이 직접 접근해도 KeyError가 나지 않게 한다.

---

## File Structure

| 파일 | 역할 |
|---|---|
| `config.py` | `RewardConfig.w_sametool_setup` 필드 + `reward_params_dict()`/`apply_reward_params()` 등록 (수정) |
| `simulation/simulator.py` | `stats["sametool_setup_count"]` 초기화, `_same_setup_reward()` 배치 비교 확장, 호출부 인자 전달, history 스냅샷에 노출 (수정) |
| `agent/kpi_eval.py` | `KPI`/`KPIResult`에 `sametool_setup` 필드 추가, `_episode_kpi()`에서 채움 (수정) |
| `tests/test_sametool_setup.py` | 위 세 가지 변경을 검증하는 신규 테스트 (생성) |

---

### Task 1: `config.py` — `w_sametool_setup` 리워드 가중치 추가

**Files:**
- Modify: `config.py:713` (RewardConfig 필드), `config.py:778` (`reward_params_dict`), `config.py:803-809` (`apply_reward_params`)
- Test: `tests/test_sametool_setup.py` (신규 생성)

**Interfaces:**
- Produces: `CONFIG.reward.w_sametool_setup: float` (기본값 `0.5`), `reward_params_dict()["w_sametool_setup"]`, `apply_reward_params({"w_sametool_setup": ...})`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sametool_setup.py` 새로 생성:

```python
"""
tests/test_sametool_setup.py

sametool_setup(BATCHID는 동일·PPK/OPER만 전환)을 실제 TOOL 전환과 구분해
카운트/리워드하는 기능을 검증한다. 설계서:
docs/superpowers/specs/2026-08-14-sametool-setup-dedication-design.md
"""
from pathlib import Path

import pytest

from config import CONFIG, apply_reward_params, reward_params_dict


def test_reward_params_round_trip_includes_w_sametool_setup():
    original = CONFIG.reward.w_sametool_setup
    try:
        d = reward_params_dict()
        assert "w_sametool_setup" in d
        assert d["w_sametool_setup"] == pytest.approx(original)

        apply_reward_params({"w_sametool_setup": 0.75})
        assert CONFIG.reward.w_sametool_setup == pytest.approx(0.75)
    finally:
        CONFIG.reward.w_sametool_setup = original
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `pytest tests/test_sametool_setup.py::test_reward_params_round_trip_includes_w_sametool_setup -v`
Expected: FAIL — `AttributeError: 'RewardConfig' object has no attribute 'w_sametool_setup'`
(또는 `KeyError`/`AssertionError`, 필드가 아직 없으므로)

- [ ] **Step 3: `config.py`에 필드 추가**

`config.py:713` 바로 아래(`w_same_setup:      float = 1.0` 다음 줄)에 추가:

```python
    # 배치(BATCHID=LOT_CD/TEMP)는 유지한 채 PPK/OPER만 바뀌는 경우(=sametool_setup,
    # TOOL 전환 없음)의 소액 페널티. w_same_setup과 별개 축 — 직전과 완전히
    # 동일하면 보너스(w_same_setup), 배치만 같고 PPK/OPER가 바뀌면 이 페널티.
    w_sametool_setup:  float = 0.5
```

`config.py:778` (`reward_params_dict` 함수 안, `"w_same_setup": r.w_same_setup,` 다음 줄)에 추가:

```python
        "w_sametool_setup": r.w_sametool_setup,
```

`config.py:802-809` (`apply_reward_params` 함수 안 `float_keys` 튜플, `"w_same_setup", "w_idle_per_min",` 부분)을 아래처럼 수정:

```python
    float_keys = (
        "w_same_setup", "w_sametool_setup", "w_idle_per_min",
        "w_plan_hit", "w_pacing", "pacing_coverage_scale", "w_conversion",
        "w_avoidable_conversion", "conversion_amortize_factor",
        "w_bulk_block_bonus", "w_dedication_misuse", "w_redundant_cover",
        "w_flow_balance", "flow_balance_starving_cover_min", "reward_clip",
        "w_terminal_throughput", "w_terminal_conversion", "terminal_reward_clip",
    )
```

- [ ] **Step 4: 테스트 실행해 통과 확인**

Run: `pytest tests/test_sametool_setup.py::test_reward_params_round_trip_includes_w_sametool_setup -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add config.py tests/test_sametool_setup.py
git commit -m "feat: add w_sametool_setup reward weight"
```

---

### Task 2: `simulator.py` — `_same_setup_reward`에 배치 비교 추가

**Files:**
- Modify: `simulation/simulator.py:294-300` (stats 초기화), `simulation/simulator.py:1691-1712` (`_same_setup_reward`), `simulation/simulator.py:2324` (호출부)
- Test: `tests/test_sametool_setup.py` (Task 1에서 만든 파일에 이어서 작성)

**Interfaces:**
- Consumes: `CONFIG.reward.w_sametool_setup`(Task 1에서 추가)
- Produces: `sim.stats["sametool_setup_count"]: int`, `SchedulingSimulator._same_setup_reward(self, eqp, ppk, oper_id, wf_qty, lot_cd, temp) -> float` (시그니처에 `lot_cd, temp` 2개 인자 추가됨 — 이후 태스크가 이 시그니처를 그대로 사용)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sametool_setup.py`에 이어서 추가(파일 상단 import 블록에 아래 추가):

```python
from data.generator import generate_sample_data
from data.loader.fetch import load_data
from data.loader.preprocess import preprocess
from simulation.simulator import SchedulingSimulator

RULE_TIMEKEY = "20260712070000"


@pytest.fixture()
def sim(tmp_path: Path) -> SchedulingSimulator:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    generate_sample_data(scenario="default", output_dir=input_dir)
    raw = load_data(input_dir)
    env_data = preprocess(raw, period_key=RULE_TIMEKEY)
    return SchedulingSimulator(env_data, record_history=False, record_event_log=False)


def _real_ppk_oper(sim: SchedulingSimulator):
    """샘플 데이터 안에서 실제로 feasible한 (ppk, oper) 하나를 가져온다."""
    flat, _ = next(iter(sim.get_feasible_assignments()))
    return sim.ppk_oper_from_flat(flat)


def test_same_ppk_oper_still_gives_bonus_unchanged(sim, monkeypatch):
    """배치 비교 로직 추가 후에도 기존 same_oper&&same_prod 보너스 경로는 그대로."""
    monkeypatch.setattr(CONFIG.reward, "w_same_setup", 1.0)
    ppk, oper_id = _real_ppk_oper(sim)
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = ppk
    eqp.prev_oper = oper_id
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, ppk, oper_id, 1, "LOTA", "T1")

    assert reward == pytest.approx(1.0)
    assert sim.stats["sametool_setup_count"] == before  # 카운트 변화 없음


def test_same_batch_different_ppk_oper_is_sametool_setup(sim, monkeypatch):
    monkeypatch.setattr(CONFIG.reward, "w_sametool_setup", 0.5)
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = "PPKX"
    eqp.prev_oper = "OP1"
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTA", "T1")

    assert reward == pytest.approx(-0.5)
    assert sim.stats["sametool_setup_count"] == before + 1


def test_different_batch_is_not_counted_as_sametool_setup(sim):
    """BATCHID 자체가 다르면(=TOOL 전환) 이 함수는 관여하지 않는다(0.0, 카운트 변화 없음)."""
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = "PPKX"
    eqp.prev_oper = "OP1"
    eqp.prev_lot_cd = "LOTA"
    eqp.prev_temp = "T1"
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTB", "T9")

    assert reward == 0.0
    assert sim.stats["sametool_setup_count"] == before


def test_first_assignment_no_prior_batch_is_not_sametool_setup(sim):
    """prev_lot_cd가 None(첫 배정)이면 비교 대상이 없어 0.0."""
    eqp = next(iter(sim.eqps.values()))
    eqp.prev_prod = None
    eqp.prev_oper = None
    eqp.prev_lot_cd = None
    eqp.prev_temp = None
    before = sim.stats["sametool_setup_count"]

    reward = sim._same_setup_reward(eqp, "PPKY", "OP2", 1, "LOTA", "T1")

    assert reward == 0.0
    assert sim.stats["sametool_setup_count"] == before
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `pytest tests/test_sametool_setup.py -v`
Expected: FAIL — `sim.stats["sametool_setup_count"]`에서 `KeyError`, 그리고
`_same_setup_reward()` 호출이 `TypeError: takes 4 positional arguments but 6 were given`

- [ ] **Step 3: `simulator.py` 수정**

`simulator.py:294-300`(`self.stats = {` 블록)을 아래로 교체:

```python
        self.stats = {
            "idle_total":    0,
            "oper_switches": 0,
            "prod_switches": 0,
            "conversions":   0,
            "sametool_setup_count": 0,
            "completed_qty": {},   # {(prod, oper): qty} — 배정 시점 증가(완료+처리중 포함)
        }
```

`simulator.py:1691-1712`(`_same_setup_reward` 전체)를 아래로 교체:

```python
    def _same_setup_reward(
        self, eqp: Equipment, ppk: str, oper_id: str, wf_qty: int,
        lot_cd: str, temp: str,
    ) -> float:
        """제품·공정이 '모두' 직전과 동일할 때만 연속 보너스.

        공정 전환·제품 전환을 따로 보상하지 않고, 둘 다 같은 경우(=전환 없음,
        동일 라우트 단계 유지)에만 +를 준다. switch 통계는 그대로 집계.
        해당 PPK 재공 고갈(투입 불가) 시에는 보너스를 죽인다.
        식: same_oper AND same_prod AND ppk_has_feasible_assignment → +w_same_setup, 아니면 0

        배치(BATCHID=LOT_CD/TEMP)는 그대로인데 PPK/OPER만 바뀌는 경우
        (=sametool_setup, TOOL 전환은 없음)는 별도로 소액 페널티를 주고
        stats["sametool_setup_count"]에 집계한다. 배치 자체가 바뀌는 실제
        TOOL 전환은 여기서 다루지 않는다(기존 stats["conversions"] 경로에서
        이미 카운트됨).
        """
        cfg = self._reward_cfg
        same_oper = (eqp.prev_oper == oper_id)
        same_prod = (eqp.prev_prod == ppk)
        if eqp.prev_oper is not None and not same_oper:
            eqp.oper_switches += 1
            self.stats["oper_switches"] += 1
        if eqp.prev_prod is not None and not same_prod:
            eqp.prod_switches += 1
            self.stats["prod_switches"] += 1

        if same_oper and same_prod:
            if not self._ppk_has_feasible_assignment(ppk):
                return 0.0
            return cfg.w_same_setup

        same_batch = (
            eqp.prev_lot_cd is not None
            and eqp.prev_lot_cd == lot_cd
            and (eqp.prev_temp or "") == (temp or "")
        )
        if same_batch:
            self.stats["sametool_setup_count"] += 1
            return -cfg.w_sametool_setup
        return 0.0
```

`simulator.py:2324` 호출부(`t = self._same_setup_reward(eqp, ppk, oper_id, wf_qty)`)를 아래로 교체
(바로 위 `2314`행에서 이미 `lot_cd, temp`가 계산돼 있으므로 그대로 전달):

```python
        t = self._same_setup_reward(eqp, ppk, oper_id, wf_qty, lot_cd, temp)
```

- [ ] **Step 4: 테스트 실행해 통과 확인**

Run: `pytest tests/test_sametool_setup.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: 기존 회귀 테스트도 통과하는지 확인** (시그니처 변경이 다른 호출부를 깨지 않았는지)

Run: `pytest tests/test_eqp_conv_down.py tests/test_discrete_conv_eligibility.py tests/test_wip_inflow_discrete_eligibility.py -v`
Expected: PASS (기존 테스트 전부 그대로 통과 — `_same_setup_reward`의 유일한 호출부는
`simulator.py:2324` 한 곳뿐이므로 다른 곳에서 깨질 여지 없음)

- [ ] **Step 6: 커밋**

```bash
git add simulation/simulator.py tests/test_sametool_setup.py
git commit -m "feat: distinguish sametool_setup from tool conversion in same_setup_reward"
```

---

### Task 3: `simulator.py` — history 스냅샷에 `sametool_setup` 노출

**Files:**
- Modify: `simulation/simulator.py:2538-2540` (`history.append({...})` 블록)
- Test: `tests/test_sametool_setup.py` (이어서 작성)

**Interfaces:**
- Consumes: `sim.stats["sametool_setup_count"]` (Task 2에서 추가)
- Produces: `sim.history[i]["sametool_setup"]: int` — 기존 `oper_sw`/`prod_sw`와 같은 위치·같은 패턴

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sametool_setup.py`에 이어서 추가:

```python
def test_history_snapshot_exposes_sametool_setup(sim):
    """history를 켠 시뮬에서 스냅샷마다 sametool_setup 키가 stats와 일치해야 한다."""
    sim._record_history = True
    eqp_id = sim.current_idle_eqp()
    assert eqp_id is not None
    lots = sim.available_lots(eqp_id)
    assert lots
    sim.assign_lot(eqp_id, lots[0]["lot_id"])

    assert sim.history, "assign_lot 이후 history가 최소 1건 기록돼야 함"
    last = sim.history[-1]
    assert "sametool_setup" in last
    assert last["sametool_setup"] == sim.stats["sametool_setup_count"]
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `pytest tests/test_sametool_setup.py::test_history_snapshot_exposes_sametool_setup -v`
Expected: FAIL — `KeyError: 'sametool_setup'` (또는 `AssertionError`, 키가 아직 없으므로)

- [ ] **Step 3: `simulator.py` 수정**

`simulator.py:2538-2540`:

```python
            "oper_sw":    self.stats["oper_switches"],
            "prod_sw":    self.stats["prod_switches"],
```

를 아래로 교체:

```python
            "oper_sw":    self.stats["oper_switches"],
            "prod_sw":    self.stats["prod_switches"],
            "sametool_setup": self.stats["sametool_setup_count"],
```

- [ ] **Step 4: 테스트 실행해 통과 확인**

Run: `pytest tests/test_sametool_setup.py::test_history_snapshot_exposes_sametool_setup -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add simulation/simulator.py tests/test_sametool_setup.py
git commit -m "feat: expose sametool_setup count in simulator history snapshots"
```

---

### Task 4: `agent/kpi_eval.py` — KPI 리포트에 `sametool_setup` 노출

**Files:**
- Modify: `agent/kpi_eval.py:39-57` (`KPI` dataclass), `agent/kpi_eval.py:60-93` (`KPIResult`), `agent/kpi_eval.py:96-107` (`_episode_kpi`)
- Test: `tests/test_sametool_setup.py` (이어서 작성)

**Interfaces:**
- Consumes: `sim.stats["sametool_setup_count"]` (Task 2), `KPI(sametool_setup: int)`
- Produces: `KPI.as_dict()["sametool_setup"]`, `KPIResult.sametool_setup: int` (property), `KPIResult.as_dict()["sametool_setup"]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sametool_setup.py`에 이어서 추가(파일 상단에 import 추가):

```python
from agent.kpi_eval import KPI, KPIResult


def test_kpi_as_dict_includes_sametool_setup():
    k = KPI(produced=5, producible=10, conversions=2, sametool_setup=3, reward=1.0, steps=5)
    assert k.as_dict()["sametool_setup"] == 3


def test_kpi_result_aggregates_sametool_setup():
    k1 = KPI(produced=5, producible=10, conversions=2, sametool_setup=3, reward=1.0, steps=5)
    k2 = KPI(produced=4, producible=10, conversions=1, sametool_setup=1, reward=0.5, steps=4)
    result = KPIResult(per_dataset=[k1, k2])
    assert result.sametool_setup == 4
    assert result.as_dict()["sametool_setup"] == 4
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `pytest tests/test_sametool_setup.py::test_kpi_as_dict_includes_sametool_setup tests/test_sametool_setup.py::test_kpi_result_aggregates_sametool_setup -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'sametool_setup'`

- [ ] **Step 3: `agent/kpi_eval.py` 수정**

`agent/kpi_eval.py:39-57`(`KPI` dataclass 전체)를 아래로 교체:

```python
@dataclass
class KPI:
    produced: int = 0
    producible: int = 0
    conversions: int = 0
    sametool_setup: int = 0
    reward: float = 0.0
    steps: int = 0

    def score(self, conv_weight: float = 1.0) -> float:
        return float(self.produced - conv_weight * self.conversions)

    def as_dict(self, conv_weight: float = 1.0) -> dict:
        return {
            "produced": self.produced,
            "producible": self.producible,
            "conversions": self.conversions,
            "sametool_setup": self.sametool_setup,
            "reward": round(float(self.reward), 3),
            "score": round(self.score(conv_weight), 3),
        }
```

`agent/kpi_eval.py:60-93`(`KPIResult` 클래스)를 아래로 교체:

```python
@dataclass
class KPIResult:
    """여러 데이터셋 합산 KPI."""

    per_dataset: List[KPI] = field(default_factory=list)

    @property
    def produced(self) -> int:
        return sum(k.produced for k in self.per_dataset)

    @property
    def producible(self) -> int:
        return sum(k.producible for k in self.per_dataset)

    @property
    def conversions(self) -> int:
        return sum(k.conversions for k in self.per_dataset)

    @property
    def sametool_setup(self) -> int:
        return sum(k.sametool_setup for k in self.per_dataset)

    @property
    def reward(self) -> float:
        return float(sum(k.reward for k in self.per_dataset))

    def score(self, conv_weight: float = 1.0) -> float:
        return float(self.produced - conv_weight * self.conversions)

    def as_dict(self, conv_weight: float = 1.0) -> dict:
        return {
            "produced": self.produced,
            "producible": self.producible,
            "conversions": self.conversions,
            "sametool_setup": self.sametool_setup,
            "reward": round(self.reward, 3),
            "score": round(self.score(conv_weight), 3),
            "prod_pct": round(100 * self.produced / max(self.producible, 1), 1),
        }
```

`agent/kpi_eval.py:96-107`(`_episode_kpi` 함수)를 아래로 교체:

```python
def _episode_kpi(env, produced_from_schedule: bool = True) -> KPI:
    sim = env.sim
    sim_end = sim.sim_end
    if produced_from_schedule:
        produced = sum(1 for r in sim.schedule if r.get("END_TM", 0) <= sim_end)
    else:
        produced = int(sum(sim.stats["completed_qty"].values()))
    return KPI(
        produced=produced,
        producible=env.producible_carriers(),
        conversions=int(sim.stats.get("conversions", 0)),
        sametool_setup=int(sim.stats.get("sametool_setup_count", 0)),
    )
```

- [ ] **Step 4: 테스트 실행해 통과 확인**

Run: `pytest tests/test_sametool_setup.py -v`
Expected: PASS (전체, 이번 태스크에서 총 9개 테스트 누적)

- [ ] **Step 5: kpi_eval 관련 기존 테스트도 통과하는지 확인**

Run: `pytest tests/test_kpi_eval.py tests/test_experts.py tests/test_bc_dagger.py tests/test_bc_pretrain.py -v`
Expected: PASS (KPI에 필드가 추가됐을 뿐 기존 필드/동작은 그대로이므로 영향 없음)

- [ ] **Step 6: 커밋**

```bash
git add agent/kpi_eval.py tests/test_sametool_setup.py
git commit -m "feat: surface sametool_setup count in KPI reporting"
```

---

## Self-Review Notes (완료 후 확인용)

- **스펙 커버리지**: 설계서 6.5절(구현 지점 3곳: `_same_setup_reward`/stats/호출부)은 Task 2, 8절의
  history 노출 항목은 Task 3, 9절의 "리포트 가시성" 요구는 Task 4가 커버한다. 5·6절(알고리즘 A/B)은
  설계서 자체가 "분석 프레임워크로만 남긴다"고 명시했으므로 신규 프로덕션 코드 태스크가 없는 것이
  정상이다.
- **타입 일관성**: `_same_setup_reward`의 새 시그니처 `(self, eqp, ppk, oper_id, wf_qty, lot_cd, temp)`는
  Task 2에서 정의되고 Task 2의 호출부 수정에서만 쓰인다(다른 호출부 없음, Step 5의 회귀 테스트로 확인).
  `KPI(sametool_setup: int)`는 Task 4에서 정의되고 같은 태스크의 `_episode_kpi`/`KPIResult`가 그대로
  사용한다.
- **하위 호환**: 기존 필드(`w_same_setup`, `stats["conversions"]`, `KPI.conversions` 등)는 이름·의미
  모두 변경하지 않았다. 새 필드는 전부 기본값이 있어 기존 호출부가 깨지지 않는다.
