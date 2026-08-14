# sametool_setup 기반 장비 전용화(Dedication) 설계서

- 작성일: 2026-08-14
- 대상 코드: `config.py`, `simulation/simulator.py`
- 관련 이전 시도: `allocation/eqp_allocator.py` (PR #277, 로직 복잡도로 인해 #278에서 revert됨)

## 1. 배경 및 목표

현재 시뮬레이터는 EQP가 idle이 될 때마다 어느 (PPK, OPER)를 투입할지 결정한다. 실제 운영 결과,
**TOOL 전환**과 **PPK/OPER 전환**이 지나치게 자주 발생하는 문제가 있었다. `w_same_setup` 리워드로
"직전과 완전히 동일한 경우"에만 보너스를 줘봤지만 효과가 부족했다(`simulator.py:1691` `_same_setup_reward`).

이번 설계의 목표는 **PPK/OPER별로 "지금부터 남은 시간 동안 몇 대 정도를 계속 붙여둬야 계획을 맞추는가"
(적정 장비댓수)를 계산**하고, 이를 이용해 장비가 idle이 될 때마다 "계속 같은 작업을 유지할지 / 전환할지"를
판단하는 기준을 만드는 것이다. 전환이 필요할 때도 **가능하면 TOOL 전환(1시간 setup)까지는 피하고,
같은 TOOL 내에서 PPK/OPER만 바뀌는(=`sametool_setup`) 선에서 흡수**하도록 우선순위를 둔다.

**설계 원칙(전작 revert 사유 반영)**: #277은 사전 배치(pre-allocation) 전체를 RULE_TIMEKEY마다 한 번에
계산해 DB에 저장하고, PPK 우선순위·전용/공용 모델 분리·EQP_ID 단위 확정까지 처리하다 복잡도가 커져
유지보수가 어려워졌다. 이번 설계는:
- **사전 계산/영속화 없음**: 매 idle 결정 시점에 그때의 실제 상태로 즉시 계산 (미래 예측 로직 없음)
- **하드 마스킹 없음**: 목표치는 우선순위 정렬에만 쓰는 소프트 신호. 후보가 하나뿐이면 그대로 진행되어
  계획 미달 리스크가 없음
- **공식 하나로 설명 가능**: PPK 간 조정, 전용/공용 모델 분리, 라운드로빈 같은 별도 규칙 없이,
  기존 함수(`_achievable_qty`, `_bucket_projected_cover`)를 그대로 재사용

## 2. 범위

| 대상 | 적용 여부 |
|---|---|
| 휴리스틱/dedication 배정 경로 (idle EQP → 다음 버킷 선택) | O (이번 설계의 본체) |
| RL 액션 마스킹 | X (건드리지 않음. #277이 이 경로에서 복잡해졌던 지점) |
| RL 리워드 shaping | O (`_same_setup_reward` 확장, 간접 신호로만 사용) |

## 3. 핵심 개념 정의

| 용어 | 정의 |
|---|---|
| **TOOL / BATCHID** | 기존 `(LOT_CD, TEMP)` 조합과 **동일 개념**(신규 마스터 축 아님). 장비가 지금 세팅된 레시피 |
| **TOOL 전환** | 장비의 현재 BATCHID ≠ 다음 작업이 요구하는 BATCHID. 기존 `conversion_minutes`(기본 60분)가 그대로 적용됨. 변경 없음 |
| **sametool_setup** | BATCHID는 동일(=TOOL 전환 없음)하지만 (PPK, OPER)가 직전과 다름. **시간 비용은 0**이나 카운트로 집계해 최소화 대상으로 삼는 신규 개념 |
| **target_eqp_count(ppk, oper)** | 리포트/설명용 지표. "이 (PPK,OPER)를 지금부터 끝까지 처리하려면 총 몇 대가 필요한가" |
| **gap(ppk, oper, eqp)** | 판단용 핵심값. "이 장비를 제외한, 이미 배치된 다른 장비들만으로 부족한 양" |
| **grade** | gap을 "장비 몇 대분 부족한가"로 정수 환산해 0~GRADE_MAX로 등급화한 값. 실제 STAY/SWITCH 판단은 이 grade로 함 |

## 4. 데이터 입력 매핑

사용자가 원래 제시한 5개 입력은 기존 시뮬레이터 데이터로 아래와 같이 매핑된다(신규 데이터 소스 불필요):

| 원 요구사항 | 기존 데이터 소스 |
|---|---|
| ① PPK별 계획량 | `plan_meta[(ppk,oper)]["d0_plan_qty"]` (D0_PLAN_QTY) |
| ② PPK/OPER/SEQ 공정 수순 | `flow` (OPER_SEQ), `_flow_prev()` |
| ③ PPK/OPER 재공 | `_wip_wafers(ppk, oper)`, WIP pool |
| ④ 현재 PPK/OPER 장비 수·시간당 생산량 | `eqp.prev_prod/prev_oper` 매칭 장비 수, `abstract_arrange`의 ST(EQP_MODEL별) |
| ⑤ BATCHID/PPK/OPER 마스터 | `batch_info`의 `(PPK,OPER) → (LOT_CD,TEMP)` 맵 (기존 `_lot_cd_temp()`가 참조하는 것과 동일 소스) |

## 5. 알고리즘 A — 적정 장비댓수(target_eqp_count) 산출

```
remaining_target(ppk, oper) = achievable_qty(ppk, oper) - completed_qty(ppk, oper)
avg_ST(ppk, oper) = abstract_arrange 상 이 (ppk,oper) 처리 가능 모델들의 ST 평균
                     (기존 _carrier_takt_minutes의 spw 계산과 동일 방식)
T_avail = soft_cutoff - current_time
capacity_per_eqp = T_avail / avg_ST
target_eqp_count(ppk, oper) = ceil(remaining_target / capacity_per_eqp)   # 0 이하면 0
```

- `remaining_target <= 0`(이미 달성) 또는 처리 가능 모델 없음 → `target_eqp_count = 0`
- `achievable_qty`가 이미 "상류 공정에서 도달 가능한 재공"까지 반영하므로(`simulator.py:1580`),
  재공 부족 상황은 별도 처리 없이 자동으로 `remaining_target`이 줄어들며 반영됨
- 이 값은 **리포트/테스트셋 표시용**이며, 실제 STAY/SWITCH 판단(알고리즘 B)에는 아래 gap 방식을 사용

## 6. 알고리즘 B — gap/grade 기반 STAY/SWITCH 판단 (핵심 판단 기능)

### 6.1 이미 배치된 장비 반영 (불필요 선점 방지)

장비가 idle이 되어 후보 버킷을 평가할 때, "내가 안 가도 이미 그 버킷에 배치된 **다른** 장비들이
끝까지 돌면 얼마나 처리되는가"를 먼저 뺀다. 기존 `_bucket_projected_cover(ppk, oper, exclude_eqp)`
(`simulator.py:1624`)를 그대로 재사용한다.

```
covered_by_others(ppk, oper, eqp) = _bucket_projected_cover(ppk, oper, exclude_eqp=eqp)
gap(ppk, oper, eqp) = remaining_target(ppk, oper) - covered_by_others(ppk, oper, eqp)
```

- `gap <= 0`: 이미 배치된 다른 장비만으로 충분 → 이 장비가 들어갈 필요 없음 (불필요 선점 방지)
- `gap > 0`: 아직 부족 → 이 장비가 들어가면 실질적으로 도움됨

이 계산은 **매 idle 결정 시점마다 새로 수행**된다(캐싱/사전계산 없음). 시뮬레이터는 idle 장비를
`_pick_next_idle_eqp()`(`simulator.py:903`)로 **한 번에 하나씩** 순차 처리하므로, 한 장비의 결정이
`prev_prod/prev_oper`에 즉시 반영된 뒤 다음 장비가 판단한다 — 여러 장비가 동시에 같은 버킷에
몰리는 동시성 문제가 구조적으로 없다. 상류 공정 유입량 변화 역시 별도로 예측하지 않고,
다음 idle 시점에 `achievable_qty`를 다시 계산하는 것만으로 자동 반영된다(DES가 시간을 진행시키며
상태를 바꿔주므로 이 알고리즘은 그 시점의 스냅샷에만 반응하면 됨).

### 6.2 gap의 등급화 (grade)

gap을 그대로 내림차순 정렬하면 소수점 단위 변동에도 순위가 뒤집혀 매 스텝 다른 버킷을 고르는
진동(oscillation)이 발생할 수 있다. gap을 "장비 대수" 단위로 환산해 정수 등급화한다.

```
capacity_per_eqp(ppk, oper, eqp) = T_avail / ST(eqp가 이 ppk,oper 처리 시 ST)
needed_more(ppk, oper, eqp) = gap > 0 ? ceil(gap / capacity_per_eqp) : 0
grade = clamp(needed_more, 0, GRADE_MAX)     # GRADE_MAX 기본값 3
```

| grade | 의미 |
|---|---|
| 0 | 이미 배치된 장비들만으로 충분 (이 버킷에 갈 필요 없음) |
| 1 | 대략 장비 1대분 부족 |
| 2 | 장비 2대분 부족 |
| 3 (=GRADE_MAX) | 장비 3대 이상 크게 부족 — 매우 시급 |

### 6.3 우선순위 계층(Tier) 판단

```python
def decide_next_bucket(eqp, candidates):
    my_batch = current_batch(eqp)   # eqp의 현재 BATCHID (없으면 None)
    for bucket in candidates:
        bucket.grade = compute_grade(bucket, exclude_eqp=eqp)   # 6.1~6.2
        bucket.gap = ...                                         # tie-break용
        bucket.same_ppk_oper = (bucket.ppk == eqp.prev_prod and bucket.oper == eqp.prev_oper)
        bucket.same_batch = (batch_of(bucket) == my_batch)

    def best(pool):
        return max(pool, key=lambda b: (b.grade, b.gap))   # grade 우선, 동률만 gap으로 tie-break

    # Tier 1: 직전과 완전 동일(PPK/OPER 그대로) + 아직 필요(grade>=1) → 전환 비용 0, 최우선 유지
    tier1 = [b for b in candidates if b.same_ppk_oper and b.grade >= 1]
    if tier1: return best(tier1)

    # Tier 2: 같은 BATCH(TOOL 전환 없음)지만 PPK/OPER는 다름(=sametool_setup 발생) + 필요(grade>=1)
    tier2 = [b for b in candidates if b.same_batch and b.grade >= 1]
    if tier2: return best(tier2)

    # Tier 3: 다른 BATCH(TOOL 전환 필요) + 필요(grade>=1) 중 가장 급한 것 → 불가피한 전환만 수행
    tier3 = [b for b in candidates if b.grade >= 1]
    if tier3: return best(tier3)

    # Tier 4: 모든 후보가 이미 충분(grade=0) → 가장 덜 과잉인 버킷 fallback
    return best(candidates)
```

- **grade가 다르면 반드시 grade가 승리**하고, 같은 grade 안에서만 `gap`으로 미세 정렬한다. grade
  경계를 넘을 때만 전환이 트리거되고, 같은 등급 내 미세 변동으로는 전환이 발생하지 않는다.
- 하드 마스킹이 아니라 **후보 정렬**이다. 후보가 하나뿐이면 grade와 무관하게 그것을 선택해
  계획 미달을 막는다(Tier 1~3에 걸리지 않아도 Tier 4 fallback이 항상 하나를 반환).

### 6.4 카운터 판정

```
tool_conversion = (eqp.batch is not None) and (선택된 버킷의 batch != eqp.batch)
sametool_setup   = (not tool_conversion) and (eqp.batch is not None)
                    and (선택된 버킷의 (ppk,oper) != (eqp.prev_prod, eqp.prev_oper))
```

## 7. Config 확장 (`config.py`)

```python
# EnvConfig
eqp_dedication_enabled: bool = True   # 기능 on/off. False면 기존 동작 그대로(하위호환)
dedication_grade_max: int = 3         # 등급 상한(GRADE_MAX)

# RewardConfig
w_sametool_setup: float = 0.5         # sametool_setup 발생 시 페널티(소액). 기존 w_same_setup과 별개 축
```

`w_same_setup`(직전과 완전 동일할 때 보너스, 기존 유지)과 `w_sametool_setup`(같은 배치·다른
PPK/OPER 전환 페널티, 신규)은 서로 다른 축이다. 기존 필드/동작은 변경하지 않는다.

## 8. 통합 지점

| 위치 | 내용 |
|---|---|
| `simulator.py` 신규 함수 | `_dedication_grade(ppk, oper, eqp) -> int` (6.1~6.2 구현) |
| `simulator.py` 신규 함수 | `_decide_next_bucket(eqp, candidates) -> bucket` (6.3 구현) |
| `simulator.py` 적용 지점 | 휴리스틱/dedication 배정 경로에서 버킷 후보 정렬 시 사용 |
| `simulator.py::_same_setup_reward` 확장 | `same_batch and not (same_oper and same_prod)` 케이스에 `w_sametool_setup` 페널티 추가 (RL 간접 신호) |
| `simulator.py::stats` 신규 카운터 | `stats["sametool_setup_count"]`, `stats["tool_conversion_count"]`(기존 conversion 카운트 재사용 가능 여부 확인 후 정합) |
| `config.py::EnvConfig/RewardConfig` | 7절 필드 추가 |

`eqp_dedication_enabled=False`면 새 함수들이 호출되지 않고 기존 배정 로직이 그대로 동작한다
(카운터 계측은 계속 켜둘 수 있음 — 도입 전후 비교용).

## 9. 지표 정의

| 지표 | 정의 |
|---|---|
| `tool_conversion_count` | BATCHID 변경 발생 횟수 (기존 conversion 이벤트) |
| `sametool_setup_count` | BATCHID 동일·(PPK,OPER) 변경 발생 횟수 (신규) |
| `target_eqp_count(ppk,oper)` | 시점별 스냅샷, 시계열 로깅 대상 |
| `plan_hit_rate` | 기존 `completed_qty / D0_PLAN_QTY` |

## 10. 테스트셋 (50개)

5개 그룹 × 10개. 참조 계산기 `docs/superpowers/specs/gen_sametool_setup_scenarios.py`가 본
설계서의 알고리즘 A/B를 그대로 구현해 아래 표의 값을 산출한다(수치는 수기 추정이 아니라 스크립트
실행 결과). 실제 구현 시 `pytest`로 옮겨 회귀 테스트화한다.

| 그룹 | 검증 목적 | 케이스 # |
|---|---|---|
| G1 - 기본 단일 버킷 | `target_eqp_count`/`grade` 계산 정확성 (기배치 수·ST·remain 변화) | 1~10 |
| G2 - 동일 배치 다중 PPK/OPER | 다른 장비가 이미 커버 시 grade=0→Tier2 전환, `sametool_setup` 카운트 | 11~20 |
| G3 - 배치 전환 필요 | 같은 배치로 해결 가능하면 Tier1 유지, 불가피할 때만 Tier3(TOOL 전환) | 21~30 |
| G4 - 상류 유입 변화 | `remaining_target`/`T_avail` 변화에 따라 `target_eqp_count`가 동적으로 재계산됨 | 31~40 |
| G5 - grade 경계/동률 | grade 경계(대수 환산 임계값)에서만 등급이 바뀌고, 등급 내 미세 변동에는 안정적(진동 없음) | 41~50 |

### 10.1 전체 결과표

| # | 그룹 | 입력 요약 | grade | gap | 선택 버킷(Tier) | TOOL전환 | sametool_setup | 비고 |
|---|---|---|---|---|---|---|---|---|
| 1 | G1-기본 | remain=50, T=480, ST=8, 기배치=0대 | 1 | 50.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=1 |
| 2 | G1-기본 | remain=90, T=480, ST=10, 기배치=1대 | 1 | 42.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=2 |
| 3 | G1-기본 | remain=130, T=480, ST=12, 기배치=2대 | 2 | 50.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=4 |
| 4 | G1-기본 | remain=170, T=480, ST=14, 기배치=0대 | 3 | 170.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=5 |
| 5 | G1-기본 | remain=210, T=480, ST=8, 기배치=1대 | 3 | 150.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=4 |
| 6 | G1-기본 | remain=250, T=480, ST=10, 기배치=2대 | 3 | 154.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=6 |
| 7 | G1-기본 | remain=290, T=480, ST=12, 기배치=0대 | 3 | 290.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=8 |
| 8 | G1-기본 | remain=330, T=480, ST=14, 기배치=1대 | 3 | 295.71 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=10 |
| 9 | G1-기본 | remain=370, T=480, ST=8, 기배치=2대 | 3 | 250.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=7 |
| 10 | G1-기본 | remain=410, T=480, ST=10, 기배치=0대 | 3 | 410.0 | PPK1/OPER1 (Tier3) | - | - | target_eqp_count=9 |
| 11 | G2-동일배치 | OPER1 remain=20(E_other 커버:False), OPER2 remain=40, ST2=10 | 1 | 20.0 | PPK2/OPER1 (Tier1) | - | - | |
| 12 | G2-동일배치 | OPER1 remain=30(E_other 커버:True), OPER2 remain=52, ST2=15 | 3 | 52.0 | PPK2/OPER2 (Tier2) | - | O | |
| 13 | G2-동일배치 | OPER1 remain=20(E_other 커버:False), OPER2 remain=64, ST2=20 | 1 | 20.0 | PPK2/OPER1 (Tier1) | - | - | |
| 14 | G2-동일배치 | OPER1 remain=30(E_other 커버:True), OPER2 remain=76, ST2=10 | 3 | 76.0 | PPK2/OPER2 (Tier2) | - | O | |
| 15 | G2-동일배치 | OPER1 remain=20(E_other 커버:False), OPER2 remain=88, ST2=15 | 1 | 20.0 | PPK2/OPER1 (Tier1) | - | - | |
| 16 | G2-동일배치 | OPER1 remain=30(E_other 커버:True), OPER2 remain=100, ST2=20 | 3 | 100.0 | PPK2/OPER2 (Tier2) | - | O | |
| 17 | G2-동일배치 | OPER1 remain=20(E_other 커버:False), OPER2 remain=112, ST2=10 | 1 | 20.0 | PPK2/OPER1 (Tier1) | - | - | |
| 18 | G2-동일배치 | OPER1 remain=30(E_other 커버:True), OPER2 remain=124, ST2=15 | 3 | 124.0 | PPK2/OPER2 (Tier2) | - | O | |
| 19 | G2-동일배치 | OPER1 remain=20(E_other 커버:False), OPER2 remain=136, ST2=20 | 1 | 20.0 | PPK2/OPER1 (Tier1) | - | - | |
| 20 | G2-동일배치 | OPER1 remain=30(E_other 커버:True), OPER2 remain=148, ST2=10 | 3 | 148.0 | PPK2/OPER2 (Tier2) | - | O | |
| 21 | G3-배치전환 | 동일배치 remain=40(기배치0대), 타배치 remain=80 | 2 | 40.0 | PPK3/OPER1 (Tier1) | - | - | |
| 22 | G3-배치전환 | 동일배치 remain=80(기배치1대), 타배치 remain=100 | 3 | 60.0 | PPK3/OPER1 (Tier1) | - | - | |
| 23 | G3-배치전환 | 동일배치 remain=40(기배치2대), 타배치 remain=120 | 3 | 120.0 | PPK4/OPER2 (Tier3) | O | - | |
| 24 | G3-배치전환 | 동일배치 remain=80(기배치0대), 타배치 remain=140 | 3 | 80.0 | PPK3/OPER1 (Tier1) | - | - | |
| 25 | G3-배치전환 | 동일배치 remain=40(기배치1대), 타배치 remain=160 | 1 | 20.0 | PPK3/OPER1 (Tier1) | - | - | |
| 26 | G3-배치전환 | 동일배치 remain=80(기배치2대), 타배치 remain=180 | 2 | 40.0 | PPK3/OPER1 (Tier1) | - | - | |
| 27 | G3-배치전환 | 동일배치 remain=40(기배치0대), 타배치 remain=200 | 2 | 40.0 | PPK3/OPER1 (Tier1) | - | - | |
| 28 | G3-배치전환 | 동일배치 remain=80(기배치1대), 타배치 remain=220 | 3 | 60.0 | PPK3/OPER1 (Tier1) | - | - | |
| 29 | G3-배치전환 | 동일배치 remain=40(기배치2대), 타배치 remain=240 | 3 | 240.0 | PPK4/OPER2 (Tier3) | O | - | |
| 30 | G3-배치전환 | 동일배치 remain=80(기배치0대), 타배치 remain=260 | 3 | 80.0 | PPK3/OPER1 (Tier1) | - | - | |
| 31 | G4-유입변화 | remain 추이=[30, 120, 50], T_avail 추이=[360, 300, 210] -> target 추이=[1, 4, 3] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 32 | G4-유입변화 | remain 추이=[40, 130, 60], T_avail 추이=[360, 300, 210] -> target 추이=[2, 5, 3] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 33 | G4-유입변화 | remain 추이=[50, 140, 70], T_avail 추이=[360, 300, 210] -> target 추이=[2, 5, 4] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 34 | G4-유입변화 | remain 추이=[60, 150, 80], T_avail 추이=[360, 300, 210] -> target 추이=[2, 5, 4] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 35 | G4-유입변화 | remain 추이=[70, 160, 90], T_avail 추이=[360, 300, 210] -> target 추이=[2, 6, 5] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 36 | G4-유입변화 | remain 추이=[80, 170, 100], T_avail 추이=[360, 300, 210] -> target 추이=[3, 6, 5] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 37 | G4-유입변화 | remain 추이=[90, 180, 110], T_avail 추이=[360, 300, 210] -> target 추이=[3, 6, 6] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 38 | G4-유입변화 | remain 추이=[100, 190, 120], T_avail 추이=[360, 300, 210] -> target 추이=[3, 7, 6] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 39 | G4-유입변화 | remain 추이=[110, 200, 130], T_avail 추이=[360, 300, 210] -> target 추이=[4, 7, 7] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 40 | G4-유입변화 | remain 추이=[120, 210, 140], T_avail 추이=[360, 300, 210] -> target 추이=[4, 7, 7] | - | - | target_eqp_count 시계열 검증 | - | - | 상류 유입 반영 시 target 재계산됨 |
| 41 | G5-경계동률 | remain=15.0 (capacity/eqp=20.0 근방 미세변동 -5칸) | 1 | 15.0 | PPK6/OPER4 (Tier1) | - | - | |
| 42 | G5-경계동률 | remain=16.0 (capacity/eqp=20.0 근방 미세변동 -4칸) | 1 | 16.0 | PPK6/OPER4 (Tier1) | - | - | |
| 43 | G5-경계동률 | remain=17.0 (capacity/eqp=20.0 근방 미세변동 -3칸) | 1 | 17.0 | PPK6/OPER4 (Tier1) | - | - | |
| 44 | G5-경계동률 | remain=18.0 (capacity/eqp=20.0 근방 미세변동 -2칸) | 1 | 18.0 | PPK6/OPER4 (Tier1) | - | - | |
| 45 | G5-경계동률 | remain=19.0 (capacity/eqp=20.0 근방 미세변동 -1칸) | 1 | 19.0 | PPK6/OPER4 (Tier1) | - | - | |
| 46 | G5-경계동률 | remain=20.0 (capacity/eqp=20.0 근방 미세변동 +0칸) | 1 | 20.0 | PPK6/OPER4 (Tier1) | - | - | |
| 47 | G5-경계동률 | remain=21.0 (capacity/eqp=20.0 근방 미세변동 +1칸) | 2 | 21.0 | PPK6/OPER4 (Tier1) | - | - | |
| 48 | G5-경계동률 | remain=22.0 (capacity/eqp=20.0 근방 미세변동 +2칸) | 2 | 22.0 | PPK6/OPER4 (Tier1) | - | - | |
| 49 | G5-경계동률 | remain=23.0 (capacity/eqp=20.0 근방 미세변동 +3칸) | 2 | 23.0 | PPK6/OPER4 (Tier1) | - | - | |
| 50 | G5-경계동률 | remain=24.0 (capacity/eqp=20.0 근방 미세변동 +4칸) | 2 | 24.0 | PPK6/OPER4 (Tier1) | - | - | |

전체 재생성: `python docs/superpowers/specs/gen_sametool_setup_scenarios.py`

### 10.2 요약

- 이 50개 중 배치/전환 관련 열이 채워지는 G2/G3 20개 케이스 기준: **TOOL 전환 2회, sametool_setup
  5회** 발생. 나머지(G1/G4/G5)는 각각 다른 속성(수치 정확성/동적 재계산/등급 안정성)을 격리해서
  검증하므로 전환 카운트를 채우지 않음(단일 후보 또는 배치 정보 미부여).
- G2 결과(11~20행)는 "OPER1을 이미 다른 장비(E_other)가 충분히 커버할 때만 Tier2(sametool_setup)로
  넘어간다"는 6.1의 핵심 주장을 직접 검증한다: `E_other 커버:False`인 홀수 케이스는 전부 Tier1
  유지, `True`인 짝수 케이스는 전부 Tier2로 sametool_setup 발생.
- G3 결과(21~30행)는 "같은 배치 버킷이 기배치 장비로 이미 충분(예: 23·29행, 기배치 2대)하면 그때만
  불가피하게 Tier3(TOOL 전환)로 넘어간다"를 검증한다 — 같은 배치로 흡수 가능한 나머지 8개 케이스는
  전부 Tier1에서 해결되어 TOOL 전환이 발생하지 않는다.
- G5 결과(41~50행)는 `remain`이 15→20으로 6단계 변하는 동안 `grade=1`로 고정되다가, 21에서 처음
  `grade=2`로 바뀜을 보여준다 — **등급 경계를 넘을 때만 판단이 바뀌고, 그 안에서는 안정적**이라는
  6.2의 설계 의도가 실제로 성립함을 확인.

## 11. 한계 및 트레이드오프

- **Lookahead 없음**: 현재 스냅샷에 반응하는 greedy 방식이다. "곧 상류가 마를 것"을 미리 대비하지
  못하고, 실제로 그 시점이 와야 반영된다. 복잡도를 낮추기 위한 의도적 선택이다.
- **전역 최적 아님**: 각 장비가 독립적으로(순차적으로) 판단하므로 전체 최적 배분을 보장하지 않는다.
  같은 모델을 여러 (PPK,OPER)가 공유할 때 총 수요가 실제 보유 대수를 넘는 상황은 별도로 정규화하지
  않는다(설계 논의 중 "Approach B: 모델 총량 정규화"로 언급됨 — 필요 시 향후 확장 지점).
- **RL 경로는 리워드 shaping만**: 액션 마스킹은 적용하지 않으므로 RL이 이 신호를 무시할 수도 있다.
  효과가 부족하면 이후 별도 논의.

## 12. 향후 확장(Out of scope, 참고용)

- 모델 총량 정규화(Approach B): 같은 모델을 공유하는 버킷들의 grade 합이 실제 보유 대수를 넘으면
  비례배분으로 스케일다운. 이번 범위에는 포함하지 않음.
