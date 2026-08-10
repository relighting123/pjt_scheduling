# pjt_scheduling

반도체 **Scheduling** 문제를 강화학습(RL)과 휴리스틱으로 실행하는 프로젝트입니다.

`discrete_arrange` / `abstract_arrange` / `plan` / `flow` 입력을 전처리해 DES(Discrete Event Simulation)를 돌리고,
EQP가 idle이 될 때마다 **투입할 (PPK, OPER)** 를 선택합니다. LOT·EQP 세부 배정은 규칙/휴리스틱이 담당합니다.

---

## 프로젝트 목표

| 상황 | 원하는 동작 |
|------|-------------|
| 재공이 계획 대비 충분 | **takt time**에 맞춰 공정별로 꾸준히 생산 |
| 재공 부족·특정 공정 편중 | 몰린 공정에 장비 투입 → 후공정 재공 축적 → flow 밸런스 회복 |

RL 보상(`w_pacing`, `w_plan_hit`, `w_flow_balance`, `use_achievable_target`)은 위 목표를 반영하도록 설계되어 있습니다.
다만 RL action은 **(PPK, OPER) 버킷**만 선택하고, EQP·LOT은 simulator 규칙이 처리합니다.

---

## 현재 구조 요약

```
pjt_scheduling/
├── main.py                 # CLI: collect / train / test / infer / db-load / ui
├── config.py               # 경로, 환경 축, RL·reward 파라미터
│
├── agent/                  # RL(PPO) + 휴리스틱
├── env/scheduling_env.py   # Gymnasium Env
├── simulation/             # DES 엔진, reward, decision_log
├── data/
│   ├── loader/             # Oracle/JSON → env_data
│   ├── writer/             # 추론 결과 → output.json / SQL / DB 적재
│   ├── sql.example/        # 입력·출력 SQL 템플릿
│   └── dataset/            # {FAC}/{split}/{RULE_TIMEKEY}/input|output
├── inference/runner.py
├── api/ + frontend/
└── models/                 # PPO 체크포인트
```

---

## 데이터 모델

### 입력 레이어

| 레이어 | 단위 | 설명 |
|--------|------|------|
| `discrete_arrange` | `(EQP, LOT, OPER)` | 이 carrier가 이 EQP에 투입 가능하다는 선언(ST, WF_QTY 포함, 전건 자유배정 후보) |
| `abstract_arrange` | `(PPK, OPER, EQP_MODEL)` | **arrange** = 장비 재공 투입 가능 여부 템플릿(모델 단위, 더 성긴 단위) |
| Runtime WIP | `(PPK, OPER)` + LOT list | 현재/유입 재공, `oper_in_time` |

부가 입력: `flow`, `batch_info`(LOT_CD/TEMP), `tool_capacity`, `eqp_initial_state`, `split`, `conversion_group`,
`eqp_conv_plan`(외부 확정 EQP 전환 계획), `eqp_down`(EQP 다운타임), `eqp_queue_init`(이미 투입 확정된 EQP별 큐)

#### `discrete_arrange`는 "실적"이 아니라 투입 가능 관계(M:N)다

`discrete_arrange`의 `(EQP, LOT)` 행은 "이 EQP가 과거에 이 LOT/제품을 만든 적 있다"는
이력·실적 데이터가 **아니다**. "이 carrier가 이 EQP에 물리적으로 투입 가능하다"는 선언이며,
같은 LOT이 여러 EQP에 대해 동시에 선언될 수도 있는 **M:N 관계**다(한 carrier가 여러 대에
투입 가능, 한 EQP가 여러 carrier를 받을 수 있음). `abstract_arrange`가 더 성긴
(PPK, OPER, EQP_MODEL) 단위의 투입 가능 템플릿이라면, `discrete_arrange`는 그보다 세밀한
(EQP, carrier) 단위의 투입 가능 템플릿이라고 보면 된다 — 둘 다 "가능 여부" 선언이지 실적 기록이 아니다.

**discrete vs abstract, 언제 어느 쪽을 신뢰하는가 (`_lot_conv_discrete_eligible`)**

같은 PPK·같은 장비 모델이라도, 실제로는 LOT/EQP_ID 조합마다 투입 가능 여부가 다를 수 있다
(설비 점검·챔버 상태 등 모델 단위로는 잡히지 않는 실제 제약). 그래서 시뮬레이터는 상황에
따라 두 정보 중 하나를 골라 신뢰한다:

- **전환 불필요 + 현재 공정에 이미 존재하는(초기 스냅샷) 재공**: 지금 당장 확정적으로 알 수
  있는 상태이므로, `discrete_arrange`에 선언된 (EQP, LOT) 조합만 투입 가능한 것으로 본다.
  abstract(모델만 일치)만으로는 이 세밀한 제약을 대신할 수 없다.
- **전환이 필요하거나(EQP의 현재 셋업과 달라 전환 발생) / 이전 공정에서 아직 넘어오지 않은
  (유입 예정) 재공**: 둘 다 "지금은 확정할 수 없고 앞으로 벌어질 일"이라는 예측의 영역이다.
  전환 후 상태는 아직 discrete로 정밀 검증된 적 없는 새 셋업이고, 유입 재공은 아직 이
  공정에 도착하지도 않았기 때문에, 이때는 discrete를 참조하지 않고 PPK×모델 단위로
  추상화된 `abstract_arrange` 기준으로만 판단한다.
- **전환 후 원래 tool로 복귀**: 장비가 A→B로 전환했다가 다시 B→A로 돌아오면, 그 시점부터는
  A가 다시 "이미 확정된 현재 셋업"이 되므로 discrete 정보를 다시 사용한다(더 이상 예측이
  아니라 현재 상태이므로).

이 규칙은 `discrete_wait_enabled=True`(기본값)에서만 적용되고, `discrete_wait_enabled=False`면
이 구분 없이 항상 abstract 폴백을 허용한다.

따라서 입력 데이터가 실제로는 여러 EQP에 대해 M:N으로 투입 가능한 carrier를, 편의상 EQP
1대에 대해서만 1:1로 좁혀 선언해 두면(예: 벤치마크 데이터 생성기가 라운드로빈으로 EQP
하나씩만 배정), 그 데이터는 "전환 불필요 + 현재 공정 재공"에 해당하므로 시뮬레이터는 그
좁은 선언을 그대로 신뢰해 나머지 EQP에서는 그 LOT을 투입 불가로 취급한다 — 모델이 같아
실제로는 전부 처리 가능해야 하는 조합이라도, declared 관계가 좁으면 결과도 좁게 나온다.

#### 외부 확정 EQP 전환 계획 (`eqp_conv_plan.json`, 선택)

MES 등 외부에서 이미 확정된(현재 진행 중이거나 예정된) EQP 전환을 반영한다. 행: `EQP_ID`,
`FROM_LOT_CD`, `FROM_TEMP`, `TO_LOT_CD`, `TO_TEMP`, `START_TM`(RULE_TIMEKEY 형식).

- `START_TM`이 RULE_TIMEKEY 이전/동일이면 시뮬 시작과 동시에 즉시 전환이 개시된다.
- 아직 시작 전이면 해당 EQP는 그때까지 정상적으로 배정을 받다가, 시작 시각에 idle 상태이면
  전환으로 전이한다(진행 중이던 가공은 선점하지 않음 — 종료 후 다음 idle 시점에 적용).
- 전환 소요 시간은 `conversion_minutes`(기존 설정)를 그대로 사용한다.
- Gantt에는 기존 conversion과 동일하게 `CONV` 바로 표시되고(`conversion_plans`에 병합,
  `source: "SCHEDULED"`로 구분), `RTS_EQPCONVPLAN_INF` 출력에도 함께 반영된다.

#### EQP 다운타임 (`eqp_down.json`, 선택)

PM/개조 등으로 인한 EQP 다운 구간. 행: `EQP_ID`, `DOWN_START_TM`, `DOWN_END_TM`(선택,
RULE_TIMEKEY 형식).

- `DOWN_START_TM`이 RULE_TIMEKEY 이전/동일이면 시뮬 시작과 동시에 즉시 다운 처리된다.
- `DOWN_END_TM`이 없으면 무제한 다운으로, 해당 EQP는 이후 배정 대상에서 영구 제외된다.
- 다운 적용도 진행 중이던 가공을 선점하지 않는다(종료 후 다음 idle 시점에 적용).
- Gantt에는 `DOWN` 바로 표시된다(`down_windows`, UI 전용 — RTS 출력에는 포함되지 않음).

#### 이미 투입 확정된 큐 (`eqp_queue_init.json`, 선택)

AI 스케줄러가 배정을 결정하지 않는, 이미 투입이 확정된 EQP별 가공 큐(초기 상태)입니다.
`discrete_arrange`의 옛 `LOT_STAT_CD`(PROC/LOAD/SELE/RESV) 강제배정을 대체합니다 —
`discrete_arrange`는 이제 전건 자유배정(WAIT) 후보만 나타냅니다.

행: `EQP_ID`, `SEQ_NO`(같은 EQP 안에서의 처리 순서), `LOT_ID`, `CARRIER_ID`(선택),
`OPER_ID`, `START_TM`, `END_TM`(둘 다 RULE_TIMEKEY 형식), `PLAN_PROD_ATTR_VAL`, `WF_QTY`.

- ST로 소요시간을 역산하지 않고 입력된 `START_TM`/`END_TM`을 그대로 스케줄에 남깁니다.
- 같은 EQP의 여러 건은 `SEQ_NO` 오름차순으로 처리되며, 앞 건이 끝나면(`_on_process_end`)
  곧바로 다음 건이 이어서 활성화됩니다. `START_TM`이 아직 미래면 그 시각까지는 EQP가
  자유배정(WAIT LOT) 후보로 열려 있습니다 — 두 항목 사이 시간 간격은 AI가 채울 수 있습니다.
- tool 동시성/전환그룹/전환횟수/WIP 가용성 등 자유 스케줄링 제약을 전부 우회하고 항상
  그대로 반영됩니다(물리적으로 이미 그 상태로 진행 중인 확정 사실이므로).
- 이 LOT들은 자유배정 후보(`available_lots`)로 전혀 노출되지 않습니다.
- `discrete_arrange`에 전혀 등장하지 않는(자유배정 후보가 전혀 없을 만큼 이미 확정 큐로
  꽉 찬) EQP도 `eqp_queue_init`만으로 EQP 목록에 등록됩니다. 단 `EQP_MODEL_CD`는
  `discrete_arrange`에서만 나오므로, 그런 EQP는 tool 동시성 추적 대상에서는 제외됩니다.
- `CONFIG.env.discrete_wait_enabled`와 무관하게 항상 그대로 반영됩니다.

##### 대체 입력 경로: `discrete_arrange`의 `LOT_STAT_CD` (선택)

`eqp_queue_init.json`을 별도로 만들기 어려운 소스를 위한 경로입니다. `discrete_arrange`
행에 `LOT_STAT_CD`(`PROC`/`LOAD`/`SELE`/`RESV`, 비어 있으면 기본값 `WAIT`)를 채우면, 그
carrier는 AI가 결정하지 않고 같은 `EQP_ID`에 `PROC → LOAD → RESV → SELE` 순서(동순위는
입력 순서)로 강제 배정됩니다.

- 내부적으로 `eqp_queue_init`과 동일한 큐 포맷으로 변환되어 합류합니다 — 실행 경로·제약
  우회 동작(tool 동시성/전환그룹/WIP 가용성 무시)은 위 `eqp_queue_init` 항목과 같습니다.
- 소요시간은 `START_TM`/`END_TM`을 직접 받는 `eqp_queue_init`과 달리 `ST × WF_QTY`로
  역산합니다. 같은 EQP에 `eqp_queue_init` 큐가 이미 있으면 그 큐의 마지막 `END_TM`부터
  이어서 순차 배정됩니다(이중 예약 방지).
- 이 carrier들은 `discrete_arrange`의 자유배정(WAIT) 후보(`eqp_lot_map`)에서 제외됩니다 —
  `LOT_STAT_CD`가 없거나 `WAIT`인 행만 자유배정 후보로 남습니다.
- 한 EQP에 `eqp_queue_init`과 `LOT_STAT_CD` 강제 행이 동시에 있으면 함께 병합되어
  하나의 큐로 처리됩니다.

#### 전환 그룹 제약 (`config.CONVERSION_GROUPS`, 선택)

같은 그룹 안의 `(LOT_CD, TEMP)`로만 전환을 허용하고 **다른 그룹으로의 전환은 배정 후보에서 제외**합니다(행동 공간 축소 → 문제 단순화). `config.py`의 `CONVERSION_GROUPS` 딕셔너리에 `FAC_ID`별로 설정하며, 해당 FAC_ID 항목이 없으면 제약은 비활성(기존 동작) — train/test/infer 등 split·기간과 무관하게 fac 전체에 공통 적용됩니다.

```python
# config.py
CONVERSION_GROUPS = {
    "FAC001": [
        {"GROUP_ID": "G1", "LOT_CD": "LC_A", "TEMP": "T600"},
        {"GROUP_ID": "G1", "LOT_CD": "LC_B", "TEMP": "T600"},
        {"GROUP_ID": "G2", "LOT_CD": "LC_C", "TEMP": "T600"},
    ],
}
```

규칙:
- 동일 셋업(전환 없음)·첫 배정(EQP 셋업 미지정)은 항상 허용
- `from`·`to`가 **둘 다 그룹에 속할 때만**, 두 그룹이 다르면 차단
- 그룹에 없는 `(LOT_CD, TEMP)`는 제약 없음(미지정 = 와일드카드)

> 효과: EQP가 처음 잡은 그룹에 고정되어 그룹 내에서만 전환합니다. 도달 불가능한 제품이 생길 수 있으나(생산량 소폭 감소), 불필요한 교차 그룹 전환이 사라집니다.

### arrange vs discrete (런타임)

- **후보 생성**: EQP별 `abstract_arrange` × WIP LOT을 펼침
- **전환 불필요**(EQP에 이미 장착된 LOT_CD/TEMP와 목표가 같음): 이 EQP×carrier 조합이
  `proc_time_matrix`(discrete)에 있어야만 배정 가능 — 없으면 배정 불가(같은
  PPK/OPER라도 carrier별로 실제 가능한 EQP가 다를 수 있어 abstract 모델 매칭만으로는
  불충분). 있으면 discrete ST 사용 (`ABSTRACT=False`).
- **전환 필요**(LOT_CD/TEMP 중 하나라도 다름): 기존 셋업 기준 discrete 조합은 전환 후
  더는 유효하지 않으므로 discrete를 참조하지 않고 arrange(모델 평균) ST만 사용
  (`ABSTRACT=True`).
- **첫 배정**(EQP가 아직 한 번도 셋업된 적 없음, `prev_lot_cd` 미지정)은 비교 대상이
  없어 위 제약에서 제외되고 항상 abstract로 허용됩니다.
- `discrete_wait_enabled=False`는 이 구분과 무관하게 항상 abstract 폴백을
  허용합니다(기존 동작 유지). 이미 투입 확정된 큐(`eqp_queue_init`)는 이 판단 자체를
  타지 않고 항상 그대로 반영됩니다.
- 위 자격 조건을 만족하는 carrier가 하나도 없는 `(PPK,OPER)` 버킷은 해당 EQP의
  액션 마스크(`get_feasible_ppk_oper`)에서 제외됩니다.

Earliest-ST는 idle EQP 전체 × feasible LOT을 한 번에 비교해 예상 종료 시각이 가장 이른 **EQP×carrier** 1건을 선택합니다.

---

## 알고리즘

| ID | 설명 |
|----|------|
| `rl` | MaskablePPO — 현재 idle EQP에서 `(PPK, OPER)` 선택 |
| `minprogress` | 계획 진행률·잔여 WIP 기준 버킷 선택 |
| `earliest_st` | EQP×carrier 전역 최소 `end_time + ST×qty` (split 이후) |

```bash
# API/UI 비교 또는 runner
run_inference(env_data, algorithm="earliest_st")
```

---

## 의사결정 흐름 (한 step)

1. simulator가 idle EQP 결정 시점 탐색
2. **RL/MinProgress**: 현재 EQP에서 `(PPK, OPER)` 선택 → `_auto_select_lot()`
3. **Earliest-ST**: `pick_earliest_st_assignment()`로 EQP×LOT 선정 → `assign_earliest_st_pending()`
4. conversion / tool cap / WIP 소비 / 이벤트 큐 갱신
5. `enable_wip_inflow=True`이면 공정 완료 시 `flow_next`로 후공정 WIP 유입

---

## 강화학습 구성

### Action / Observation

| 항목 | 값 |
|------|-----|
| Action | `MultiDiscrete([O×P, L])` = `(OPER/PPK bucket, 블록 크기 레벨)` |
| Mask | 현재 idle EQP feasible bucket ｜ 크기 레벨 |
| obs_dim (기본 env) | `6 + O×P×6 + O×P×K×5` = **936** (O=3, P=10, K=5) |
| obs_dim (`SchedulingRLEnv`) | 936 + **6**(결정 컨텍스트) = **942** |

**Bucket feature (po 6ch + pom 5ch)**: WIP 비율, urgency, achievable_ratio, projected_cover_ratio, starve_time_norm / ST, conversion·tool 가용, avoidable_frac, setup_changed 등.
prev/post takt·LOT_CD/TEMP 인코딩 채널은 제거됨 — takt는 정적 설비 수(`n_eqp_per_oper`) 기반이라 설비 공유·실시간 배정 상태를 반영 못 했고, LOT_CD/TEMP는 범주형 ID를 순서 있는 스칼라로 인코딩해 신호 품질이 낮았음 (전환 관련 정보는 pom_feats에 이미 더 정확히 포함).

**결정 컨텍스트 6채널** (`SchedulingRLEnv` 전용, 항상 활성):
결정 장비 유무 / 범용성(model_breadth) / 첫 셋업 여부 / **블록 진행 여부** /
**블록 잔여율** / idle 장비 비율.

핵심은 **블록 상태(3·4번 채널)** 다. `self._block`(어느 버킷을 몇 carrier 남기고
커밋 중인지)은 env 레이어에만 있고 시뮬레이터에는 없어서, "새 블록을 시작하는
스텝"과 "블록이 3장 남은 스텝"의 관측이 같았다 — 행동 의미는 전혀 다른데도
(블록 중이면 마스크가 같은 버킷을 강제하고 `size_level`은 무시된다). 크리틱
입장에서 남은 커밋 길이에 따라 기대 return이 크게 다른데 구분할 수가 없어
value 추정과 advantage가 같이 흐려진다.

> ⚠️ **호기(장비 고유 번호)는 관측에 넣지 않는다.**
> 한때 "같은 모델 설비가 여러 대면 대칭이라 구분이 안 된다"며
> `(장비 순번)/(장비 수)` 채널을 넣었으나 근거가 틀렸다. 결정은 **순차적**이라
> 앞 설비가 배정하면 WIP·cover가 바뀌어 다음 설비는 이미 다른 상태를 본다 —
> SYM_5x5 실제 롤아웃 30개 결정에서 '다른 설비인데 관측이 동일한' 쌍은 **0개**였다
> (동일하게 보였던 건 t=0 상태를 고정한 채 `_current_eqp`만 바꿔치기한 인위적
> 테스트에서였다). 반대로 순번 채널의 해악은 분명하다: ① 정책이 "0.6번 설비는
> 제품 3"을 외워서 설비가 다운·추가되면 인덱스가 밀려 매핑이 통째로 깨지고,
> ② 명목형 ID를 순서형 축에 얹어 없는 순서 관계를 주입한다(설비 수가 바뀌면
> 같은 호기의 값도 달라진다). 설비를 구분해야 한다면 식별자가 아니라
> **성질**(범용성·셋업 상태·블록 상태)로 구분한다.
> `tests/test_terminal_reward.py`가 이 두 가지를 회귀 테스트로 잡아둔다.

### Reward (`config.py` 기본값)

| 항목 | 가중치 | 역할 |
|------|--------|------|
| `w_same_setup` | 1.0 | 직전과 제품·공정 모두 동일할 때 연속 생산 보너스 |
| `w_conversion` | -10.0 | LOT_CD/TEMP 전환 |
| `w_avoidable_conversion` | -8.0 | 회피 가능한 전환(이미 같은 셋업의 다른 장비가 커버 가능) 추가 패널티 |
| `w_bulk_block_bonus` | 3.0 | (Bulk-Fill 전용) 같은 제품군을 큰 블록으로 커밋할수록 보상 |
| `w_dedication_misuse` | -4.0 | (Bulk-Fill 전용) 범용 장비가 더 전용적인 idle 장비 몫의 버킷을 잡으면 감점 |
| `w_redundant_cover` | -5.0 | (Bulk-Fill 전용) 이미 다른 장비가 충분히 커버 중인 버킷을 또 잡으면 감점 |
| `reward_clip` | ±20.0 | step reward clip (PPO 안정화) |
| `w_terminal_throughput` | 30.0 | **에피소드 종료 시** `완료 carrier / 생산가능 carrier` 비율 보상 |
| `w_terminal_conversion` | -1.0 | **에피소드 종료 시** 전환 총횟수 패널티 |
| `terminal_reward_clip` | ±60.0 | 종단 보상 clip |
| `w_plan_hit` / `w_pacing` / `w_flow_balance` / `w_idle_per_min` | 0.0 | cover 무시·전담 방해로 판단되어 제거됨(주석 참고) |

`use_achievable_target=True`: 재공이 부족하면 무리한 계획 추격을 막고, 선행 공정 투입 유도.

**종단(terminal) KPI 보상**: 위 step reward는 전부 대리지표라 "보상이 가장 높은
정책"이 "벤치마크 KPI가 가장 좋은 정책"이라는 보장이 없었다. `w_terminal_*`는
벤치마크가 실제로 재는 값(sim_end 안에 끝난 carrier 수, 전환 총횟수)을 에피소드
종료 시 그대로 보상으로 준다 → 최적화 대상과 평가 대상을 일치시킨다. 이 항은
**step clip 이후**에 더해지고(clip에 눌리면 의미가 없음), 신호가 초반 결정까지
역전파되도록 `RLConfig.gamma`를 0.997로 올렸다.

---

### 모방학습(BC) → 강화학습 파이프라인

"Dedication 시연으로 모방학습한 뒤 PPO로 개선한다"는 구조 자체는 맞지만, 초기
구현은 **RL을 돌릴수록 성적이 나빠지고 학습 곡선이 x축과 평행**한 문제가 있었다
(BENCH_SUITE 8종 실측: Dedication 점수 77.0 → RL 23.0). 원인은 하나가 아니라
아래 8가지가 겹친 것이고, 각각을 개별적으로 고쳤다.

| # | 원인 | 대응 | 위치 |
|---|------|------|------|
| ① | **BC 표본 부족** — 데이터셋당 결정적 시연 1 에피소드(수십 스텝)뿐 | ε 교란 후 **전문가 라벨 재부여**(DAgger식)로 상태분포 확장 | `agent/bc.py: collect_expert_dataset()` |
| ② | **크리틱 미학습** — value head가 무작위 → PPO 첫 업데이트의 advantage가 순수 잡음이 되어 복제 정책을 즉시 파괴 | 시연의 할인 return-to-go로 **value head도 동시 회귀** | `agent/bc.py: behavior_clone(value_coef)` |
| ③ | **결정적 붕괴** — full-batch NLL을 오래 돌리면 엔트로피≈0 → 탐색·학습 불가 | 미니배치 + 엔트로피 정규화 + **검증 NLL 조기종료** | `agent/bc.py: behavior_clone()` |
| ④ | **파국적 망각** — 워밍스타트만으로는 RL 잡음에 전문가 정책에서 이탈 | 롤아웃마다 BC 보조 그래디언트를 섞고 계수를 감쇠 | `agent/bc.py: ExpertAnchorCallback` |
| ⑤ | **보상 스케일** — step reward ±20 × 수백 스텝 → return 수백 규모, value loss 폭주, `explained_variance≈0` → advantage가 잡음 → **곡선이 평평** | `VecNormalize(norm_reward=True)` | `RLConfig.normalize_reward` |
| ⑥ | **업데이트 횟수 부족** — `n_steps=2048` × 데이터셋 8개 = 롤아웃 16,384 → 200k 예산에서 PPO 업데이트가 **12번** | `n_steps` 2048 → 512 | `RLConfig.n_steps` |
| ⑦ | **시연자가 전담 하나로 고정** — 정책의 출발점 상한이 전담 휴리스틱이었다. 전담이 늘 최선은 아니다(LOAD_stmix에서 Earliest-ST가 2배 점수) | 시나리오마다 후보 전문가를 굴려 KPI 최고를 시연자로 선택 | `agent/experts.py`, `RLConfig.bc_expert_candidates` |
| ⑧ | **시나리오 암기** — 8종만 co-train하니 그 8종에선 이기고 학습에 안 쓴 6종에선 졌다 | 에피소드마다 시나리오 풀에서 뽑는 도메인 랜덤화 | `RandomizedSchedulingRLEnv`, `RLConfig.dataset_sampling` |

추가로:

- **KPI 기반 best 모델 선택** (`RLConfig.kpi_eval_enabled`, 기본 `True`):
  `EvalCallback`은 *평균 보상*이 최고인 체크포인트를 남기는데, shaping 보상 최고점이
  벤치마크 KPI 최고점과 일치한다는 보장이 없다. 대신 eval 시점마다 실제 KPI
  (`생산량 − kpi_conversion_weight × 전환수`)로 채점해 best를 고른다
  (`agent/kpi_eval.py`). **워밍스타트 직후 성적도 후보에 포함**되므로, RL이 그보다
  나아지지 못하면 `restore_best`가 BC 시점 모델을 되돌린다 — 즉 **"RL 때문에
  나빠지는" 경우가 구조적으로 없다.**
- **Dedication 기준선 자동 측정** (`RLConfig.kpi_baseline_enabled`): 학습 시작 시
  같은 환경에서 `DedicationAgent` KPI를 재고, 매 eval마다 기준선 대비 격차를
  로그·차트에 남긴다. 최종 모델이 기준선에 못 미치면 경고한다.
- **학습/추론 환경 일치** (`RLConfig.align_train_env_with_inference`,
  `RLConfig.train_truncate_on_time`): 학습은
  simulator 기본값(`termination_mode="all_wip"`, `enable_wip_inflow=True`,
  `truncate_on_time=True`)으로,
  추론은 `"current_wip_assigned"` + `inflow=False` + `truncate_on_time=False`로
  돌고 있었다 — 학습한 MDP와
  평가받는 MDP가 다른 train/serve skew. 이제 학습용 데이터셋 복사본에 추론과 같은
  플래그를 채워 넣는다.
- **PPO 안정화**: `n_epochs` 10 → 4, `target_kl=0.03`, lr 선형 감쇠,
  `net_arch=[256,256]`, `gamma` 0.99 → 0.997.
- **엔트로피 스케줄**: 과거에는 `ent_coef` 0.05 → 0을 **초반 20%**에 끝냈는데,
  이건 "탐색이 BC 정책을 망가뜨린다"를 탐색을 죽여서 막는 대증요법이었다. 이제
  ④ 앵커가 그 역할을 제대로 하므로 0.01 → 0.001을 **80% 구간**에 걸쳐 감쇠시켜
  Dedication '위'를 찾을 탐색 여지를 남긴다.

**수렴 차트**: KPI 평가를 쓰면 `EvalCallback`이 돌지 않아 `evaluations.npz`가 없다.
대신 `models/{...}/logs/kpi_evaluations.json`에 KPI 곡선이 남고,
`agent/training_report.py`가 이걸 읽어 **KPI 점수 · 생산량 · 전환수 + Dedication
기준선**을 한 패널로 그린다. 점수(수십 규모)와 생산량(수백 규모)은 축을 분리한다
— 한 축에 그리면 축 범위가 생산량에 끌려가 점수 곡선이 다시 '평평해 보인다'.
`verdict`도 shaping 보상이 아니라 이 KPI 곡선 기준으로 판정한다.

**시연자 선택 (`agent/experts.py`)**: 후보는 `dedication` / `minprogress` /
`earliest_st`(RL env의 버킷 단위로 재구현한 버전)이고, 기본 후보군은
`("dedication", "earliest_st")`. 시나리오마다 후보를 실제로 굴려
`생산량 − 전환수`가 가장 높은 쪽을 그 시나리오의 시연자로 쓴다. BENCH_SUITE
실측으로는 대칭·제품과잉·전환과중 6종은 `dedication`, 부하 불균등 2종
(LOAD_skew·LOAD_stmix)은 `earliest_st`가 선택된다 — 이 조합의 오라클 점수는
**85.0**(전담 단독 77.0)이라, BC 출발점 자체가 8점 올라간다.

### 벤치마크 / 일반화 검증

| 스위트 | 생성 | 용도 | Dedication 기준선 |
|--------|------|------|-------------------|
| BENCH_SUITE (8종) | `benchmark/gen_bench_suite.py` | 학습 + 평가(co-train) | 생산 161/208, 전환 84, **점수 77.0** |
| HOLDOUT_SUITE (6종) | `benchmark/gen_holdout_suite.py` | **학습에 쓰지 않음** — 일반화 검증 | 생산 139/180, 전환 67, **점수 72.0** |
| TRAIN_POOL (48종) | `benchmark/gen_train_pool.py` | 도메인 랜덤화 학습 전용 | — |

```bash
python benchmark/gen_bench_suite.py            # 평가 스위트
python benchmark/gen_holdout_suite.py          # 일반화 검증 스위트
python benchmark/gen_train_pool.py --count 48  # 학습 시나리오 풀

# 학습 + 벤치마크 + 일반화 검증
python benchmark/train_bench_suite.py --timesteps 900000 --train-pool --holdout

# 저장된 모델만 채점
python benchmark/compare_vs_dedication.py --suite bench
python benchmark/compare_vs_dedication.py --suite holdout
```

`compare_vs_dedication.py`는 `dedication` 기준선 대비 승/무/패와 점수 격차를
데이터셋별로 출력한다.

> ⚠️ **평가 데이터로 학습하면 점수는 오르고 실력은 떨어진다.**
> BENCH_SUITE 8종만으로 60만 스텝 co-train하면 그 8종에서는 dedication 대비
> **+14점**(4승 4무 0패)이 나오지만, 학습에 쓰지 않은 HOLDOUT 6종에서는
> **−25점**으로 뒤집힌다 — 전담만 하면 되는 대칭 케이스(H_SYM_6x6)조차
> 전환 0회 → 17회로 무너진다. 정책이 아니라 그 8개의 정답 순열을 외운 것이다.
> 그래서 실전 학습은 `--train-pool`(도메인 랜덤화)을 함께 쓰고, 결과는 반드시
> `--suite holdout`으로 재검증한다. 학습 시나리오가 12개 이하이고 별도 평가셋도
> 없으면 `SchedulingAgent.train()`이 이 위험을 로그로 경고한다.
>
> **운영 데이터에도 같은 이야기가 적용된다** — RULE_TIMEKEY 기간을 1~2개만
> 써서 학습하면 그 기간을 외운다. 기간을 최대한 많이 모아
> (`main.py train --all`) `dataset_sampling="random_reset"`(기본)으로 돌리는 걸
> 권장한다.

### FAC_ID별 모델 관리

RL 모델은 `models/{FAC_ID}/`(체크포인트·best·logs 포함) 아래에 FAC_ID별로 분리해서
저장·로드된다. `SchedulingAgent.train()/save()/load()/model_exists()`에 `fac_id`를
넘기면 그 FAC_ID 전용 경로를 쓰고, 넘기지 않으면 기존 공용 경로(`models/` 바로 아래)를
그대로 써서 이전에 학습된 모델과 호환된다. `main.py train`/`infer`, API의
`/api/train`·`/api/train/start`·`/api/inference`·`/api/inference/compare`는
전달된(또는 현재 선택된) FAC_ID로 자동 라우팅한다.

---

## 추론 옵션

| 옵션 | 기본 | 의미 |
|------|------|------|
| `enable_wip_inflow` | `False`(추론) / `True`(학습 sim) | flow 다음 공정 WIP 유입 |
| `termination_mode` | `current_wip_assigned` | 현재 재공 배정 완료 시 종료 |
| `record_history` | `False` | UI 재생용 snapshot |
| `decision_log` | `False` | step별 진단 로그 |

---

## 출력 · DB 적재

### 파일 (`dataset/.../output/`)

| 파일 | 설명 |
|------|------|
| `output.json` | RTS 적재 payload (`RTS_RSLT_MAS`, `RTS_EQPCONVPLAN_INF`) |
| `output/sql/*.sql` | DELETE+INSERT 스크립트 |
| `result_full.json` | UI/디버그용 전체 결과 |

### 테이블 DDL (최초 1회)

```bash
cp data/sql.example/rts_output_tables.sql data/sql/
# @db alias 를 환경에 맞게 수정

python main.py db-load --ddl-only
# 또는 적재와 함께
python main.py db-load --ddl --facid FAC001 --split infer
```

| 테이블 | 용도 |
|--------|------|
| `RTS_RSLT_MAS` | 스케줄 결과 (매 회차 동일 FAC_ID 전체 교체 — 최신 결과만 유지, 다른 FAC_ID는 영향 없음). `PRODUCE_QTY`/`PLAN_QTY`는 RULE_TIMEKEY(07:00) 기준 당일 값이며 다음 07:00에 다음 날 계획으로 넘어간다(누적 아님) |
| `RTS_RSLT_HIS` | 스케줄 이력 (삭제 없이 INSERT만 누적, `EXEC_TIMEKEY`가 PK에 포함되어 같은 회차 재실행도 별도 행으로 쌓임) |
| `RTS_EQPCONVPLAN_INF` | Conversion 계획 (동일 FAC_ID+RULE_TIMEKEY 기존 행만 교체, 다른 회차는 누적. 옵션: `CONFIG.env.conv_output_enabled`, 기본 True. RULE_TIMEKEY 기준 `CONFIG.env.conv_output_window_minutes`, 기본 60분 이내에 시작하는 건만) |
| `RTS_EQPCONVPLAN_HIS` | Conversion 이력(위와 동일한 옵션/window 적용, 삭제 없이 INSERT만 누적, `EXEC_TIMEKEY`가 PK에 포함) |
| `RTS_PERFMON_HIS` | KPI 이력 (옵션: `--save-kpi` / `save_kpi=true`) |
| `RTS_VALIDATION` | 투입 불가 장비 재공 선택 건수 집계, EQP/PPK/OPER 조합별 (옵션: `--save-kpi` / `save_kpi=true`) |

### DB 적재

`main.py infer`와 `POST /api/inference`는 추론 후 **항상** output/sql을 Oracle RTS 테이블에 적재합니다
(별도 옵션 아님). `--db`/`db_alias`로 대상 DB alias를, `--no-history`/`no_history`로 HIS 테이블 적재
여부를 조정할 수 있습니다.

**테이블별로 다른 DB에 적재**하고 싶으면(예: `RTS_RSLT_MAS`/`RTS_RSLT_HIS`는 운영 DB, `RTS_PERFMON_HIS`는
개발/집계 DB) `config/output_db_routing.yaml.example`을 `config/output_db_routing.yaml`로 복사해
다르게 보낼 테이블만 적으세요(스키마 DDL과는 별개의 선택 파일):

```yaml
RTS_PERFMON_HIS: Dev
RTS_VALIDATION: Dev
```

여기 없는 테이블(또는 파일 자체가 없는 경우)은 그대로 `--db`/`db_alias` 인자 하나로 적재됩니다.

`RTS_EQPCONVPLAN_INF`/`RTS_EQPCONVPLAN_HIS` 저장 자체는 `CONFIG.env.conv_output_enabled`(기본 `True`)
옵션으로 켜고 끌 수 있습니다. 켜져 있으면 RULE_TIMEKEY 기준 `CONFIG.env.conv_output_window_minutes`
(기본 60분) 이내에 시작하는 전환만 기록됩니다 — 그보다 먼 미래의 전환은 재계획 여지가 커 추측성이므로
확정 출력에서 제외합니다(간트나 API 응답의 `conversion_plans`에는 영향 없이 항상 전체가 보입니다).

```bash
# 추론 (결과는 자동으로 DB 적재됨)
python main.py infer --facid FAC001

# KPI(RTS_PERFMON_HIS)도 함께 저장/적재
python main.py infer --facid FAC001 --save-kpi

# 기존 output 폴더 적재
python main.py db-load --facid FAC001 --split test --period 20260624070000

# output.json 직접 적재
python main.py db-load --json data/dataset/FAC001/infer/output/output.json

# SQL 재생성 후 적재
python main.py db-load --facid FAC001 --split infer --regenerate-sql

# HIS 테이블 생략
python main.py db-load --facid FAC001 --split infer --no-history
```

Python:

```python
from data.writer import load_output_sql_files, load_output_json, apply_output_ddl

apply_output_ddl(db_alias="Prd")
load_output_sql_files("data/dataset/FAC001/infer/output", db_alias="Prd")
```

DB 연결: `config/databases.yaml` + `python main.py db-check`

운영/개발 서버를 별도 DB로 분리하려면 `config/databases.yaml` 최상위에 `envs:` 블록을 추가해
prd/dev 접속정보를 함께 넣고 `APP_ENV=production` / `APP_ENV=development` 로 실행하세요
(자세한 내용은 `docs/DEPLOYMENT.md` 1.2절 참고).

실행된 SQL은 `logs/sql_fetch.log`(SELECT, 입력 fetch)와 `logs/sql_load.log`(INSERT/DELETE/DDL,
DB 적재)에 남습니다. 두 로그 모두 자정에 자동 회전되고 백업 3일치까지만 보관되어(`utils/file_logger.py`)
디스크 사용량이 계속 늘어나지 않습니다. ERROR 이상(예: SQL 실행 실패)은 파일과 별도로 터미널(stderr)에도
`[ERROR] 2026-07-10 10:10:10 [rts_eqpconvplan_inf.sql] FAILED: ORA-00942 ...` 형태의 한 줄 요약이 출력됩니다.

---

## 운영 CLI

모든 명령은 `python main.py <command> ...` 형태이며, `--facid`는 대부분 필수입니다.
전체 옵션은 `python main.py <command> --help`로 확인하세요.

### 1. 데이터 수집 (collect / sample)

| 명령어 | 설명 |
|--------|------|
| `python main.py collect --facid FAC001 --split train --prevcnt 3 --once` | 최근 3개 RULE_TIMEKEY 학습용 데이터 1회 수집 (Oracle SQL → JSON) |
| `python main.py collect --facid FAC001 --split train --interval 3600` | 1시간 주기로 반복 수집 (daemon 모드, `--once`/`--interval 0`이면 1회) |
| `python main.py collect --facid FAC001 --split train --from 20260621070000 --to 20260623070000` | 구간(RULE_TIMEKEY) 지정 수집 |
| `python main.py collect --facid FAC001 --split train --snapshot --period 20260621070000` | 특정 RULE_TIMEKEY 1건 스냅샷 수집 |
| `python main.py collect --facid FAC001 --split test --lotcd LOT_A` | `:LOT_CD` 바인드 지정 수집 (기본: `COLLECTOR_LOT_CD`/`SQL_LOT_CD`) |
| `python main.py sample --facid FAC001 --bootstrap` | Oracle 없이 train 3일 + test 1일 + infer 샘플을 한 번에 생성 |
| `python main.py sample --facid FAC001 --split train --scenario pacing_steady` | 특정 시나리오(`default`/`pacing_steady`/`random` 등) 샘플 생성 |
| `python main.py sample --facid FAC001 --period 20260621070000` | 특정 RULE_TIMEKEY 폴더에 샘플 생성 |

### 2. 학습 (train)

| 명령어 | 설명 |
|--------|------|
| `python main.py train --facid FAC001 --prevcnt 3` | 이미 수집된 train 폴더 중 최근 3개로 학습 |
| `python main.py train --facid FAC001 --from 20260621070000 --to 20260623070000` | RULE_TIMEKEY 구간 지정 학습 |
| `python main.py train --facid FAC001 --ruletimekey 20260621070000` | 단일 RULE_TIMEKEY로 학습 |
| `python main.py train --facid FAC001 --all` | train 폴더 전체로 학습 |

### 3. 테스트 / 검증 (test)

| 명령어 | 설명 |
|--------|------|
| `python main.py test --facid FAC001` | 최신 test dataset JSON 검증 |
| `python main.py test --facid FAC001 --prevcnt 5` | 최근 수집된 test 폴더 중 5개 검증 |
| `python main.py test --facid FAC001 --from 20260621070000 --to 20260623070000` | RULE_TIMEKEY 구간 검증 |
| `python3 -m pytest tests/test_writer.py tests/test_db_load.py tests/test_scheduling_env.py -q` | 핵심 모듈 단위 테스트 |
| `python3 -m pytest -q` | 전체 회귀 테스트 |
| `cd frontend && npm run build` | 프론트엔드 빌드 검증 |

### 4. 벤치마크 (benchmark) — [상세](#벤치마크-증명된-최적해)

| 명령어 | 설명 |
|--------|------|
| `python -m benchmark.optimal.runner` | 등록된 전체 알고리즘으로 10개 최적해 케이스 채점 |
| `python -m benchmark.optimal.runner --algo earliest_st --algo minprogress` | 특정 알고리즘만 지정해 채점 |
| `python3 -m pytest tests/test_optimal_bench.py -v` | 벤치마크 최적값 도달 여부 회귀 테스트 |

### 5. 추론 (infer) — Oracle SQL 조회, 결과는 항상 DB 적재까지 수행

| 명령어 | 설명 |
|--------|------|
| `python main.py infer --facid FAC001` | 최신 RULE_TIMEKEY 기준 추론 + DB 적재 |
| `python main.py infer --facid FAC001 --from 20260621170000 --to 20260623170000` | 구간 조회 후 최신값으로 추론 |
| `python main.py infer --facid FAC001 --ruletimekey 20260621070000` | 특정 RULE_TIMEKEY로 추론 |
| `python main.py infer --facid FAC001 --save-kpi` | KPI(`RTS_PERFMON_HIS`)·검증 집계(`RTS_VALIDATION`)도 함께 저장/적재 |
| `python main.py infer --facid FAC001 --decision-log` | step별 EQP/PPK/OPER 결정·미할당 사유를 `result_full.json`에 기록 |
| `python main.py infer --facid FAC001 --include-history` | UI 재생용 history/event snapshot 생성 |
| `python main.py infer --facid FAC001 --enable-wip-inflow` | 공정 완료 시 다음 공정 flow 재공 유입 이벤트 활성화 |
| `python main.py infer --facid FAC001 --strict-validate` | 결과 검증 실패 시 종료코드 1로 종료 |
| `python main.py infer --facid FAC001 --db Dev --no-history` | 대상 DB alias 지정, HIS 테이블 적재 생략 |
| `python main.py infer --facid FAC001 --timeout 300` | DB 조회~DB 적재 전체 5분 제한 |

### 기타 (DB 적재 / UI / 진단)

| 명령어 | 설명 |
|--------|------|
| `python main.py db-load --ddl-only` | output 테이블 DDL만 생성 (최초 1회) |
| `python main.py db-load --facid FAC001 --split test --period 20260624070000` | 기존 output 폴더를 DB에 적재 |
| `python main.py db-load --json data/dataset/FAC001/infer/output/output.json` | `output.json`을 직접 적재 |
| `python main.py db-check` | DB alias 설정(`databases.yaml`/`.env`) 진단 |
| `python main.py ui` | React UI + API 서버 실행 |

---

## 서버 실행

최초 1회:

```bash
pip install -r requirements.txt
cp .env.example .env
cp config/databases.yaml.example config/databases.yaml   # envs: 블록에 운영/개발 DB 정보 입력
```

`APP_ENV`로 `config/databases.yaml`의 `envs:` 블록 안에서 운영/개발 DB 설정을 선택합니다
(자세한 내용은 `docs/DEPLOYMENT.md` 1.2절).

```bash
# 개발 서버 (자동 리로드)
APP_ENV=development python -m uvicorn api.server:app --reload --host 127.0.0.1 --port 8001

# 운영 서버 (다중 워커)
APP_ENV=production python api/start_production.py --host 0.0.0.0 --port 8001 --workers 4
```

Windows `cmd`에서는 환경변수를 `set`으로 지정합니다 (창을 새로 열면 다시 설정 필요):

```cmd
set APP_ENV=development
python -m uvicorn api.server:app --reload --host 127.0.0.1 --port 8001
```

프론트엔드(선택):

```bash
cd frontend
npm install
npm run dev
```

헬스 체크: `curl http://localhost:8001/api/health` (자세한 배포 절차는 `docs/DEPLOYMENT.md` 참고)

### 주기 추론 스케줄러 (선택)

API 서버 기동 시 `python main.py infer`와 동일한 추론+DB 적재를 일정 주기로 자동 반복할 수 있습니다.
기본은 **비활성**(opt-in) — 운영 DB에 의도치 않게 자동 적재되는 사고를 막기 위해, 아래 두 값을 모두
명시해야 켜집니다.

```bash
# .env
SCHEDULER_ENABLED=true
SCHEDULER_FAC_ID=FAC001
SCHEDULER_INTERVAL_SECONDS=3600   # 기본 3600(1시간)
SCHEDULER_ALGORITHM=scheduling_rl # 기본값, minprogress/earliest_st/dedication도 가능
SCHEDULER_DB_ALIAS=Prd            # 미지정 시 databases.yaml default
SCHEDULER_NO_HISTORY=false
```

서버 기동 직후 1회 즉시 실행하고, 이후 `SCHEDULER_INTERVAL_SECONDS`마다 반복합니다. 한 회차가
실패해도(DB 오류 등) 다음 주기에 다시 시도하며 서버 자체는 죽지 않습니다. 현재 상태는
`GET /api/scheduler/status`(활성 여부, 누적 실행 횟수, 마지막 실행 시각/결과)로 확인할 수 있습니다.

---

## UI

| 구성 | 포트 |
|------|------|
| FastAPI | 8001 |
| Vite | 5173 |

Inference 탭: 단일 추론·알고리즘 비교·`output.json` 오프라인 뷰어

`POST /api/inference`는 `lot_cd`만 필수이며, `fac_id`는 미지정 시 현재 선택된 입력 폴더/서버 설정을,
`rule_timekey`는 미지정 시 해당 `fac_id`의 최신 값을 자동으로 조회해 추론합니다.
추론 결과는 항상 Oracle RTS 테이블에 적재되며(`db_alias`/`no_history`로 대상 DB·HIS 테이블 적재 여부만 조정),
적재 성공 여부는 응답의 `infer_meta.db_loaded`로 확인할 수 있습니다.

---

## 벤치마크 (증명된 최적해)

`benchmark/optimal/`은 정답을 수학적으로 증명할 수 있을 만큼 단순화된 시나리오에서
알고리즘이 실제로 그 최적값에 도달하는지 채점합니다. 알고리즘끼리의 상대 비교가
아니라, "이 문제의 정답은 이 값"이라는 절대 기준 대비 평가입니다. 모든 케이스는
다중 EQP 기준이며(케이스별 증명은 `benchmark/optimal/cases.py` 주석 참고), 총 10개입니다.

#### 단일 공정 케이스 3개

| 케이스 | 구성 | 증명된 최적값 |
|--------|------|----------------|
| `dedicated_assignment` | EQP 3대 × PPK 3종(각기 다른 LOT_CD) — 홈 배정은 라운드로빈으로 일부러 섞어둠 | 생산 24개, 전환 0회 |
| `mixed_conversion_two_eqp` | EQP 2대(1대는 초기 셋업 미지정=무전환 가능, 1대는 다른 LOT_CD로 이미 세팅=전환 강제), PPK 1종 | 생산 12개, 전환 1회 |
| `overflow_conversion_three_eqp` | EQP 3대(2대는 PPK 1종씩 전담, 나머지 1대는 초기 셋업이 다른 오버플로 전용), PPK 2종 | 생산 20개, 전환 1회 |

#### 다중 공정(OPER) × 다중 제품 케이스 7개

위 3가지 단일 공정 패턴을 서로 다른 OPER(OPER001/OPER002)에 독립적으로 배치해
조합한 케이스들입니다. 공정별로 EQP·EQP_MODEL_CD를 완전히 분리하고 각 공정의
초기 재공을 처음부터 충분히 채워두므로(공정별 독립 EQP 설계 — 파이프라인 재공
이어받기 타이밍 자체는 검증하지 않음), 최적값은 두 공정 최적값의 단순 합입니다.

| 케이스 | 구성 | 증명된 최적값 |
|--------|------|----------------|
| `two_stage_dedicated_small` | OPER001·OPER002 각각 EQP2×PPK2 전담 | 생산 32개, 전환 0회 |
| `two_stage_dedicated_mixed` | OPER001: EQP3×PPK3 전담 / OPER002: 전환 강제·무료 혼합(EQP2·PPK1) | 생산 36개, 전환 1회 |
| `two_stage_mixed_mixed` | OPER001·OPER002 각각 전환 강제·무료 혼합(EQP2·PPK1) | 생산 24개, 전환 2회 |
| `two_stage_dedicated_overflow` | OPER001: EQP2×PPK2 전담 / OPER002: 전담2+오버플로1(EQP3·PPK2) | 생산 36개, 전환 1회 |
| `two_stage_overflow_overflow` | OPER001·OPER002 각각 전담2+오버플로1(EQP3·PPK2) | 생산 40개, 전환 2회 |
| `two_stage_mixed_overflow` | OPER001: 전환 강제·무료 혼합(EQP2·PPK1) / OPER002: 전담2+오버플로1(EQP3·PPK2) | 생산 32개, 전환 2회 |
| `two_stage_dedicated_large` | OPER001: EQP3×PPK3 전담 / OPER002: EQP2×PPK2 전담 | 생산 40개, 전환 0회 |

> `minprogress`는 EQP 수가 많아 동시에 idle 결정이 몰리는 일부 다중 공정
> 케이스(`two_stage_dedicated_overflow`/`two_stage_overflow_overflow`/
> `two_stage_mixed_overflow`)에서 `PYTHONHASHSEED`에 따라 conversions 결과가
> 프로세스 실행마다 달라지는 재현성 이슈가 있습니다(같은 프로세스 안에서는
> 안정적). `earliest_st`는 동일 조건에서 항상 결정적입니다. 근본 원인(시뮬레이터
> 내부 순회 순서 추정)은 아직 조사 중이며, 회귀 테스트에서는 이 조합만
> `xfail(strict=False)`로 표시해뒀습니다.

### CLI로 실행

```bash
python -m benchmark.optimal.runner                                    # 등록된 전체 알고리즘
python -m benchmark.optimal.runner --algo earliest_st --algo minprogress
```

케이스 × 알고리즘별 PASS/FAIL과 실제/최적 생산·전환 수를 콘솔에 출력하고,
`data/dataset/OPTIMAL_BENCH/optimal_bench_results.json`에 저장합니다.

### UI로 실행

`python main.py ui` 실행 후 **테스트 셋** 탭 상단의 **최적해 벤치마크** 카드에서
"벤치마크 실행" 버튼을 누르면 케이스별 PASS/FAIL·생산/전환 실제값 대 최적값·
증명 텍스트를 표로 볼 수 있습니다(API: `GET /api/test/optimal-bench`,
쿼리 `?algorithms=earliest_st,minprogress`로 알고리즘 필터링 가능).

### 회귀 테스트

```bash
python3 -m pytest tests/test_optimal_bench.py -v
```

`minprogress`/`earliest_st`가 각 케이스의 증명된 최적값에 도달하는지 검증합니다.
알려진 실제 격차(예: `earliest_st`가 전환 비용을 고려하지 않아 `dedicated_assignment`에서
손해를 보는 경우)는 `xfail(strict=True)`로 명시되어 있어, 알고리즘이 개선되어
우연히 통과하면 테스트가 실패하며 알려줍니다. 위에서 언급한 재현성 이슈가 있는
3개 조합만 `xfail(strict=False)`로 별도 표시되어 있어(값이 실행마다 달라져도
빌드를 깨뜨리지 않음), `tests/test_optimal_bench.py`의 `_KNOWN_GAPS_FLAKY`를
참고하세요.

---

## 테스트

```bash
python3 -m pytest tests/test_writer.py tests/test_db_load.py tests/test_scheduling_env.py -q
python3 -m pytest -q
cd frontend && npm run build
```

---

## 의존성

Python: gymnasium, stable-baselines3, sb3-contrib, torch, fastapi, oracledb, numpy

Frontend: React, TypeScript, Vite, Plotly

---

## 주의사항

- obs/action/reward 변경 시 기존 PPO checkpoint와 **호환되지 않을** 수 있습니다.
- RL은 EQP·LOT을 직접 선택하지 않습니다. Earliest-ST는 EQP×LOT 전역 선택 휴리스틱입니다.
- 학습(`SchedulingEnv`)은 기본 `enable_wip_inflow=True`, 추론 runner는 `False` — flow 밸런스 평가 시 옵션을 맞추세요.
- 추론 결과는 항상 저장됩니다(output/result_full 파일 및 SQL 생성). history는 기본 미생성이며, DB 적재·재생이 필요하면 옵션을 켜세요.
