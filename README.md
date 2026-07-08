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
| `discrete_arrange` | `(EQP, LOT, OPER)` | 실제 EQP×carrier 조합, ST, WF_QTY, `LOT_STAT_CD` |
| `abstract_arrange` | `(PPK, OPER, EQP_MODEL)` | **arrange** = 장비 재공 투입 가능 여부 템플릿 |
| Runtime WIP | `(PPK, OPER)` + LOT list | 현재/유입 재공, `oper_in_time` |

부가 입력: `flow`, `batch_info`(LOT_CD/TEMP), `tool_capacity`, `eqp_initial_state`, `split`, `conversion_group`

#### LOT_STAT_CD (`discrete_arrange`, 선택 – 미지정 시 `WAIT`)

LOT의 현재 상태 코드로 `PROC` / `LOAD` / `SELE` / `RESV` / `WAIT` 중 하나입니다.

- `WAIT`: 알고리즘이 자유롭게 스케줄링(장비·순서 결정)할 수 있는 재공(기존 동작과 동일).
- `PROC`/`LOAD`/`SELE`/`RESV`: 이미 확정된 재공. 반드시 해당 행의 `EQP_ID`에만 배정되며,
  같은 장비에 여러 건이 있으면 **PROC → LOAD → RESV → SELE** 순으로 강제 배정됩니다(동일 상태는 입력 순).
  시뮬 시작(`reset`) 시 강제 carrier **전건**을 장비에 선반영한다.
  `PROC`은 즉시 가공 시작, `LOAD`/`RESV`/`SELE`는 장비에 선부착(staged) 후
  가공 슬롯이 비면 자동 투입(RL step 없음).
  이 LOT들은 다른 장비의 배정 후보로는 전혀 노출되지 않고, 자신의 순번이 될 때까지는
  자기 장비에서도 다른 어떤 재공보다 우선 처리되어야 합니다.

#### 전환 그룹 제약 (`conversion_group.json`, 선택)

같은 그룹 안의 `(LOT_CD, TEMP)`로만 전환을 허용하고 **다른 그룹으로의 전환은 배정 후보에서 제외**합니다(행동 공간 축소 → 문제 단순화). 파일이 없으면 제약은 비활성(기존 동작).

```json
[
  {"GROUP_ID": "G1", "LOT_CD": "LC_A", "TEMP": "T600"},
  {"GROUP_ID": "G1", "LOT_CD": "LC_B", "TEMP": "T600"},
  {"GROUP_ID": "G2", "LOT_CD": "LC_C", "TEMP": "T600"}
]
```

규칙:
- 동일 셋업(전환 없음)·첫 배정(EQP 셋업 미지정)은 항상 허용
- `from`·`to`가 **둘 다 그룹에 속할 때만**, 두 그룹이 다르면 차단
- 그룹에 없는 `(LOT_CD, TEMP)`는 제약 없음(미지정 = 와일드카드)

> 효과: EQP가 처음 잡은 그룹에 고정되어 그룹 내에서만 전환합니다. 도달 불가능한 제품이 생길 수 있으나(생산량 소폭 감소), 불필요한 교차 그룹 전환이 사라집니다.

### arrange vs discrete (런타임)

- **후보 생성**: EQP별 `abstract_arrange` × WIP LOT을 펼침
- **discrete**: `(LOT, EQP, OPER)`가 `proc_time_matrix`에 있으면 ST를 discrete 값으로 사용 (`ABSTRACT=False`)
- **abstract**: discrete 행이 없으면 arrange ST 사용 (`ABSTRACT=True`)

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
| Action | `Discrete(O×P)` = `(OPER, PPK)` bucket |
| Mask | 현재 idle EQP feasible bucket |
| obs_dim | `6 + O×P×6 + O×P×K×5` = **936** (O=3, P=10, K=5) |

**Bucket feature (po 6ch + pom 5ch)**: WIP 비율, urgency, achievable_ratio, projected_cover_ratio, starve_time_norm / ST, conversion·tool 가용, avoidable_frac, setup_changed 등.
prev/post takt·LOT_CD/TEMP 인코딩 채널은 제거됨 — takt는 정적 설비 수(`n_eqp_per_oper`) 기반이라 설비 공유·실시간 배정 상태를 반영 못 했고, LOT_CD/TEMP는 범주형 ID를 순서 있는 스칼라로 인코딩해 신호 품질이 낮았음 (전환 관련 정보는 pom_feats에 이미 더 정확히 포함).

### Reward (`config.py` 기본값)

| 항목 | 가중치 | 역할 |
|------|--------|------|
| `w_plan_hit` | 3.0 | achievable 상한 대비 계획 gap 감소 |
| `w_pacing` | 2.0 | 선형 takt ideal 추종 (재공 한도 반영) |
| `w_flow_balance` | 1.5 | WIP 편중 공정 배정·후공정 feeding |
| `flow_balance_starving_cover_min` | 120 | 후속 ready WIP÷capa(분) ≤ 이 값일 때만 feeding 보너스 |
| `w_same_oper` | 1.0 | 동일 OPER 연속 (과생산 시 억제) |
| `w_idle_per_min` | -0.1 | idle 패널티 |
| `w_conversion` | -10.0 | LOT_CD/TEMP 전환 |
| `reward_clip` | ±10.0 | PPO 안정화 |

`use_achievable_target=True`: 재공이 부족하면 무리한 계획 추격을 막고, 선행 공정 투입 유도.

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
| `output.json` | RTS 적재 payload (`RTS_RSLT_INF`, `RTS_EQPCONVPLAN_INF`) |
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
| `RTS_RSLT_INF` | 스케줄 결과 (RULE_TIMEKEY 단위 교체) |
| `RTS_RSLT_HIS` | 스케줄 이력 |
| `RTS_EQPCONVPLAN_INF` | Conversion 계획 |
| `RTS_EQPCONVPLAN_HIS` | Conversion 이력 |

### DB 적재

```bash
# 추론 후 output/sql 적재
python main.py infer --facid FAC001 --db-load

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

---

## 운영 CLI

```bash
# 데이터 수집
python main.py collect --facid FAC001 --split train --prevcnt 3 --once

# 학습 / 검증 (dataset JSON)
python main.py train --facid FAC001 --prevcnt 3
python main.py test --facid FAC001

# 추론 (Oracle SQL 조회)
python main.py infer --facid FAC001
python main.py infer --facid FAC001 --db-load
python main.py infer --facid FAC001 --from 20260621170000 --to 20260623170000

# 샘플 데이터
python main.py sample --facid FAC001 --bootstrap

# UI
python main.py ui
```

---

## UI

| 구성 | 포트 |
|------|------|
| FastAPI | 8000 |
| Vite | 5173 |

Inference 탭: 단일 추론·알고리즘 비교·`output.json` 오프라인 뷰어

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
- 빠른 UI 추론은 `save_output=false` / history 미생성이 기본입니다. DB 적재·재생이 필요하면 옵션을 켜세요.
