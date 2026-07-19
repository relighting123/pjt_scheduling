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

부가 입력: `flow`, `batch_info`(LOT_CD/TEMP), `tool_capacity`, `eqp_initial_state`, `split`, `conversion_group`,
`eqp_conv_plan`(외부 확정 EQP 전환 계획), `eqp_down`(EQP 다운타임)

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
| `RTS_RSLT_INF` | 스케줄 결과 (매 회차 동일 FAC_ID 전체 교체 — 최신 결과만 유지, 다른 FAC_ID는 영향 없음) |
| `RTS_RSLT_HIS` | 스케줄 이력 |
| `RTS_EQPCONVPLAN_INF` | Conversion 계획 (삭제 없이 INSERT만 누적. 옵션: `CONFIG.env.conv_output_enabled`, 기본 True. RULE_TIMEKEY 기준 `CONFIG.env.conv_output_window_minutes`, 기본 60분 이내에 시작하는 건만) |
| `RTS_EQPCONVPLAN_HIS` | Conversion 이력(위와 동일한 옵션/window 적용, 마찬가지로 INSERT만 누적) |
| `RTS_PERFMON_HIS` | KPI 이력 (옵션: `--save-kpi` / `save_kpi=true`) |
| `RTS_VALIDATION` | 투입 불가 장비 재공 선택 건수 집계, EQP/PPK/OPER 조합별 (옵션: `--save-kpi` / `save_kpi=true`) |

### DB 적재

`main.py infer`와 `POST /api/inference`는 추론 후 **항상** output/sql을 Oracle RTS 테이블에 적재합니다
(별도 옵션 아님). `--db`/`db_alias`로 대상 DB alias를, `--no-history`/`no_history`로 HIS 테이블 적재
여부를 조정할 수 있습니다.

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

운영/개발 서버를 별도 DB로 분리하려면 `config/databases.prd.yaml` / `config/databases.dev.yaml`
을 각각 준비하고 `APP_ENV=production` / `APP_ENV=development` 로 실행하세요
(자세한 내용은 `docs/DEPLOYMENT.md` 1.2절 참고).

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
cp config/databases.prd.yaml.example config/databases.prd.yaml   # 운영 DB 정보 입력
cp config/databases.dev.yaml.example config/databases.dev.yaml   # 개발 DB 정보 입력
```

`APP_ENV`로 운영/개발 DB 설정을 선택합니다 (`config/databases.prd.yaml` / `databases.dev.yaml`,
미지정 시 레거시 `config/databases.yaml` 사용 — 자세한 내용은 `docs/DEPLOYMENT.md` 1.2절).

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
