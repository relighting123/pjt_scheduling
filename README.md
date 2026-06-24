# pjt_scheduling

반도체 **Scheduling** 문제를 강화학습(RL)과 휴리스틱으로 실행하는 프로젝트입니다.
`discrete_arrange` / `abstract_arrange` / `plan` / `flow` 입력을 전처리해 DES(Discrete Event Simulation)를 돌리고,
EQP가 idle이 될 때마다 **현재 결정 EQP에 투입할 (PPK, OPER)** 를 선택합니다. LOT은 규칙 기반으로 자동 선택됩니다.

---

## 현재 구조 요약

```
pjt_scheduling/
├── main.py                 # 운영 CLI (collect / train / validate / infer / ui / db-check)
├── config.py               # 경로, 환경 축 크기, RL, reward 파라미터
├── requirements.txt
│
├── agent/
│   ├── rl_agent.py         # MaskablePPO 학습/로드/예측
│   ├── minprogress_agent.py
│   ├── earliest_st_agent.py
│   └── registry.py         # 알고리즘 목록
│
├── env/
│   └── scheduling_env.py   # Gymnasium Env, Discrete action, action mask
│
├── simulation/
│   ├── simulator.py        # DES 엔진, EQP/LOT/WIP, 보상, 전환, tool cap
│   ├── decision_log.py     # step별 결정/미할당 사유 진단
│   └── events.py
│
├── data/
│   ├── dataset/            # {FAC_ID}/{train|test|infer}/.../input|output (git 제외)
│   ├── loader/             # Oracle SQL/JSON 로드 + preprocess
│   ├── writer/             # 추론 결과 output.json / SQL 생성
│   ├── sql.example/        # SQL 템플릿
│   ├── sql/                # 환경별 SQL (git 제외)
│   └── generator.py        # 개발/테스트용 샘플 생성 함수
│
├── inference/
│   └── runner.py           # 단일/비교 추론 실행
│
├── api/
│   ├── server.py           # FastAPI
│   ├── serializers.py      # UI 응답 경량/상세 직렬화
│   └── train_service.py
│
├── frontend/               # React + TypeScript + Vite + Plotly
├── tests/
└── models/                 # PPO 체크포인트 (.zip, git 제외)
```

| 모듈 | 역할 |
|------|------|
| `data.loader` | Oracle/JSON 입력 로드, 검증, `env_data` 전처리 |
| `simulation` | EQP 상태, WIP 풀, flow 유입, conversion/tool, reward, event log |
| `env` | Gymnasium `reset/step/action_masks` |
| `agent` | RL/휴리스틱 정책 |
| `inference` | 추론 루프, 빠른/상세 실행 옵션 |
| `api` + `frontend` | UI, 학습/추론/비교 API |

---

## 데이터 모델

| 레이어 | 단위 | 설명 |
|--------|------|------|
| `discrete_arrange` | `(EQP_ID, LOT_ID, OPER_ID)` | 실제 LOT/EQP feasible 조합, ST, WF_QTY, EQP_MODEL_CD |
| `abstract_arrange` | `(PPK, OPER, EQP_MODEL_CD)` | 추상 route 템플릿, 후속 flow 유입 가능성 |
| Runtime WIP pool | `(PPK, OPER)` + LOT list | 현재/유입 재공 수량, `oper_in_time`, LOT meta |

주요 부가 입력:

- `flow.json`: PPK별 공정 순서. 유입 재공 이벤트 ON 시 공정 완료 LOT이 다음 OPER WIP로 주입됩니다.
- `batch_info.json`: `(PPK, OPER) -> LOT_CD, TEMP`. conversion/tool 판단의 우선 소스입니다.
- `tool_capacity.json`: `(LOT_CD, EQP_MODEL)` 단위 동시 가공 cap.
- `eqp_initial_state.json`: EQP별 초기 LOT_CD/TEMP/직전 PPK/OPER.
- `split.json`: wafer lot split 규칙.

---

## 의사결정 흐름

한 step의 현재 구현은 다음과 같습니다.

1. simulator가 배정 가능한 idle EQP를 찾습니다.
   - 기본 EQP 선택: `eqp_ids` 순서상 첫 번째 feasible idle EQP
   - `earliest_st` 추론: idle EQP 중 최소 ST 후보가 있는 EQP 우선 (`eqp_selection="min_st"`)
2. 정책은 **현재 결정 EQP에서 선택할 `(PPK, OPER)` bucket** 을 고릅니다.
3. action mask는 현재 EQP에서 feasible한 `(PPK, OPER)`만 True로 표시합니다.
4. 선택 bucket이 invalid면 현재 feasible 첫 후보로 보정합니다.
5. LOT은 `_auto_select_lot()`이 자동 선택합니다.
6. conversion/tool cap을 확인하고, WIP를 `-1` 한 뒤 가공/전환 이벤트를 생성합니다.
7. 유입 재공 이벤트가 켜져 있으면 공정 완료 시 `flow_next` 기준으로 다음 OPER WIP를 주입합니다.

중요: 현재 RL action은 EQP를 직접 선택하지 않습니다. EQP는 simulator의 현재 결정 EQP입니다.

---

## 강화학습/환경 구성

### MDP

| 요소 | 현재 값/구조 |
|------|--------------|
| 알고리즘 | `MaskablePPO` (`sb3-contrib`, `MlpPolicy`) |
| 행동 공간 | `Discrete(O * P)` = `(OPER, PPK)` bucket |
| action mask | 현재 idle EQP 기준 feasible `(PPK, OPER)` mask |
| LOT 선택 | 규칙 기반 자동 선택 |
| 기본 종료(학습 Env) | assignable work 없음 + busy/converting 없음, 또는 시간/step truncate |
| 추론 종료 | 기본: 현재 재공 배정 완료 기준. 옵션으로 flow 유입 재공까지 확장 |

현재 기본 축(`config.py`):

```text
O = max_oper_count  = 3
P = max_prod_count  = 10
K = max_model_count = 5
F = BUCKET_FEATURES = 14

obs_dim = Global(6) + O*P*K*F + current_eqp(6) + Context(4)
        = 6 + 3*10*5*14 + 6 + 4
        = 2116
```

### 관측 벡터

| 블록 | 내용 |
|------|------|
| Global(6) | 경과 시간, soft cutoff 잔여, LOT pool 비율, 완료율, conversion EQP 비율, tool utilization |
| Bucket `(OPER, PPK, EQP_MODEL, F)` | valid, WIP 비율, min_end, throughput, same_ppk, takt, ST, urgency, LOT_CD/TEMP, conversion/tool 가능 여부 |
| Current EQP(6) | 현재 결정 EQP idle/busy, 직전 LOT_CD/TEMP, 잔여 busy 시간, conversion 필요 여부 |
| Context(4) | 직전 배정 PPK/OPER/EQP/LOT_CD |

### Reward

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `w_same_oper` | `+2.0` | 동일 OPER 연속 |
| `w_same_prod` | `+0.5` | 동일 PPK + feasible 재공 있을 때 |
| `w_prod_switch` | `+0.8` | 이전 PPK 재공 고갈 후 전환 |
| `w_idle_per_min` | `-0.5` | EQP idle 시간 |
| `w_completion` | `+1.0` | LOT 투입량 보상 |
| `w_plan_hit` | `+5.0` | 계획 gap 감소 |
| `w_pacing` | `+3.0` | 계획 직선 대비 누적 편차 shaping |
| `w_conversion` | `-30.0` | LOT_CD/TEMP 전환 |
| `w_late_finish` | `-2.0` | soft cutoff 이후 END_TM |

---

## 추론 종료/유입 재공 옵션

추론은 현재 운영 요구에 맞춰 기본이 **현재 재공 기준**입니다.

| 옵션 | 기본값 | 의미 |
|------|--------|------|
| `enable_wip_inflow` | `False` | OFF면 공정 완료 후 다음 공정 WIP를 주입하지 않습니다. 현재 재공만 배정합니다. |
| `current_wip_only` | `None` | 하위 호환용. `None`이면 `not enable_wip_inflow`로 자동 결정됩니다. |
| `termination_mode` | `current_wip_assigned` | 현재 재공이 모두 배정되면 busy 장비 후속 완료를 기다리지 않고 종료합니다. |
| `enable_wip_inflow=True` | - | 공정 완료 시 `flow_next` 기준으로 다음 OPER WIP를 주입하고 계속 배정합니다. |

결과 `stats`에는 다음이 포함됩니다.

- `remaining_wip`
- `remaining_current_wip`
- `terminated`
- `truncated`
- `current_time`
- `steps`
- `termination_mode`
- `enable_wip_inflow`

---

## 추론 빠르게 실행하기

### 기본 원칙

추론 속도와 UI 표시 속도에 가장 큰 영향을 주는 것은 아래 세 가지입니다.

1. `history/event_log/decision_log` payload 생성/전송
2. step별 arrange/WIP snapshot 계산
3. 결과 파일/SQL 저장 I/O

빠른 결과 확인이 목적이면 아래처럼 실행합니다.

| 경로 | 빠른 설정 |
|------|-----------|
| Python | `run_inference(..., record_history=False, record_decision_log=False, enable_wip_inflow=False)` |
| API | `include_history=false`, `decision_log=false`, `enable_wip_inflow=false`, `save_output=false` |
| UI | 기본값 그대로 실행: history/event payload와 output 저장을 생략 |
| CLI | 기본적으로 `--include-history`를 주지 않으면 history snapshot을 만들지 않음 |

### Python

```python
from inference.runner import run_inference

# 가장 빠른 현재 재공 기준 추론
result = run_inference(
    env_data,
    algorithm="rl",
    agent=agent,
    record_history=False,
    record_decision_log=False,
    enable_wip_inflow=False,
)

# flow 유입 재공까지 이어서 추론
result = run_inference(
    env_data,
    algorithm="rl",
    agent=agent,
    record_history=False,
    enable_wip_inflow=True,
)

# UI 재생/step별 분석용 상세 추론
result = run_inference(
    env_data,
    algorithm="rl",
    agent=agent,
    record_history=True,
    record_decision_log=True,
)
```

### API

`POST /api/inference`

```json
{
  "algorithm": "rl",
  "input_folder": "FAC001/infer",
  "enable_wip_inflow": false,
  "include_history": false,
  "decision_log": false,
  "save_output": false
}
```

| 필드 | 설명 |
|------|------|
| `include_history=false` | `history` / replay용 event payload 생략 |
| `decision_log=false` | step별 진단 로그 생략 |
| `save_output=false` | UI 즉시 표시용. output JSON/SQL 저장 I/O 생략 |
| `enable_wip_inflow=false` | 현재 재공만 배정 |

`POST /api/inference/compare`

```json
{
  "algorithms": ["rl", "minprogress", "earliest_st"],
  "input_folder": "FAC001/test/20260624070000",
  "include_history": false,
  "decision_log": false,
  "enable_wip_inflow": false
}
```

비교 화면은 기본적으로 schedule/stats 중심이므로 `include_history=false`가 권장됩니다.

### UI

Inference 화면 옵션:

| 옵션 | 권장 |
|------|------|
| `시뮬레이션 재생 데이터 포함` | 빠른 결과만 필요하면 OFF |
| `결정 로그` | 분석 필요 시에만 ON. ON이면 상세 payload가 필요합니다. |
| `유입 재공 이벤트 사용` | 현재 재공만 보면 OFF, 다음 공정 flow까지 보면 ON |

기본 단일 추론은 빠른 표시를 위해 schedule/stats만 받고, 결과는 스케줄 비교 탭에서 즉시 표시합니다.
재생이 필요하면 `시뮬레이션 재생 데이터 포함`을 켜고 다시 실행합니다.

### CLI

```bash
# 빠른 현재 재공 기준 RL 추론 (기본)
python main.py infer --facid FAC001 --nodb

# 결정 로그 포함
python main.py infer --facid FAC001 --nodb --decision-log

# UI 재생용 history/event snapshot 포함
python main.py infer --facid FAC001 --nodb --include-history

# 다음 공정 flow 유입 재공까지 추론
python main.py infer --facid FAC001 --nodb --enable-wip-inflow
```

CLI `infer`는 현재 RL 모델 추론 경로입니다. 알고리즘 비교는 UI/API의 `/api/inference/compare`를 사용합니다.

---

## 운영 CLI

### 데이터 수집

```bash
mkdir -p data/sql
cp data/sql.example/*.sql data/sql/
# data/sql/*.sql 의 -- @db alias, 테이블명, WHERE 조건을 환경에 맞게 수정

python main.py collect --facid FAC001 --split train --prevdays 3 --once
python main.py collect --facid FAC001 --split test --from 20260601070000 --to 20260603070000 --once
```

### 학습

```bash
# DB 수집 포함 최근 N일 학습
python main.py train --facid FAC001 --prevdays 3

# 기존 JSON만 사용
python main.py train --facid FAC001 --from 20260601070000 --to 20260603070000 --nodb

# 단일 RULE_TIMEKEY
python main.py train --facid FAC001 --ruletimekey 20260624070000 --nodb
```

### 검증/추론/UI

```bash
python main.py validate --facid FAC001 --nodb
python main.py infer --facid FAC001 --nodb
python main.py ui
python main.py db-check
```

---

## UI

| 구성 | 포트 |
|------|------|
| FastAPI | 8000 |
| Vite dev server | 5173 |

```bash
python main.py ui
```

개별 실행:

```bash
uvicorn api.server:app --reload --port 8000
cd frontend && npm install && npm run dev
```

| 탭 | 기능 |
|----|------|
| Dataset | 입력 폴더/데이터 요약 |
| Train | 기간/폴더 선택, PPO 학습, 학습 로그/차트 |
| Test | test dataset 벤치마크 |
| Inference | 빠른 단일 추론, 알고리즘 비교, 선택 시 상세 재생 |

---

## 출력 파일

CLI 추론 또는 API `save_output=true`일 때 `data/dataset/.../output/`에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `output.json` | RTS 적재용 결과 payload |
| `output/sql/*.sql` | RTS insert/delete SQL |
| `result_full.json` | UI/디버그용 schedule/history/event/decision log |

UI 기본 추론은 빠른 표시를 위해 `save_output=false`로 호출합니다.

---

## 테스트

```bash
python3 -m pytest tests/test_decision_log.py tests/test_scheduling_env.py -q
cd frontend && npm run build
```

전체 테스트 실행:

```bash
python3 -m pytest -q
```

현재 일부 테스트 파일은 별도 시나리오 모듈(`data.conversion_scenarios`, `data.pacing_scenarios`)을 요구합니다. 해당 모듈이 없는 환경에서는 collection 단계에서 실패할 수 있습니다.

---

## 의존성

Python:

- gymnasium
- stable-baselines3
- sb3-contrib
- torch
- numpy / pandas / scipy
- fastapi / uvicorn
- oracledb
- python-dotenv / pyyaml

Frontend:

- React
- TypeScript
- Vite
- Plotly

---

## 주의사항

- 관측/action/reward 구조가 바뀌면 기존 PPO checkpoint와 호환되지 않을 수 있습니다.
- 현재 RL action은 EQP를 직접 선택하지 않습니다. EQP는 simulator가 현재 idle feasible EQP로 정합니다.
- 빠른 UI 추론은 history/event payload를 만들지 않으므로 시뮬레이션 재생 탭에는 데이터가 없습니다.
- 결정 로그나 재생이 필요하면 `include_history` 또는 UI의 `시뮬레이션 재생 데이터 포함` 옵션을 켜고 다시 실행하세요.
