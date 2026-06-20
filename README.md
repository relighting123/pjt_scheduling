# pjt_scheduling

반도체 **Post-Scheduling** 문제를 강화학습(RL)으로 해결하는 프로젝트입니다.  
초기 스케줄링 결과를 입력으로 받아, EQP가 idle이 될 때마다 배정할 LOT을 에이전트가 선택하여 **공정 전환·제품 전환·Idle 시간**을 최소화합니다.

---

## 폴더 구조

```
pjt_scheduling/
├── main.py                 # CLI 워크플로우 진입점 (샘플생성 / 학습 / 추론 / UI)
├── config.py               # 경로, 환경, RL, 보상 파라미터 중앙 설정
├── requirements.txt        # Python 의존성
│
├── agent/
│   └── rl_agent.py         # Stable-Baselines3 PPO 에이전트 래퍼 (학습·저장·예측·평가)
│
├── env/
│   └── scheduling_env.py   # Gymnasium 커스텀 환경 (SchedulingSimulator 래핑)
│
├── simulation/
│   └── simulator.py        # 이산 사건 시뮬레이션(DES) 엔진, 관측·보상 계산
│
├── data/
│   ├── loader.py           # JSON 로드·검증, 샘플 데이터 생성
│   └── preprocessor.py     # 원시 JSON → RL 환경 입력 변환
│
├── inference/
│   └── runner.py           # 학습된 모델로 추론 실행 및 결과 JSON 저장
│
├── api/
│   ├── server.py           # FastAPI REST API (학습·추론·데이터)
│   └── serializers.py      # API 응답 JSON 직렬화
│
├── frontend/               # React + TypeScript UI (Vite)
│   ├── src/
│   │   ├── App.tsx         # 메인 레이아웃
│   │   ├── pages/          # 학습 / 추론 페이지
│   │   ├── components/     # 사이드바, Plotly 차트
│   │   └── lib/            # API 클라이언트, 차트 빌더
│   └── package.json
│
├── external/               # DB 연동 JSON (git 제외)
│   ├── input/              # 기본 입력 폴더 (이름 변경 가능)
│   ├── <case_name>/        # 시나리오별 입력 폴더
│   └── output/
│       └── <case_name>/    # 시나리오별 결과
│
├── utils/
│   └── helpers.py          # 날짜 변환, 인덱스 맵, 검증 유틸
│
└── models/                 # 학습된 모델 저장 (.zip, git 제외)
```

| 모듈 | 역할 |
|------|------|
| `data/` | DB 조회 결과(JSON) 로드·검증·전처리 |
| `simulation/` | EQP/LOT 상태 관리, 결정 시점 탐색, 보상·관측 계산 |
| `env/` | Gymnasium `reset` / `step` 인터페이스 제공 |
| `agent/` | PPO 학습, 체크포인트·평가 콜백, 모델 저장/로드 |
| `inference/` | 추론 실행 후 `external/output/`에 결과 저장 |
| `api/` | FastAPI 백엔드 — React UI용 REST API |
| `frontend/` | React TypeScript UI — 간트 차트·분석·학습/추론 제어 |

---

## 강화학습 구현

### 문제 정의

- **상황**: 초기 스케줄링이 끝난 뒤, 각 EQP 큐에 대기 중인 LOT 중 하나를 선택해 재배정
- **결정 시점**: EQP가 idle이 되고 배정 가능한 LOT이 있을 때
- **목표**: 동일 공정(OPER)·동일 제품(PROD) 연속 가공을 유도하고, Idle·전환 횟수를 줄임

### MDP 구성

| 요소 | 내용 |
|------|------|
| **알고리즘** | PPO (`stable-baselines3`, `MlpPolicy`) |
| **행동 공간** | `Discrete(max_queue_size)` — 현재 idle EQP 큐에서 LOT 인덱스 선택 (기본 20) |
| **관측 공간** | `Box(0, 1, shape=(obs_dim,))` — 이분 그래프 기반 고정 크기 벡터 |
| **에피소드 종료** | 모든 LOT 배정 완료 (`terminated`) 또는 시뮬레이션 시간 초과 (`truncated`) |

관측 차원 (`config.env` 기준, O=15, P=10, M=10):

```
obs_dim = O*P*5 + O*(4 + M*2) + 6 = 1,026
```

### 관측 벡터 구조 (`simulator.get_observation`)

이분 그래프 집계 방식으로 EQP/LOT 수와 무관하게 고정 크기를 유지합니다.

| 그룹 | 크기 | 내용 |
|------|------|------|
| **Group A** | O×P×3 | WIP 노드 — (OPER, PROD)별 대기 LOT 수·수량·우선순위 |
| **Group B** | O×4 | EQP 노드 — OPER별 처리 가능 설비 수, idle 비율, 잔여시간, 전환율 |
| **Group E** | O×M×2 | WIP×EQP 엣지 — (OPER, EQP) 조합별 평균 처리시간·호환 LOT 수 |
| **Group C** | O×P×2 | 계획 노드 — (OPER, PROD) 달성률·우선순위 |
| **Group D** | 6 | 컨텍스트 — 경과 시간, 전체 달성률, 현재 EQP 이전 OPER/PROD 등 |

처리시간은 LOT 단독 속성이 아니라 **(LOT, EQP) 조합**으로 `proc_time_matrix`에 저장됩니다.

### 보상 함수 (`simulator.assign_lot`)

`config.reward` 가중치로 즉각 보상을 계산합니다.

| 항목 | 가중치 (기본값) | 설명 |
|------|----------------|------|
| 동일 OPER 연속 | `+2.0` | 이전 공정과 같으면 보너스 |
| 동일 PROD 연속 | `+1.0` | 이전 제품과 같으면 보너스 |
| Idle 패널티 | `-0.5`/분 | EQP 대기 시간에 비례해 감점 |
| LOT 완료 | `+1.0` × (wf_qty/25) | 완료 시 수량 기반 보상 |
| 계획 달성 | `+5.0` | (설정값, 향후 확장용) |

공정·제품이 바뀌면 `oper_switches`, `prod_switches` 통계가 증가합니다.

### 에이전트 (`agent/rl_agent.py`)

```python
agent = SchedulingAgent()
agent.train(env_data)          # PPO 학습
agent.save()                   # models/scheduling_rl.zip
agent = SchedulingAgent.load() # 모델 로드
action = agent.predict(obs)    # 행동 예측
metrics = agent.evaluate(env_data, n_episodes=5)
```

학습 시 콜백:
- **EvalCallback** — 주기적 평가, `models/best/`에 최적 모델 저장
- **CheckpointCallback** — `models/checkpoints/`에 체크포인트 저장

PPO 하이퍼파라미터 (`config.rl`):

| 파라미터 | 기본값 |
|----------|--------|
| `learning_rate` | 3e-4 |
| `n_steps` | 2,048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `total_timesteps` | 200,000 |
| `eval_freq` | 10,000 |

### 환경 루프 (`env/scheduling_env.py`)

```
reset() → sim 초기화, 첫 관측 반환
  ↓
step(action):
  1. current_idle_eqp() 로 결정 대기 EQP 확인
  2. available_lots() 큐에서 action 인덱스로 LOT 선택
  3. assign_lot() → 보상 계산, 스케줄 기록
  4. save_history_step() → UI 재생용 히스토리 저장
  5. is_done() / 시간 초과 확인 → 다음 관측 반환
```

`action_masks()` 메서드가 있어 **MaskablePPO**(`sb3-contrib`) 확장도 가능합니다.

### 데이터 흐름

```
external/input/*.json
    → loader.load_data() + validate_data()
    → preprocessor.preprocess()  → env_data
    → SchedulingEnv(env_data) / SchedulingSimulator
    → SchedulingAgent.train() / run_inference()
    → external/output/result.json
```

입력 JSON 4종:
- `schedule.json` — 초기 스케줄
- `availability.json` — EQP별 LOT 투입 가능 여부·수량
- `plan.json` — (제품, 공정) 계획 수량
- `flow.json` — 제품별 공정 순서

---

## UI (React + FastAPI)

Streamlit 대신 **React TypeScript + FastAPI** 구조로 동작합니다.

| 구성 | 포트 | 설명 |
|------|------|------|
| FastAPI (`api/server.py`) | 8000 | 데이터·학습·추론 API |
| React (`frontend/`) | 5173 | 간트 차트, WIP/달성률, 시뮬레이션 재생 |

```bash
# 일괄 실행 (API + 프론트엔드)
python main.py ui

# 개별 실행
uvicorn api.server:app --reload --port 8000
cd frontend && npm install && npm run dev
```

UI 기능:
- **학습 모드** — 파라미터 설정, PPO 학습, 평가 지표 표시
- **추론 모드** — Post-Scheduling 실행, 스텝별 간트 재생, 초기 vs Post KPI 비교

---

## 사용 방법

```bash
pip install -r requirements.txt
```

데이터 경로는 `external/dataset/{FAC_ID}/{train|test|infer}/{RULE_TIMEKEY}/input/` 형식입니다.  
예: `FAC001/train/20260619070000`

### 데이터 생성

```bash
# 단일 train 샘플 (현재 시각 RULE_TIMEKEY)
python main.py sample --fac-id FAC001 --split train

# pacing 검증용 미니 시나리오
python main.py sample -s pacing_steady --fac-id FAC001 --split train
python main.py sample -s pacing_steady --fac-id FAC001 --split test

# train/test/infer 골격 + 샘플 (train 3기간 + test 1기간)
python main.py sample --bootstrap --fac-id FAC001

# RULE_TIMEKEY 구간 일괄 생성
python main.py sample -s default --fac-id FAC001 --split train --from-date 20260601 --to-date 20260607
```

### DB 조회 (Oracle)

```bash
python main.py fetch --fac-id FAC001 --split train --from-date 20260601 --to-date 20260607
python main.py fetch --fac-id FAC001 --split test --snapshot 20260608070000
```

### 학습 / 추론 (개별)

`-i` / `--input`은 서브커맨드 **앞·뒤** 모두 가능합니다.

```bash
# train 폴더 지정 후 학습
python main.py train -i FAC001/train/20260619070000 --timesteps 50000
python main.py -i FAC001/train/20260619070000 train --timesteps 50000

# test 폴더로 RL 추론 (결과: infer/output/ 또는 test/output/)
python main.py infer -i FAC001/test/20260619070000 --algorithm rl
python main.py infer -i FAC001/test/20260619070000 --algorithm minprogress
```

### 한 번에: train 학습 → test 추론 (`run`)

```bash
# 단일 기간 (빠른 스모크 테스트)
python main.py run -i FAC001/train/20260619070000 --timesteps 5000

# test 폴더 직접 지정 + 휴리스틱 비교
python main.py run -i FAC001/train/20260619070000 --test FAC001/test/20260619070000 --compare

# FAC 전체 train/test (기존 dataset 폴더 전부)
python main.py run --all --fac-id FAC001 --timesteps 200000

# 샘플 생성부터 전체 워크플로우
python main.py run --all --bootstrap --scenario default --timesteps 200000

# 기간 필터 (train/test RULE_TIMEKEY 구간만)
python main.py run --all --from-date 20260601 --to-date 20260607 --timesteps 200000
```

| 명령 | 설명 |
|------|------|
| `run -i ...` | 단일 train 학습 → 대응 test 1개 추론 |
| `run --all` | train **모든 기간** VecEnv 학습 → test **모든 기간** 추론 |
| `run --bootstrap` | run 전에 train 3 + test 1 샘플 자동 생성 |
| `run --compare` | test마다 PPO + Min-Progress 실행 |
| `all` | bootstrap + train + **infer** 추론 (test 평가 아님) |

test 추론 결과는 각 `external/dataset/FAC001/test/{RULE_TIMEKEY}/output/`에 저장됩니다.

### UI

```bash
python main.py ui
```

- **학습** — train 기간 단일/구간/복수 선택, PPO 학습
- **테스트** — test 데이터셋 전체 벤치마크 (RL vs 휴리스틱)
- **추론** — 단일 폴더 Post-Scheduling 실행·차트 확인

개별 실행:

```bash
uvicorn api.server:app --reload --port 8000
cd frontend && npm install && npm run dev
```

입력 폴더 기본값 변경 (선택):

```bash
# Windows
set SCHEDULING_INPUT=FAC001/train/20260619070000

# Linux / macOS
export SCHEDULING_INPUT=FAC001/train/20260619070000
```

---

## 의존성

- **gymnasium** — RL 환경 표준 인터페이스
- **stable-baselines3** — PPO 구현
- **sb3-contrib** — MaskablePPO (선택 확장)
- **torch** — 신경망 백엔드
- **fastapi / uvicorn** — REST API 백엔드
- **numpy / pandas / scipy** — 수치·데이터 처리

### Frontend (frontend/)

- **react / react-dom** — UI 프레임워크
- **plotly.js / react-plotly.js** — 간트·KPI 차트
- **vite** — 빌드·개발 서버
