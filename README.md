# pjt_scheduling

반도체 **Scheduling** 문제를 강화학습(RL)으로 해결하는 프로젝트입니다.  
`discrete_arrange`·`abstract_arrange`를 입력으로 받아, EQP가 idle이 될 때마다 **(PPK, OPER) + EQP**를 선택하고 LOT은 규칙으로 자동 배정하여 **공정·제품 전환, Idle, LOT_CD/TEMP 전환, 계획 페이싱**을 최적화합니다.

---

## 폴더 구조

```
pjt_scheduling/
├── main.py                 # CLI (sample / fetch / train / infer / run / ui)
├── config.py               # 경로, 환경, RL, 보상 파라미터
├── requirements.txt
│
├── agent/
│   ├── rl_agent.py         # MaskablePPO 학습·저장·예측
│   ├── minprogress_agent.py
│   ├── earliest_st_agent.py
│   └── registry.py         # 알고리즘 목록 (RL / 휴리스틱)
│
├── env/
│   └── scheduling_env.py   # Gymnasium 환경 (MultiDiscrete + action mask)
│
├── simulation/
│   └── simulator.py        # DES 엔진, WIP 풀, 관측·보상·전환·tool
│
├── data/
│   ├── dataset/            # {FAC_ID}/{train|test|infer}/{RULE_TIMEKEY}/input|output
│   ├── loader/             # 입력 fetch + 전처리 (Oracle SQL → JSON → env_data)
│   │   ├── fetch.py
│   │   └── preprocess.py
│   ├── writer/             # 출력 적재 (추론 → output JSON + SQL)
│   ├── sql.example/        # Oracle → JSON 쿼리 예시
│   ├── sql/                # 환경별 실행 SQL (git 제외)
│   └── generator.py        # 샘플 시나리오·dataset 생성
│
├── inference/
│   └── runner.py           # 추론 실행·히스토리 기록
│
├── api/
│   ├── server.py           # FastAPI REST API
│   └── serializers.py
│
├── frontend/               # React + TypeScript (Vite)
│   ├── src/pages/          # Dataset / Train / Test / Inference
│   ├── src/components/     # ChartPanel, ArrangeTable, …
│   └── src/lib/charts.ts   # Plotly 차트 빌더
│
├── tests/
└── models/                 # 학습 체크포인트 (.zip, git 제외)
```

| 모듈 | 역할 |
|------|------|
| `data/loader` | Oracle → input JSON (fetch) + JSON → env_data (preprocess) |
| `data/writer` | 추론 결과 → output JSON + 적재 SQL |
| `data/generator` | 샘플·dataset 생성 |
| `simulation/` | EQP/LOT/WIP 상태, 결정 시점, 관측·보상 |
| `env/` | Gymnasium `reset` / `step`, `action_masks()` |
| `agent/` | MaskablePPO 및 휴리스틱 에이전트 |
| `inference/` | 단일·비교 추론 실행 (`data.writer`로 결과 기록) |
| `api/` + `frontend/` | 학습·테스트·추론 UI |

---

## 도메인 개요

### 데이터 3층

| 레이어 | 단위 | 역할 |
|--------|------|------|
| **discrete_arrange** | `(EQP_ID, LOT_ID, OPER_ID)` | Actual feasible 조합, LOT 현재 공정, `proc_time_matrix`, `eqp_oper_cap` |
| **abstract_arrange** | `(PPK, OPER, EQP_MODEL_CD, ST)` | route·평균 ST **템플릿** (전처리 시 inventory 생성) |
| **WIP 풀** (런타임) | `(PPK, OPER)` | LOT 수 `+1`/`-1`, `oper_in_time` — 전공정 완료·배정 시 갱신 |

- **LOT_CD / TEMP**: `batch_info.json` **(PPK, OPER)별** 권장. 없으면 `lot_master.json`(LOT별) 또는 PPK 추정. EQP 직전 값과 다르면 **전환 60분** + tool `(LOT_CD, EQP_MODEL)` cap 검사.
- **EQP_MODEL_CD**: 장비 **군**(1:N). 여러 `EQP_ID`가 동일 MODEL을 공유. discrete/abstract route/split 입력은 `EQP_MODEL_CD`, tool은 `EQP_MODEL` 단위, 배정·conversion은 EQP_ID 단위.

### 결정 흐름 (한 step)

1. idle이면서 assignable한 EQP가 있을 때까지 시간 전진  
2. 에이전트: `(PPK×OPER bucket, EQP index)` — **LOT은 선택하지 않음**  
3. `get_feasible_assignments()`로 `(flat, eqp_idx)` feasible 목록 계산 → mask·resolve  
4. `assign_ppk_oper()` → `_auto_select_lot()` → conversion/tool → WIP `-1`

---

## 강화학습 구현

### MDP 구성

| 요소 | 내용 |
|------|------|
| **알고리즘** | **MaskablePPO** (`sb3-contrib`, `MlpPolicy`) |
| **행동 공간** | `MultiDiscrete([O×P, M])` = `[150, 10]` — (PPK×OPER flat, EQP index) |
| **LOT 배정** | 우선순위 · `oper_in_time` · seq 규칙으로 **자동** |
| **관측** | `Box(0, 1, shape=(6060,))` |
| **종료** | assignable WIP·Actual 없음 + busy EQP 없음 (`terminated`), 또는 horizon (`truncated`) |

`O=15, P=10, M=10, K=4, F=10` (`config.env`):

```
obs_dim = 6 + O×P×K×F + M×5 + 4
        = 6 + 15×10×4×10 + 10×5 + 4
        = 6060
```

### 관측 벡터 (`simulator.get_observation`)

| 블록 | 크기 | 내용 |
|------|------|------|
| **Global** | 6 | 경과 시간, soft cutoff 잔여, lot pool 비율, 완료율, conversion EQP 비율, tool utilization |
| **Bucket** | O×P×K×F = 6000 | (OPER, PPK, **EQP_MODEL**) — valid, WIP 비율, min_end, throughput, same_ppk, takt, ST, urgency |
| **EQP local** | M×5 = 50 | EQP별 idle/busy, prev lot_cd/temp, busy 잔여시간 |
| **Context** | 4 | 직전 배정 PPK/OPER/EQP/LOT_CD 인코딩 |

- Bucket **K축 = EQP_MODEL**. Action에는 MODEL 축 없음 → **EQP 선택으로 MODEL이 암묵 결정**.
- Bucket WIP는 `(PPK, OPER)` 풀 집계; **EQP별 feasible**은 `get_feasible_assignments()`에서 별도 계산.

### 보상 (`config.reward`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `w_same_oper` | +2.0 | 동일 OPER 연속 |
| `w_same_prod` | +0.5 | 동일 PPK + 해당 PPK feasible 재공 있을 때 |
| `w_prod_switch` | +0.8 | 이전 PPK 재공 고갈 후 전환 |
| `w_idle_per_min` | -0.5 | EQP idle 시간 |
| `w_completion` | +1.0 | LOT 투입 (wf_qty/25) |
| `w_plan_hit` | +5.0 | 계획 gap 감소 |
| `w_pacing` | +3.0 | 계획 직선 대비 누적 편차 shaping |
| `w_conversion` | -30.0 | LOT_CD/TEMP 전환 1회 |
| `w_late_finish` | -2.0 | soft cutoff(05:00) 이후 END_TM |

시뮬 horizon: **07:00 → 익일 07:00** (1440분), soft cutoff **05:00** (1320분).

### 에이전트

```python
from agent.rl_agent import SchedulingAgent

agent = SchedulingAgent()
agent.train(env_data, total_timesteps=200_000)
agent.save()
agent = SchedulingAgent.load()
action = agent.predict(obs, action_masks=env.action_masks())
```

휴리스틱: `minprogress`, `earliest_st` (`agent/registry.py`).

### 환경 (`env/scheduling_env.py`)

```
reset() → sim + abstract WIP 풀 초기화
step(action=[ppk_oper_flat, eqp_idx]):
  1. idle assignable EQP까지 _advance_to_next_decision()
  2. get_feasible_assignments() → action_masks / resolve
  3. assign_ppk_oper(eqp, ppk, oper) → LOT 자동, WIP -1
  4. save_history_step() (UI 재생)
```

`action_masks()`: `[ppk_oper mask (O×P), eqp mask (M)]` — feasible pair의 **OR** (교차곱 mask 아님, invalid 조합은 `_resolve_action_to_feasible` 보정).

---

## 입력 JSON

경로: `data/dataset/{FAC_ID}/{train|test|infer}/{RULE_TIMEKEY}/input/`

| 파일 | 설명 |
|------|------|
| `discrete_arrange.json` | `(EQP, LOT)` Actual + **OPER_ID**(현재 공정), WF_QTY, ST, EQP_MODEL_CD |
| `abstract_arrange.json` | `(PPK, OPER, EQP_MODEL_CD, ST)` route (없으면 discrete에서 집계 생성) |
| `plan.json` | (PPK, OPER) 계획량·우선순위 |
| `flow.json` | PPK별 공정 순서 (`OPER_SEQ`) |
| `batch_info.json` | **(PPK, OPER) → LOT_CD, TEMP** — conversion/tool (Oracle: `batch_info.sql`) |
| `lot_master.json` | LOT별 LOT_CD, TEMP (레거시·batch_info 없을 때 보조) |
| `tool_capacity.json` | `(LOT_CD, EQP_MODEL)` 동시 가공 상한 |
| `eqp_initial_state.json` | EQP별 초기 LOT_CD/TEMP/직전 PPK·OPER |
| `split.json` | wafer lot split 규칙 (`EQP_MODEL_CD`, 선택) |

### Takt pacing 검증 시나리오 (`data/pacing_scenarios.py`)

| ID | 용도 |
|----|------|
| `takt_1p1o` | 1제품·1공정·2EQP·WIP100 — 직선 생산 기본 |
| `takt_2stage` | 2단 파이프라인 (OPER1 WIP → OPER2 유입) |
| `takt_wip_buffer` | 느린 upstream → WIP 축적 후 downstream |
| `takt_2ppk` | 2제품 경쟁·PPK_B 재공 부족 (test용) |

```bash
python scripts/run_takt_suite.py --timesteps 80000   # 생성→학습→추론→MAE 리포트
python main.py sample -s takt_1p1o --fac-id FAC_TAKT --split train
```

리포트: `data/dataset/FAC_TAKT/takt_suite_report.json`

```
loader → validate → preprocess → env_data
  ├── abstract_inventory      # PPK×OPER×MODEL 템플릿
  ├── abstract_wip_init       # (PPK,OPER) 초기 WIP·oper_in_time
  ├── proc_time_matrix        # (LOT, EQP) 처리시간
  └── eqp_oper_cap / abstract_route_map / …
```

---

## UI (React + FastAPI)

| 구성 | 포트 |
|------|------|
| FastAPI | 8000 |
| Vite dev | 5173 |

```bash
python main.py ui
```

| 탭 | 기능 |
|----|------|
| **Dataset** | 입력 폴더·요약 |
| **Train** | 기간 선택, MaskablePPO 학습, reward/loss 차트 |
| **Test** | test 전 기간 벤치마크 (RL vs 휴리스틱), KPI·간트 비교 |
| **Inference** | 데이터셋 선택, 단일 추론 / 알고리즘 비교, 시뮬 재생 |

**Inference 차트 UX**

- **차트 표시** 패널: 차트별 show/hide (localStorage 저장)
- **ChartPanel**: 제목 바 **드래그** 또는 **↗** → 별도 창 pop-out (스텝 재생 시 pop-out도 갱신)
- Arrange Actual / Abstract 테이블, soft cutoff·간트 시간축 설정
- **RL 구조 애니메이션**: `frontend/public/rl-training-animation.html` (dev 서버 실행 시 `/rl-training-animation.html`)

개별 실행:

```bash
uvicorn api.server:app --reload --port 8000
cd frontend && npm install && npm run dev
```

---

## CLI 사용법

```bash
pip install -r requirements.txt
```

### 데이터 생성

```bash
python main.py sample --fac-id FAC001 --split train
python main.py sample -s pacing_steady --fac-id FAC001 --split train
python main.py sample --bootstrap --fac-id FAC001
python main.py sample -s default --fac-id FAC001 --split train --from-date 20260601 --to-date 20260607
```

시나리오: `default`, `single_heavy_wip`, `pacing_steady`, `random` (`python main.py sample --list-scenarios`)

### DB 조회

```bash
mkdir -p data/sql
cp data/sql.example/*.sql data/sql/
# data/sql/*.sql 의 -- @db alias, 테이블명, WHERE 조건을 환경에 맞게 수정
python main.py fetch --fac-id FAC001 --split train --from-date 20260601 --to-date 20260607
```

`discrete_arrange`, `abstract_arrange`, `plan`, `flow`, `split`, `batch_info` SQL은 필수입니다.
`lot_master`, `tool_capacity`, `eqp_initial_state` SQL은 파일이 있으면 함께 수집하고 없으면 건너뜁니다.

### 학습 / 추론

`-i` / `--input`은 서브커맨드 앞·뒤 모두 가능.

```bash
python main.py train -i FAC001/train/20260619070000 --timesteps 50000
python main.py infer -i FAC001/test/20260619070000 --algorithm rl
python main.py infer -i FAC001/test/20260619070000 --algorithm minprogress
```

추론 결과는 `dataset/.../output/`에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `output.json` | **RTS_RSLT_INF** / **RTS_EQPCONVPLAN_INF** Oracle 스키마 |
| `output/sql/*.sql` | `output.json` 기반 INSERT (INF + HIS) |
| `result_full.json` | UI·디버그용 전체 schedule/history |

`output.json` 예시 구조:

```json
{
  "meta": { "FAC_ID", "RULE_TIMEKEY", "CRT_USER_ID", "ALGORITHM" },
  "RTS_RSLT_INF": [
    { "RULE_TIMEKEY", "LOT_CD", "TEMPER_VAL", "EQP_ID", "EQP_MODEL_CD",
      "SEQ_NO", "PLAN_PROD_KEY", "OPER_ID", "LOT_ID", "CARRIER_ID",
      "START_TIME", "END_TIME", "PRODUCE_QTY", "CRT_USER_ID" }
  ],
  "RTS_EQPCONVPLAN_INF": [ "... conversion 계획 ..." ]
}
```

JSON만 다시 SQL로 변환 (적재):

```bash
python main.py write-sql -i path/to/output.json
python main.py write-sql   # 현재 dataset output/output.json
```

### train → test 한 번에 (`run`)

```bash
python main.py run -i FAC001/train/20260619070000 --timesteps 5000
python main.py run -i FAC001/train/20260619070000 --test FAC001/test/20260619070000 --compare
python main.py run --all --fac-id FAC001 --timesteps 200000
python main.py run --all --bootstrap --scenario default --timesteps 200000
python main.py run --all --from-date 20260601 --to-date 20260607 --timesteps 200000
```

| 옵션 | 설명 |
|------|------|
| `run -i …` | 단일 train 학습 → 대응 test 1개 추론 |
| `run --all` | train 전 기간 VecEnv 학습 → test 전 기간 추론 |
| `run --bootstrap` | train 3 + test 1 샘플 자동 생성 후 run |
| `run --compare` | test마다 RL + Min-Progress |

환경 변수 (입력 폴더 기본값):

```bash
set SCHEDULING_INPUT=FAC001/train/20260619070000   # Windows
export SCHEDULING_INPUT=FAC001/train/20260619070000
```

---

## 테스트

```bash
python -m pytest tests/ -q
```

---

## 의존성

**Python**

- gymnasium, stable-baselines3, **sb3-contrib** (MaskablePPO)
- torch, fastapi, uvicorn, numpy, pandas

**Frontend**

- react, plotly.js, react-plotly.js, vite, typescript

---

## 참고

- 관측·action·reward 구조 변경 후 **기존 PPO 체크포인트는 호환되지 않음** → 재학습 필요.
- Bucket obs의 `valid`는 discrete `eqp_oper_cap` 기반; 배정 feasibility는 abstract route + WIP 풀 + tool까지 포함 (`get_feasible_assignments`).
