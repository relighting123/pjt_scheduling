# 헬스 체크 엔드포인트 가이드

프로덕션 환경에서 시스템의 상태를 실시간으로 모니터링하기 위한 헬스 체크 엔드포인트입니다.

## 엔드포인트

### 1. 간단 헬스 체크 (Liveness Probe)

```http
GET /api/health
```

**목적**: 서버 및 주요 컴포넌트의 기본 상태 확인

**응답 예**:
```json
{
  "status": "ok",
  "timestamp": "2026-07-12T12:49:59.107586+00:00",
  "components": {
    "api": {"status": "healthy"},
    "database": {"status": "healthy"},
    "model": {"status": "not_found", "exists": false},
    "input_folder": {"status": "healthy", "path": "..."},
    "output_folder": {"status": "healthy", "path": "..."},
    "current_input_folder": {
      "status": "healthy",
      "input_folder": "FAC001/train/20260627070000",
      "fac_id": "FAC001"
    }
  }
}
```

**상태 코드**:
- `200 OK`: 서버가 정상 작동 중
- `503 Service Unavailable`: 서비스 이용 불가

**상태 의미**:
- `ok`: 모든 컴포넌트 정상
- `degraded`: 일부 컴포넌트 문제 있음 (예: DB 미연결)

---

### 2. 상세 헬스 체크 (Readiness Probe)

```http
GET /api/health/detailed
```

**목적**: 모든 구성 요소의 상세 정보 및 시스템 리소스 조회

**응답 예**:
```json
{
  "status": "degraded",
  "timestamp": "2026-07-12T12:50:21.777098+00:00",
  "components": {
    "api": {"status": "healthy"},
    "database": {
      "status": "unhealthy",
      "error": "DB alias 'main'에 접속 정보가 없습니다."
    },
    "model": {
      "status": "not_found",
      "exists": false,
      "model_dir": "C:\\Users\\jaehw\\Desktop\\dev\\pjt_scheduling\\models"
    },
    "input_folder": {
      "status": "healthy",
      "path": "C:\\Users\\jaehw\\Desktop\\dev\\pjt_scheduling\\data\\dataset\\FAC001\\train\\20260627070000\\input",
      "exists": true
    },
    "output_folder": {
      "status": "healthy",
      "path": "C:\\Users\\jaehw\\Desktop\\dev\\pjt_scheduling\\data\\dataset\\FAC001\\train\\20260627070000\\output",
      "exists": true
    },
    "current_input_folder": {
      "status": "healthy",
      "input_folder": "FAC001/train/20260627070000",
      "fac_id": "FAC001"
    }
  },
  "system": {
    "cpu_percent": 80.2,
    "memory_percent": 88.4,
    "disk_percent": 55.1
  }
}
```

**시스템 정보**:
- `cpu_percent`: CPU 사용률 (%)
- `memory_percent`: 메모리 사용률 (%)
- `disk_percent`: 디스크 사용률 (%)

---

## 컴포넌트 상태 설명

| 컴포넌트 | 역할 | 문제 발생 시 해결 방법 |
|---------|------|----------------------|
| **api** | FastAPI 서버 | 일반적으로 항상 정상. 심각한 서버 오류 시만 표시 |
| **database** | Oracle DB 연결 | `config/databases.yaml` 설정 확인, DB 서버 상태 확인 |
| **model** | RL 모델 파일 | `python main.py train` 으로 모델 학습 필요 |
| **input_folder** | 입력 데이터 폴더 | 폴더 경로 및 권한 확인, 데이터 수집(`collect`) 필요 |
| **output_folder** | 출력 결과 폴더 | 폴더 경로 및 권한 확인 |
| **current_input_folder** | 현재 선택 입력 폴더 | 설정 확인, API `/api/config/input` 호출 |

---

## Kubernetes 헬스 체크 설정

### Liveness Probe (서버 생존 확인)

```yaml
livenessProbe:
  httpGet:
    path: /api/health
    port: 8001
    scheme: HTTP
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe (트래픽 수신 가능 확인)

```yaml
readinessProbe:
  httpGet:
    path: /api/health/detailed
    port: 8001
    scheme: HTTP
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 2
```

---

## 모니터링 및 알림 설정

### Prometheus 메트릭 (향후 추가 예정)

```python
# api/server.py 에 다음 추가 가능
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'API request duration')

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### 기본 모니터링 항목

- **응답 시간**: `/api/health` 응답 시간 < 100ms
- **상태 체크**: 5분마다 헬스 체크 호출
- **DB 연결**: DB 상태가 "unhealthy" 인 경우 알림
- **모델 누락**: `model.exists == false` 인 경우 경고
- **시스템 리소스**: CPU/메모리/디스크 사용률 모니터링

---

## 프로덕션 배포 체크리스트

배포 전 다음을 확인하세요:

- [ ] `config/databases.yaml` 작성 및 검증
  ```bash
  python main.py db-check
  ```

- [ ] RL 모델 학습 및 저장
  ```bash
  python main.py train --facid FAC001 --prevcnt 3
  ```

- [ ] 입력/출력 폴더 권한 확인
  ```bash
  ls -la data/dataset/FAC001/
  ```

- [ ] 헬스 체크 엔드포인트 테스트
  ```bash
  curl http://localhost:8001/api/health
  ```

- [ ] API 서버 시작
  ```bash
  python api/start_production.py --host 0.0.0.0 --port 8001 --workers 4
  ```

---

## 문제 해결

### 데이터베이스 연결 실패

**증상**: `database` 상태가 "unhealthy"

**해결**:
1. DB 설정 확인:
   ```bash
   python -m data.db_registry --json
   ```

2. 설정 파일 작성:
   ```bash
   cp config/databases.yaml.example config/databases.yaml
   # config/databases.yaml 편집
   ```

3. 연결 테스트:
   ```bash
   python main.py db-check
   ```

### 모델 파일 없음

**증상**: `model` 상태가 "not_found"

**해결**:
```bash
# 데이터 수집
python main.py collect --facid FAC001 --once

# 모델 학습
python main.py train --facid FAC001 --prevcnt 3

# 모델 확인
ls -la models/
```

### 입력 폴더 접근 불가

**증상**: `input_folder` 상태가 "unhealthy"

**해결**:
```bash
# 폴더 생성
mkdir -p data/dataset/FAC001/train/

# 데이터 수집
python main.py collect --facid FAC001 --once

# 권한 확인
ls -la data/dataset/FAC001/
```

---

## API 시작 명령어

### 개발 환경

```bash
python -m uvicorn api.server:app --reload --host 127.0.0.1 --port 8001
```

### 프로덕션 환경 (단일 워커)

```bash
python api/start_production.py --host 0.0.0.0 --port 8001
```

### 프로덕션 환경 (다중 워커)

```bash
python api/start_production.py --host 0.0.0.0 --port 8001 --workers 4
```

### HTTPS 지원

```bash
python api/start_production.py \
  --host 0.0.0.0 \
  --port 8001 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

---

## 참고자료

- [FastAPI Health Check](https://fastapi.tiangolo.com/deployment/concepts/#health-checks)
- [Kubernetes Liveness/Readiness Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Docker HEALTHCHECK](https://docs.docker.com/engine/reference/builder/#healthcheck)
