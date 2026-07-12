# 프로덕션 배포 가이드

## 개요

이 문서는 pjt_scheduling을 프로덕션 환경에 배포하기 위한 단계별 가이드입니다.

---

## 1. 사전 준비

### 1.1 환경 확인

```bash
# Python 버전 확인 (3.11+ 권장)
python --version

# 의존성 설치
pip install -r requirements.txt

# 선택: 개발 도구
pip install pytest pytest-cov black flake8
```

### 1.2 데이터베이스 설정

```bash
# 설정 파일 작성
cp config/databases.yaml.example config/databases.yaml

# 설정 편집 (DB 접속 정보 입력)
# vi config/databases.yaml

# 연결 테스트
python main.py db-check

# DDL 실행 (최초 1회)
python main.py db-load --ddl-only
```

### 1.3 환경 변수 설정

```bash
# .env 파일 작성
cp .env.example .env

# 필수 환경변수 확인
# DB_CONFIG=config/databases.yaml
# COLLECTOR_FAC_ID=FAC001
```

---

## 2. 모델 준비

### 2.1 데이터 수집

```bash
# 이전 3개 기간 데이터 수집
python main.py collect --facid FAC001 --prevcnt 3 --once

# 또는 DB에서 직접 수집
python main.py collect --facid FAC001 --split train --once
```

### 2.2 모델 학습

```bash
# 학습 실행
python main.py train --facid FAC001 --prevcnt 3

# 학습 상태 확인
ls -la models/
```

### 2.3 테스트 데이터 준비

```bash
# 테스트 데이터 생성
python main.py sample --facid FAC001 --bootstrap

# 또는 DB에서 수집
python main.py collect --facid FAC001 --split test --once
```

---

## 3. API 서버 배포

### 3.1 로컬 테스트

```bash
# 개발 모드 실행
python -m uvicorn api.server:app --reload --host 127.0.0.1 --port 8001

# 다른 터미널에서 테스트
curl http://localhost:8001/api/health
```

### 3.2 프로덕션 서버 실행

#### 단일 워커 (낮은 트래픽)

```bash
python api/start_production.py --host 0.0.0.0 --port 8001
```

#### 다중 워커 (높은 트래픽, CPU 개수 기준)

```bash
# CPU 개수 확인
nproc

# 워커 수 = CPU 개수 * 2 + 1 (권장)
python api/start_production.py --host 0.0.0.0 --port 8001 --workers 8
```

#### Systemd 서비스 등록

```bash
# /etc/systemd/system/scheduling-api.service 작성
cat > /etc/systemd/system/scheduling-api.service <<EOF
[Unit]
Description=Scheduling RL API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/scheduling
Environment="PATH=/opt/scheduling/venv/bin"
ExecStart=/opt/scheduling/venv/bin/python api/start_production.py --host 0.0.0.0 --port 8001 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 시작
sudo systemctl start scheduling-api
sudo systemctl enable scheduling-api

# 상태 확인
sudo systemctl status scheduling-api
```

### 3.3 HTTPS 설정

```bash
# Let's Encrypt 인증서 생성 (Certbot)
certbot certonly --standalone -d scheduling.example.com

# 서버 시작 (HTTPS)
python api/start_production.py \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile /etc/letsencrypt/live/scheduling.example.com/privkey.pem \
  --ssl-certfile /etc/letsencrypt/live/scheduling.example.com/fullchain.pem
```

---

## 4. Docker 배포

### 4.1 Docker 이미지 빌드

```bash
# 이미지 빌드
docker build -t scheduling-api:latest .

# 이미지 확인
docker images | grep scheduling
```

### 4.2 Docker 컨테이너 실행

#### 단일 컨테이너

```bash
docker run -d \
  --name scheduling-api \
  -p 8001:8001 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e DB_CONFIG=config/databases.yaml \
  scheduling-api:latest
```

#### Docker Compose

```bash
# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f api

# 헬스 체크
curl http://localhost:8001/api/health
```

### 4.3 Docker 레지스트리에 푸시

```bash
# 레지스트리 로그인
docker login registry.example.com

# 태그 지정
docker tag scheduling-api:latest registry.example.com/scheduling-api:latest

# 푸시
docker push registry.example.com/scheduling-api:latest
```

---

## 5. Kubernetes 배포

### 5.1 Kubernetes 매니페스트 작성

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scheduling-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scheduling-api
  template:
    metadata:
      labels:
        app: scheduling-api
    spec:
      containers:
      - name: api
        image: registry.example.com/scheduling-api:latest
        ports:
        - containerPort: 8001
        env:
        - name: DB_CONFIG
          value: "/config/databases.yaml"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/health/detailed
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 2
        volumeMounts:
        - name: config
          mountPath: /config
        - name: data
          mountPath: /data
        - name: models
          mountPath: /models
      volumes:
      - name: config
        configMap:
          name: scheduling-config
      - name: data
        persistentVolumeClaim:
          claimName: scheduling-data
      - name: models
        persistentVolumeClaim:
          claimName: scheduling-models

---
apiVersion: v1
kind: Service
metadata:
  name: scheduling-api
spec:
  selector:
    app: scheduling-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
  type: LoadBalancer
```

### 5.2 Kubernetes 배포

```bash
# 설정 맵 생성
kubectl create configmap scheduling-config \
  --from-file=config/databases.yaml

# 퍼시스턴트 볼륨 클레임 생성 (별도 구성)
kubectl apply -f pvc.yaml

# 배포
kubectl apply -f deployment.yaml

# 상태 확인
kubectl get pods
kubectl get svc scheduling-api
```

---

## 6. 프론트엔드 배포

### 6.1 React 빌드

```bash
cd frontend

# 프로덕션 빌드
npm run build

# 빌드 결과
ls dist/
```

### 6.2 정적 파일 서빙

#### Nginx 설정

```nginx
server {
    listen 80;
    server_name example.com;

    root /var/www/scheduling-ui;
    index index.html;

    # SPA 라우팅
    try_files $uri $uri/ /index.html;

    # API 프록시
    location /api/ {
        proxy_pass http://api:8001;
    }
}
```

#### 파일 배포

```bash
# 빌드된 파일 복사
sudo cp -r frontend/dist/* /var/www/scheduling-ui/

# 권한 설정
sudo chown -R www-data:www-data /var/www/scheduling-ui
```

---

## 7. 모니터링 및 로깅

### 7.1 로그 수집

```bash
# Systemd 로그 확인
journalctl -u scheduling-api -f

# Docker 로그 확인
docker logs -f scheduling-api

# 로그 파일 위치
tail -f /var/log/scheduling-api/access.log
tail -f /var/log/scheduling-api/error.log
```

### 7.2 헬스 체크 모니터링

```bash
# 주기적 헬스 체크
watch -n 30 'curl -s http://localhost:8001/api/health | jq .'

# 모니터링 스크립트 (향후 추가)
# - Prometheus 메트릭
# - Grafana 대시보드
# - ELK 로그 스택
```

### 7.3 알림 설정

```bash
# 헬스 체크 실패 시 알림 (예: Slack)
cat > /usr/local/bin/health-check.sh <<'EOF'
#!/bin/bash
RESPONSE=$(curl -s http://localhost:8001/api/health)
STATUS=$(echo $RESPONSE | jq -r '.status')

if [ "$STATUS" != "ok" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"Scheduling API Alert: $STATUS\"}" \
        $SLACK_WEBHOOK_URL
fi
EOF

# Cron에 등록
# */5 * * * * /usr/local/bin/health-check.sh
```

---

## 8. 보안 체크리스트

배포 전 다음을 확인하세요:

- [ ] HTTPS/TLS 활성화
- [ ] 방화벽 규칙 설정 (포트 8001 외부 개방 차단)
- [ ] API 인증 구현 (API 키 또는 JWT)
- [ ] CORS 설정 확인
- [ ] SQL 인젝션 취약점 점검
- [ ] 의존성 보안 업데이트 확인
  ```bash
  pip audit
  npm audit
  ```
- [ ] 환경 변수 보안 (DB 암호 암호화)
- [ ] 로그에 민감 정보 제외
- [ ] 정기적 백업 계획

---

## 9. 배포 확인

배포 후 다음을 확인하세요:

```bash
# 1. 헬스 체크
curl -v http://localhost:8001/api/health

# 2. API 문서
curl http://localhost:8001/docs

# 3. 데이터 로드
curl http://localhost:8001/api/data/summary

# 4. 모델 상태
curl http://localhost:8001/api/model/status

# 5. 추론 실행
curl -X POST http://localhost:8001/api/inference \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"earliest_st"}'
```

---

## 10. 문제 해결

### API 시작 실패

```bash
# 포트 이미 사용 중
lsof -i :8001
kill -9 <PID>

# DB 연결 실패
python main.py db-check

# 모델 로드 실패
python -c "from agent.rl_agent import SchedulingAgent; print(SchedulingAgent().model_exists())"
```

### 높은 메모리 사용

```bash
# 프로세스 모니터링
ps aux | grep start_production

# 메모리 프로파일링
python -m memory_profiler api/server.py
```

### 느린 응답

```bash
# 데이터베이스 쿼리 성능
python main.py db-check --verbose

# API 응답 시간 측정
curl -w "Total: %{time_total}s\n" http://localhost:8001/api/health
```

---

## 11. 참고 자료

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Uvicorn Production Settings](https://www.uvicorn.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Security Best Practices](https://owasp.org/www-project-top-ten/)
