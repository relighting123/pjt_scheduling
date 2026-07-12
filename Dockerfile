# 프로덕션 API 서버 Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드
COPY . .

# 환경 변수
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001

EXPOSE ${PORT}

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# 서버 시작
CMD ["python", "api/start_production.py", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
