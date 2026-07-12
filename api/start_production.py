#!/usr/bin/env python
"""
프로덕션 API 서버 시작 스크립트

사용:
    python api/start_production.py --host 0.0.0.0 --port 8001 --workers 4
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="프로덕션 Scheduling API 서버 시작",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="바인드 호스트 주소"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="바인드 포트"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Gunicorn 워커 수"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="개발 모드 (자동 리로드)"
    )
    parser.add_argument(
        "--ssl-keyfile", default=None,
        help="SSL 개인키 파일 경로"
    )
    parser.add_argument(
        "--ssl-certfile", default=None,
        help="SSL 인증서 파일 경로"
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as e:
        print(f"오류: uvicorn 패키지가 필요합니다.", file=sys.stderr)
        print(f"설치: pip install uvicorn[standard]", file=sys.stderr)
        return 1

    config = {
        "app": "api.server:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "log_level": "info",
        "access_log": True,
    }

    if args.ssl_keyfile and args.ssl_certfile:
        config["ssl_keyfile"] = args.ssl_keyfile
        config["ssl_certfile"] = args.ssl_certfile
        proto = "https"
    else:
        proto = "http"

    if not args.reload and args.workers > 1:
        config["workers"] = args.workers
        config.pop("reload", None)

    print(f"[Scheduling API] {proto}://{args.host}:{args.port} 서버 시작")
    print(f"헬스 체크: {proto}://{args.host}:{args.port}/api/health")
    print(f"API 문서: {proto}://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(**config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
