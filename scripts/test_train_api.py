"""학습 API 통합 테스트 (서버 실행 중: uvicorn api.server:app --port 8000)"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path: str):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return r.status, json.loads(r.read())


def post(path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read())


def main() -> int:
    try:
        code, health = get("/api/health")
        print("health", code, health)
    except urllib.error.URLError as exc:
        print("API 서버 없음:", exc)
        print("실행: python -m uvicorn api.server:app --host 127.0.0.1 --port 8000")
        return 1

    _, cfg = get("/api/config")
    folder = cfg.get("input_folder", "FAC001/train")
    print("input_folder", folder)

    try:
        code, resp = post("/api/train/start", {
            "total_timesteps": 2048,
            "learning_rate": 0.0003,
            "w_same_oper": 1.0,
            "w_idle_per_min": -0.5,
            "input_folder": folder,
        })
        print("train/start", code, resp)
    except urllib.error.HTTPError as exc:
        print("train/start FAIL", exc.code, exc.read().decode())
        return 1

    for i in range(60):
        try:
            code, status = get("/api/train/status")
        except urllib.error.HTTPError as exc:
            print(f"train/status FAIL iter={i}", exc.code, exc.read().decode()[:500])
            return 1
        except urllib.error.URLError as exc:
            print(f"train/status URLError iter={i}", exc)
            return 1

        st = status.get("status")
        print(
            f"  [{i}] status={st} progress={status.get('progress'):.3f} "
            f"steps={status.get('timesteps')}/{status.get('total_timesteps')}"
        )
        if st in ("completed", "failed", "idle"):
            if status.get("error"):
                print("ERROR:", status["error"])
            if status.get("logs"):
                print("last log:", status["logs"][-1]["message"])
            break
        time.sleep(2)
    else:
        print("timeout waiting for training")
        return 1

    return 0 if st == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
