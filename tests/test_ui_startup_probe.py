"""`python main.py ui` 기동 확인 경로 검증.

증상: 서버는 멀쩡히 떠 있는데 "[오류] API 서버가 시작되지 않았습니다"가 떴다.

원인: 기동 대기를 `/api/health`로 했는데, 이 엔드포인트는 Oracle 실제 연결
시도(`test_db_connection`)와 모델 zip 로드(`model_exists`)를 포함해서 응답이
초 단위가 될 수 있다(실측: 모델이 있으면 첫 호출 1.58초). 옛 코드는 그걸
`timeout=1`로 30회 찔러서, 매번 타임아웃 → 서버가 살아 있어도 실패로 판정했다.

대응: 의존성 검사가 없는 `/api/ping`으로 기동을 확인하고, `model_exists`는
파일 기준 캐시를 둬서 health 폴링이 매번 모델을 다시 로드하지 않게 한다.
"""
import time

# fastapi.testclient 는 httpx 를 요구하는데 이 프로젝트 의존성에 없다.
# 라우트 함수를 직접 호출해도 "무엇을 하는 엔드포인트인가"는 그대로 검증된다.
from api.server import app, health, ping


def _routes() -> set:
    return {r.path for r in app.routes}


def test_ping_route_is_registered():
    assert "/api/ping" in _routes()


def test_ping_has_no_dependency_checks():
    """/api/ping 은 의존성 상태와 무관하게 항상 최소 응답."""
    assert ping() == {"status": "ok"}


def test_ping_is_fast_enough_for_startup_probe():
    """기동 대기용이므로 DB·모델 상태와 무관하게 빨라야 한다."""
    ping()                                       # 워밍업
    start = time.perf_counter()
    for _ in range(50):
        ping()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"/api/ping 50회에 {elapsed:.2f}초 — 기동 확인용으로 너무 느리다"


def test_health_still_reports_components():
    """/api/health 는 의존성 진단 용도로 그대로 유지된다(느려도 됨)."""
    body = health()
    assert "components" in body
    for key in ("api", "database", "model"):
        assert key in body["components"]


def test_ui_startup_probe_targets_ping_not_health():
    """main.py 의 기동 대기가 다시 /api/health 로 돌아가지 않도록 고정."""
    import inspect
    import main

    src = inspect.getsource(main.cmd_ui)
    # 설명 주석에 /api/health 가 등장할 수 있으므로, 실제로 찌르는 URL
    # 리터럴만 본다.
    assert '"http://127.0.0.1:8001/api/ping"' in src, "기동 확인이 /api/ping 을 쓰지 않는다"
    assert '"http://127.0.0.1:8001/api/health"' not in src, (
        "기동 확인에 /api/health 를 쓰면 안 된다 — DB 연결·모델 로드가 섞여 "
        "서버가 살아 있어도 타임아웃으로 오판한다"
    )


def test_model_exists_is_cached(tmp_path, monkeypatch):
    """health 가 폴링돼도 모델 zip을 매번 다시 로드하지 않아야 한다."""
    from config import CONFIG
    from agent import rl_agent

    monkeypatch.setattr(CONFIG.path, "model_dir", tmp_path)
    rl_agent._MODEL_OBS_DIM_CACHE.clear()

    calls = {"n": 0}
    real_load = rl_agent.MaskablePPO.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    # 모델이 없으면 로드 자체가 없으므로, 존재하는 zip을 하나 만든다
    src = tmp_path / f"{CONFIG.rl.model_name}.zip"
    src.write_bytes(b"not-a-real-model")        # 로드 실패해도 캐시 경로는 동일
    monkeypatch.setattr(rl_agent.MaskablePPO, "load", staticmethod(counting_load))

    agent = rl_agent.SchedulingAgent()
    agent.model_exists()
    first = calls["n"]
    for _ in range(5):
        agent.model_exists()

    assert first >= 1
    assert calls["n"] == first, (
        f"model_exists 가 매번 모델을 다시 로드한다 ({calls['n']}회) — "
        "health 폴링마다 수 MB zip을 읽게 된다"
    )
