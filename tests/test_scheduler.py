"""API 서버 기동 시 주기적으로 infer 를 자동 실행하는 스케줄러 검증.

기본은 비활성(opt-in)이어야 하고, fac_id 없이 enabled만 켜져 있으면 시작하지
않아야 한다. 활성화되면 설정된 fac_id/algorithm/db_alias로 InferenceRequest를
만들어 inference()를 호출하고, 성공/실패 여부를 _scheduler_state에 남긴다.
"""
import asyncio

import pytest
from fastapi import HTTPException

import api.server as server


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    server._scheduler_state.update(
        running=False, run_count=0, last_run_at=None,
        last_status=None, last_error=None,
    )
    original_task = server._scheduler_task
    yield
    server._scheduler_task = original_task


def _run(coro):
    return asyncio.run(coro)


def test_startup_does_not_start_when_disabled(monkeypatch):
    monkeypatch.setattr(server.CONFIG.scheduler, "enabled", False)
    server._scheduler_task = None
    _run(server._start_scheduler_on_boot())
    assert server._scheduler_task is None


def test_startup_does_not_start_when_enabled_but_no_fac_id(monkeypatch):
    monkeypatch.setattr(server.CONFIG.scheduler, "enabled", True)
    monkeypatch.setattr(server.CONFIG.scheduler, "fac_id", "")
    server._scheduler_task = None
    _run(server._start_scheduler_on_boot())
    assert server._scheduler_task is None


def test_startup_starts_task_when_enabled_with_fac_id(monkeypatch):
    monkeypatch.setattr(server.CONFIG.scheduler, "enabled", True)
    monkeypatch.setattr(server.CONFIG.scheduler, "fac_id", "FAC001")
    monkeypatch.setattr(server.CONFIG.scheduler, "interval_seconds", 3600)

    calls = []

    def fake_inference(req):
        calls.append(req)
        return {"ok": True}

    monkeypatch.setattr(server, "inference", fake_inference)
    server._scheduler_task = None

    async def _startup_then_stop():
        await server._start_scheduler_on_boot()
        assert server._scheduler_task is not None
        # 첫 tick이 돌 시간을 잠깐 준다 (실제 sleep(3600) 전, 루프 첫 회는 즉시 실행)
        await asyncio.sleep(0.05)
        await server._stop_scheduler_on_exit()

    _run(_startup_then_stop())

    assert server._scheduler_task is None
    assert len(calls) == 1
    assert calls[0].fac_id == "FAC001"


def test_tick_success_updates_state(monkeypatch):
    monkeypatch.setattr(server, "inference", lambda req: {"ok": True})
    cfg = server.CONFIG.scheduler
    monkeypatch.setattr(cfg, "fac_id", "FAC001")

    _run(server._scheduler_tick(cfg))

    assert server._scheduler_state["last_status"] == "ok"
    assert server._scheduler_state["last_error"] is None
    assert server._scheduler_state["run_count"] == 1
    assert server._scheduler_state["running"] is False


def test_tick_failure_is_captured_not_raised(monkeypatch):
    def failing_inference(req):
        raise HTTPException(status_code=500, detail="DB 적재 실패: boom")

    monkeypatch.setattr(server, "inference", failing_inference)
    cfg = server.CONFIG.scheduler
    monkeypatch.setattr(cfg, "fac_id", "FAC001")

    _run(server._scheduler_tick(cfg))  # 예외가 새어나가면 안 됨

    assert server._scheduler_state["last_status"] == "error"
    assert "boom" in server._scheduler_state["last_error"]


def test_scheduler_status_endpoint_reports_current_state(monkeypatch):
    monkeypatch.setattr(server.CONFIG.scheduler, "enabled", False)
    server._scheduler_task = None
    status = server.scheduler_status()
    assert status["enabled"] is False
    assert status["active"] is False
    assert "run_count" in status
