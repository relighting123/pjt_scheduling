"""Tool-Change 벤치마크의 FAC별 모델 선택과 캐시 리로드 검증."""

import pytest
from fastapi import HTTPException

from api import server


def test_reload_model_evicts_selected_fac_cache(monkeypatch):
    old_agent = object()
    loaded_agent = object()
    server._benchmark_rl_agent.clear()
    server._benchmark_rl_agent["FAC002"] = old_agent

    def fake_get(fac_id=None):
        assert fac_id == "FAC002"
        assert "FAC002" not in server._benchmark_rl_agent
        return loaded_agent

    monkeypatch.setattr(server, "_get_benchmark_rl_agent", fake_get)

    result = server.reload_model("FAC002")

    assert result == {"exists": True, "fac_id": "FAC002", "reloaded": True}


def test_tool_change_benchmark_loads_requested_fac_model(monkeypatch):
    loaded_agent = object()
    captured = {}

    def fake_get(fac_id=None):
        captured["fac_id"] = fac_id
        return loaded_agent

    def fake_run(*, algorithms, rl_agent):
        captured["algorithms"] = algorithms
        captured["rl_agent"] = rl_agent
        return {"algorithms": algorithms, "summary": {}, "cases": []}

    monkeypatch.setattr(server, "_get_benchmark_rl_agent", fake_get)
    monkeypatch.setattr(server, "run_detailed_benchmark", fake_run)

    result = server.get_tool_change_bench("scheduling_rl,dedication", "FAC002")

    assert captured == {
        "fac_id": "FAC002",
        "algorithms": ["scheduling_rl", "dedication"],
        "rl_agent": loaded_agent,
    }
    assert result["fac_id"] == "FAC002"


def test_tool_change_benchmark_rejects_missing_fac_model(monkeypatch):
    monkeypatch.setattr(server, "_get_benchmark_rl_agent", lambda fac_id=None: None)

    with pytest.raises(HTTPException) as exc_info:
        server.get_tool_change_bench("scheduling_rl", "FAC404")

    assert exc_info.value.status_code == 404
    assert "FAC_ID=FAC404" in str(exc_info.value.detail)
