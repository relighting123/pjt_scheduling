"""UI 벤치마크 스위트 API + 모델 리로드 검증.

GET /api/benchmark/suite (bench 8종 + holdout 6종 전체 케이스)와
POST /api/model/reload (FAC_ID 변경 시 캐시된 에이전트를 버려 모델을 다시
로딩)의 동작을 고정한다. fastapi.testclient는 httpx가 필요해 이 프로젝트
의존성에 없으므로 라우트 함수를 직접 호출한다(test_ui_startup_probe.py와
같은 방식). 스위트 데이터셋은 data/dataset/(gitignore 대상)에 온디맨드로
생성된다 — ensure_suite가 없으면 생성하고, 있으면 재사용한다.
"""
import pytest
from fastapi import HTTPException

import api.server as server
from api.server import ModelReloadRequest, get_suite_bench, reload_model
from benchmark.compare_vs_dedication import ensure_suite


def test_ensure_suite_generates_bench_and_holdout():
    """스위트 메타가 없으면 생성기를 돌려 bench 8종·holdout 6종을 만든다."""
    ensure_suite("bench")
    ensure_suite("holdout")
    from benchmark.compare_vs_dedication import load_meta
    assert len(load_meta("bench")) == 8
    assert len(load_meta("holdout")) == 6


def test_ensure_suite_rejects_train_pool():
    """train_pool은 학습용 풀이라 평가 스위트가 아니다 — 온디맨드 생성 미지원."""
    with pytest.raises(SystemExit):
        ensure_suite("train_pool")


def test_suite_bench_returns_all_bench_cases():
    body = get_suite_bench(suite="bench", algorithms="dedication,minprogress")
    assert body["suite"] == "bench"
    assert body["fac_id"]
    assert len(body["datasets"]) == 8            # BENCH_SUITE 8종 전부
    assert set(body["algorithms"]) == {"dedication", "minprogress"}
    for row in body["datasets"]:
        assert set(row["algos"]) == {"dedication", "minprogress"}
        k = row["algos"]["dedication"]
        assert 0 <= k["prod"] <= k["max"]
        assert k["conv"] >= 0
        assert row["desc"] and row["tests"]      # UI 상세 표에 쓰는 메타 포함
    assert set(body["summary"]) == {"dedication", "minprogress"}
    # dedication은 기준선이라 vs_dedication에는 들어가지 않는다
    assert "minprogress" in body["vs_dedication"]
    assert "dedication" not in body["vs_dedication"]


def test_suite_bench_returns_all_holdout_cases():
    body = get_suite_bench(suite="holdout", algorithms="dedication")
    assert body["suite"] == "holdout"
    assert len(body["datasets"]) == 6            # HOLDOUT_SUITE 6종 전부
    for row in body["datasets"]:
        assert row["name"].startswith("H_")
        assert set(row["algos"]) == {"dedication"}
    assert body["summary"]["dedication"]["n_sets"] == 6


def test_suite_bench_rejects_unknown_suite():
    with pytest.raises(HTTPException) as exc:
        get_suite_bench(suite="train_pool", algorithms="dedication")
    assert exc.value.status_code == 400


def test_suite_bench_rejects_unknown_algorithm():
    with pytest.raises(HTTPException) as exc:
        get_suite_bench(suite="bench", algorithms="no_such_algo")
    assert exc.value.status_code == 400


def test_suite_bench_rl_only_without_model_fails():
    """scheduling_rl만 선택했는데 해당 FAC_ID 모델이 없으면 400."""
    with pytest.raises(HTTPException) as exc:
        get_suite_bench(suite="bench", algorithms="scheduling_rl", fac_id="NO_SUCH_FAC_999")
    assert exc.value.status_code == 400


def test_reload_model_clears_cached_agent():
    """리로드하면 세션 캐시의 FAC_ID 에이전트가 제거된다(다음 실행 때 zip 재로딩)."""
    server._benchmark_rl_agent["FAC999"] = None   # 캐시에 뭔가 들어 있는 상태
    body = reload_model(ModelReloadRequest(fac_id="FAC999"))
    assert body["fac_id"] == "FAC999"
    assert body["exists"] is False                # models/FAC999/ 는 없다
    assert body["cache_cleared"] is True
    assert "FAC999" not in server._benchmark_rl_agent


def test_reload_model_without_cached_agent():
    server._benchmark_rl_agent.pop("FAC998", None)
    body = reload_model(ModelReloadRequest(fac_id="FAC998"))
    assert body["exists"] is False
    assert body["cache_cleared"] is False
