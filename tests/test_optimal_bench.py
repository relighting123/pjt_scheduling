"""
tests/test_optimal_bench.py

benchmark/optimal의 실제 알고리즘 채점이 증명된 최적값에 도달하는지 검증한다
(회귀 감지).
"""
import pytest

from benchmark.optimal.cases import CASES
from benchmark.optimal.runner import evaluate_case

# 모델(휴리스틱)이 증명된 최적값에 실제로 도달하는지 회귀 감지.
# RL(scheduling_rl)은 저장된 체크포인트가 있어야 실행 가능하므로 제외.
#
# earliest_st × dedicated_assignment는 현재 알려진 실제 격차다: earliest-ST는
# "예상 종료 시각이 가장 빠른 조합"만 보고 전환 비용을 고려하지 않으므로
# 전담 배정을 스스로 찾지 못한다 (증명 가능한 최적 24개 대비 실제 성과 미달).
# strict=True: 이 조합이 어느 날 우연히 통과하면 XPASS로 테스트가 실패한다 —
# 알고리즘이 개선됐다는 신호이니 이 목록에서 지워야 한다.
_KNOWN_GAPS = {("earliest_st", "dedicated_assignment")}

_PARAMS = [
    pytest.param(
        case, algorithm,
        marks=pytest.mark.xfail(
            reason=f"{algorithm}가 {case.id}의 증명된 최적값에 아직 도달하지 못함 (알려진 격차)",
            strict=True,
        ) if (algorithm, case.id) in _KNOWN_GAPS else (),
        id=f"{case.id}-{algorithm}",
    )
    for case in CASES
    for algorithm in ["minprogress", "earliest_st"]
]


@pytest.mark.parametrize("case,algorithm", _PARAMS)
def test_heuristic_reaches_proven_optimum(case, algorithm):
    run = evaluate_case(case, algorithm)
    assert run["passed"], (
        f"{case.id}: {algorithm}가 증명된 최적값에 도달하지 못함 — "
        f"actual={run['actual']} target={run['target']} (증명: {case.optimal.proof})"
    )
