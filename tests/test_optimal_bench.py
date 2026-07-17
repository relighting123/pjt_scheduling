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
# 아래는 현재 알려진 실제 격차다(production은 맞히는 경우가 대부분이지만
# conversions가 증명된 최소값을 넘는다):
#   - earliest_st: "예상 종료 시각이 가장 빠른 조합"만 보고 전환 비용을
#     고려하지 않아, 전담 배정이 가능한 상황에서도 스스로 찾지 못한다.
#   - minprogress: PPK별 진행률만 보고 오버플로를 한 제품에 몰아주지 않아,
#     초기 셋업이 다른 EQP가 여러 제품을 번갈아 맡으며 불필요한 추가 전환이
#     발생한다(overflow_conversion_three_eqp에서 직접 시뮬레이터를 조작해
#     production=20/conversions=1이 동시에 달성 가능함을 확인함 — 진짜 최적값).
#   - two_stage_* 는 위 두 패턴을 서로 다른 공정(OPER)에 독립 배치해 조합한
#     것이라 같은 격차가 공정 수만큼 반복/누적된다.
#   - pipeline_wip_buildup_then_steady: 공정 전환 시 TEMP가 바뀌어(600→900)
#     실제 전환 비용(60분)이 든다. 증명된 최적은 EQP 4대 중 "1대만" OPER002로
#     전환하는 비대칭 배분(production=16, conversions=1, sim=340).
#       · earliest_st: "예상 종료 시각이 가장 빠른 조합"만 보고 전환 횟수
#         자체는 최소화하지 않아, 4대 모두를 대칭적으로 OPER001 2LOT→전환→
#         OPER002 2LOT씩 배정한다(관측된 실행: 전 구간 [0,200) 4대 OPER001,
#         [200,260) 4대 동시 전환, [260,340) 4대 OPER002). 시간은 동일하게
#         340에 맞추지만(production=16/16) 전환이 4회로 증명된 최소 1회를
#         초과한다.
#       · minprogress: PPK 진행률만 보고 공정 간 배분은 보지 않아 한 EQP
#         (관측된 실행에서는 EQP004)가 OPER001↔OPER002를 왔다갔다하며 전환을
#         반복하다 마지막 OPER001 LOT 완료가 sim_end 밖으로 밀려나
#         production 13/16, conversions 5로 production·conversions 둘 다
#         증명된 최적에 못 미친다.
# strict=True: 이 조합이 어느 날 우연히 통과하면 XPASS로 테스트가 실패한다 —
# 알고리즘이 개선됐다는 신호이니 이 목록에서 지워야 한다.
_KNOWN_GAPS_STRICT = {
    ("earliest_st", "dedicated_assignment"),
    ("minprogress", "overflow_conversion_three_eqp"),
    ("earliest_st", "overflow_conversion_three_eqp"),
    ("minprogress", "pipeline_wip_buildup_then_steady"),
    ("earliest_st", "pipeline_wip_buildup_then_steady"),
    ("earliest_st", "two_stage_dedicated_small"),
    ("earliest_st", "two_stage_dedicated_mixed"),
    ("earliest_st", "two_stage_dedicated_overflow"),
    ("earliest_st", "two_stage_overflow_overflow"),
    ("earliest_st", "two_stage_mixed_overflow"),
    ("earliest_st", "two_stage_dedicated_large"),
}

# 알려진 격차이면서 동시에 재현이 불안정한 조합(비-strict xfail).
#
# KNOWN ISSUE: minprogress는 여러 EQP가 동시에 idle 결정을 필요로 하는
# 시나리오(다중 공정으로 EQP 수가 많아진 케이스)에서 PYTHONHASHSEED 값에 따라
# 최종 conversions 수가 프로세스 실행마다 달라진다(같은 프로세스 안에서는
# 안정적, 재실행하면 값이 바뀜 — 예: two_stage_overflow_overflow에서 2~10회
# 사이로 관측됨). earliest_st는 동일 조건에서 항상 결정적이다. 원인은
# simulation/simulator.py 어딘가의 set/dict 순회 순서로 추정되나 이 케이스
# 추가 작업 범위를 벗어나 근본 수정은 하지 않았다 — 재현성이 중요하면
# PYTHONHASHSEED를 고정해 실행할 것. strict=False라 우연히 최적값을 맞혀도
# 테스트가 깨지지 않는다(반대로 이 목록에서 빠지지도 않으니, 근본 원인을
# 고치면 수동으로 정리할 것).
_KNOWN_GAPS_FLAKY = {
    ("minprogress", "two_stage_dedicated_overflow"),
    ("minprogress", "two_stage_overflow_overflow"),
    ("minprogress", "two_stage_mixed_overflow"),
}


def _marks_for(algorithm: str, case_id: str):
    if (algorithm, case_id) in _KNOWN_GAPS_STRICT:
        return pytest.mark.xfail(
            reason=f"{algorithm}가 {case_id}의 증명된 최적값에 아직 도달하지 못함 (알려진 격차)",
            strict=True,
        )
    if (algorithm, case_id) in _KNOWN_GAPS_FLAKY:
        return pytest.mark.xfail(
            reason=(
                f"{algorithm}가 {case_id}의 증명된 최적값에 아직 도달하지 못함 "
                "(알려진 격차 + PYTHONHASHSEED에 따라 재현 불안정 — non-strict)"
            ),
            strict=False,
        )
    return ()


_PARAMS = [
    pytest.param(case, algorithm, marks=_marks_for(algorithm, case.id), id=f"{case.id}-{algorithm}")
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
