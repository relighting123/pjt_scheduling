"""스케줄링 알고리즘 레지스트리."""
from typing import Literal

AlgorithmType = Literal["scheduling_rl", "minprogress", "earliest_st", "dedication"]

ALGORITHMS: list[dict] = [
    {
        "id": "scheduling_rl",
        "name": "Scheduling RL (PPO)",
        "description": "(PPK/OPER, 블록 크기) 선택 — 같은 제품군 N carrier 연속 점유. "
                       "eligibility·takt·tool 잔여 고려",
        "requires_model": True,
    },
    {
        "id": "minprogress",
        "name": "Min-Progress (휴리스틱)",
        "description": "PLAN_PRIORITY + 진행 기울기 최소 + 잔여(계획 또는 WIP) 우선",
        "requires_model": False,
    },
    {
        "id": "earliest_st",
        "name": "Earliest-ST (휴리스틱)",
        "description": "EQP×carrier(ST×qty) 조합 중 예상 종료 시각+소요 최소 재공 우선",
        "requires_model": False,
    },
    {
        "id": "dedication",
        "name": "Dedication (전담 배분)",
        "description": "장비별 버킷 전담 유지 + 커버 기반 분산 배치. 전환은 하류 버킷의 "
                       "데드라인 슬랙 소진 시점에만(cascade), 그 외에는 HOLD로 채터링 차단",
        "requires_model": False,
    },
]

VALID_ALGORITHMS = {a["id"] for a in ALGORITHMS}


def validate_algorithm(algorithm: str) -> str:
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"지원하지 않는 알고리즘: {algorithm}. "
            f"사용 가능: {', '.join(sorted(VALID_ALGORITHMS))}"
        )
    return algorithm
