"""스케줄링 알고리즘 레지스트리."""
from typing import Literal

AlgorithmType = Literal["rl", "bulkfill", "minprogress", "earliest_st"]

ALGORITHMS: list[dict] = [
    {
        "id": "rl",
        "name": "PPO (강화학습)",
        "description": "Stable-Baselines3 PPO 학습 모델",
        "requires_model": True,
    },
    {
        "id": "bulkfill",
        "name": "Bulk-Fill PPO (벌크 점유)",
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
]

VALID_ALGORITHMS = {a["id"] for a in ALGORITHMS}


def validate_algorithm(algorithm: str) -> str:
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"지원하지 않는 알고리즘: {algorithm}. "
            f"사용 가능: {', '.join(sorted(VALID_ALGORITHMS))}"
        )
    return algorithm
