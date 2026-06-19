"""스케줄링 알고리즘 레지스트리."""
from typing import Literal

AlgorithmType = Literal["rl", "minprogress", "earliest_st"]

ALGORITHMS: list[dict] = [
    {
        "id": "rl",
        "name": "PPO (강화학습)",
        "description": "Stable-Baselines3 PPO 학습 모델",
        "requires_model": True,
    },
    {
        "id": "minprogress",
        "name": "Min-Progress (휴리스틱)",
        "description": "PLAN_PRIORITY + 차트 기울기 최소 + 잔여 계획 우선 (제품), ST는 LOT 선택 시만",
        "requires_model": False,
    },
    {
        "id": "earliest_st",
        "name": "Earliest-ST (휴리스틱)",
        "description": "계획 미참조, idle EQP 중 최소 ST 조합(장비×자재 소요시간) 우선",
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
