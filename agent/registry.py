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
        "description": "PLAN_PRIORITY + 진행 기울기 최소 + 잔여(계획 또는 WIP) 우선",
        "requires_model": False,
    },
    {
        "id": "earliest_st",
        "name": "Earliest-ST (휴리스틱)",
        "description": "idle EQP에서 예상 종료 시각(장수×ST+conversion) 최소 재공 우선",
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
