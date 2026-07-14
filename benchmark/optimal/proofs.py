"""
benchmark/optimal/proofs.py — 벤치마크 케이스의 최적값을 계산하는 순수 함수

시뮬레이터/에이전트와 완전히 독립적으로, 문제의 수학적 구조만으로 최적값을
계산한다. benchmark/optimal/cases.py가 이 값을 "정답"으로 사용한다.
"""
from __future__ import annotations


def capacity_bound(n_available: int, st: int, horizon: int) -> int:
    """EQP 1대가 ST분짜리 carrier를 horizon분 안에 처리 가능한 최대 개수.

    증명: carrier 1개는 EQP를 ST분간 배타적으로 점유하므로, EQP의 가용
    시간(horizon)을 넘는 개수는 물리적으로 불가능하다 (상한 = floor(horizon/ST)).
    이 상한은 재공이 충분하면(n_available >= 상한) 처리 순서와 무관하게
    항상 달성 가능하므로 곧 최적값이다. horizon에 이미 전환(conversion) 등으로
    소모된 시간을 뺀 값을 넘기면, "전환 이후 남은 시간의 처리 상한"도 계산할 수 있다.
    """
    return min(n_available, horizon // st)
