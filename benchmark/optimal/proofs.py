"""
benchmark/optimal/proofs.py — 벤치마크 케이스의 최적값을 계산하는 순수 함수

시뮬레이터/에이전트와 완전히 독립적으로, 문제의 수학적 구조만으로 최적값을
계산한다. benchmark/optimal/cases.py가 이 값을 "정답"으로 사용한다.
"""
from __future__ import annotations

import itertools
from typing import Sequence


def capacity_bound(n_available: int, st: int, horizon: int) -> int:
    """EQP 1대가 ST분짜리 carrier를 horizon분 안에 처리 가능한 최대 개수.

    증명: carrier 1개는 EQP를 ST분간 배타적으로 점유하므로, EQP의 가용
    시간(horizon)을 넘는 개수는 물리적으로 불가능하다 (상한 = floor(horizon/ST)).
    이 상한은 재공이 충분하면(n_available >= 상한) 처리 순서와 무관하게
    항상 달성 가능하므로 곧 최적값이다.
    """
    return min(n_available, horizon // st)


def spt_max_completed(sts: Sequence[int], horizon: int) -> int:
    """마감 horizon 이내에 완료 가능한 최대 작업 수 (Shortest-Processing-Time 최적).

    증명(교환 논증): 임의의 스케줄이 k개 작업을 horizon 이내에 완료했다면,
    그 k개의 처리시간 합은 반드시 "전체 중 가장 작은 k개"의 합 이상이다
    (더 작은 미선택 작업과 교환하면 합이 줄거나 같아지기 때문). 따라서
    k개를 완료할 수 있는 최소 소요시간은 ST 오름차순 k개의 합이며, 이것이
    horizon을 넘지 않는 최대 k가 최적값이다.
    """
    total = 0
    count = 0
    for st in sorted(sts):
        if total + st > horizon:
            break
        total += st
        count += 1
    return count


def bruteforce_max_completed(sts: Sequence[int], horizon: int) -> int:
    """spt_max_completed()를 전수 탐색(부분집합 합)으로 교차 검증하기 위한 함수.

    작업 개수가 작을 때만 사용한다(테스트 전용, 지수 시간). 크기 k인 모든
    부분집합의 합을 직접 확인해 horizon 이내에 들어가는 가장 큰 k를 찾는다.
    """
    n = len(sts)
    for k in range(n, -1, -1):
        for combo in itertools.combinations(sts, k):
            if sum(combo) <= horizon:
                return k
    return 0
