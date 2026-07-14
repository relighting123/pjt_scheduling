"""
benchmark/optimal — 증명 가능한 최적해 기반 모델 평가 벤치마크

"우리 알고리즘이 실제로 최적해에 도달하는가?"를 판정하기 위한, 상대
비교(bench_suite.py 등)와는 별개의 절대 평가 벤치마크다.

구조 (낮은 결합도)
  proofs.py  — 순수 수학 함수로 최적값을 계산 (시뮬레이터 비의존)
  cases.py   — data.generator + data.loader.preprocess만으로 env_data를
               메모리에서 구성 (파일 I/O 없음, 기존 코드 미수정)
  runner.py  — agent.registry / inference.runner.run_inference로 실제
               알고리즘을 실행해 증명된 최적값과 비교·채점

실행
  python -m benchmark.optimal.runner
"""
from benchmark.optimal.cases import CASES, OptimalCase, OptimalTarget, CaseMetrics, measure

__all__ = [
    "CASES",
    "OptimalCase",
    "OptimalTarget",
    "CaseMetrics",
    "measure",
]
