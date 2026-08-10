"""allocation – PPK/OPER × EQP_MODEL_CD 사전 할당(장비 댓수) 및 action masking 지원."""
from allocation.eqp_allocator import compute_eqp_allocation, build_eqp_alloc_map

__all__ = ["compute_eqp_allocation", "build_eqp_alloc_map"]
