"""알고리즘 비교 API 직렬화 테스트."""
from api.serializers import serialize_compare_response


def test_serialize_compare_response_includes_sim_base_time():
    payload = {
        "results": [
            {
                "schedule": [{"EQP_ID": "E1", "LOT_ID": "L1", "START_TM": 0, "END_TM": 120}],
                "stats": {"completed_qty": {}, "idle_total": 0, "oper_switches": 0, "prod_switches": 0},
                "plan": [],
                "sim_end_minutes": 1440,
                "sim_base_time": "2026-06-21 17:00:00",
            }
        ],
        "errors": [],
        "plan": [],
        "prod_keys": [],
        "oper_ids": [],
        "eqp_ids": ["E1"],
        "sim_end_minutes": 1440,
    }
    out = serialize_compare_response(payload)
    assert out["sim_base_time"] == "2026-06-21 17:00:00"
    assert out["results"][0]["sim_base_time"] == "2026-06-21 17:00:00"
