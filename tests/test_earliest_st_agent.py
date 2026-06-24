"""Earliest-ST: 장수×ST + 예상 종료 시각 기준 LOT/PPK 선택."""
from data.generator import generate_sample_data
from data.loader.fetch import validate_data
from data.loader.preprocess import preprocess
from inference.runner import run_inference
from config import set_input_folder


def _minimal_env():
  discrete = [
      {
          "EQP_ID": "EQP001",
          "LOT_ID": "LOT_SHORT",
          "PLAN_PROD_KEY": "PPK001",
          "OPER_ID": "OPER001",
          "ST": 10,
          "EQP_MODEL_CD": "A",
          "WF_QTY": 2,
      },
      {
          "EQP_ID": "EQP001",
          "LOT_ID": "LOT_LONG",
          "PLAN_PROD_KEY": "PPK001",
          "OPER_ID": "OPER001",
          "ST": 10,
          "EQP_MODEL_CD": "A",
          "WF_QTY": 10,
      },
  ]
  raw = {
      "discrete_arrange": discrete,
      "abstract_arrange": [
          {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER001", "EQP_MODEL_CD": "A", "ST": 10},
      ],
      "plan": [
          {
              "PLAN_PROD_KEY": "PPK001",
              "OPER_ID": "OPER001",
              "D0_PLAN_QTY": 12,
              "D1_PLAN_QTY": 12,
              "PLAN_PRIORITY": 1,
          },
      ],
      "flow": [{"PLAN_PROD_KEY": "PPK001", "OPER_SEQ": 1, "OPER_ID": "OPER001"}],
      "split": [
          {
              "PLAN_PROD_KEY": "PPK001",
              "OPER_ID": "OPER001",
              "EQP_MODEL_CD": "A",
              "SPLIT_QTY": 10,
          },
      ],
      "lot_master": [{"LOT_ID": "LC001", "LOT_CD": "LC001", "TEMP": "T700"}],
      "batch_info": [
          {
              "PLAN_PROD_KEY": "PPK001",
              "OPER_ID": "OPER001",
              "LOT_CD": "LC001",
              "TEMP": "T700",
          },
      ],
      "tool_capacity": [],
      "eqp_initial_state": [],
  }
  errors = validate_data(raw)
  assert not errors, errors
  return preprocess(raw)


def test_earliest_st_picks_shorter_proc_time_lot_first():
    env = _minimal_env()
    result = run_inference(env, algorithm="earliest_st", record_history=False)
    first = sorted(
        [r for r in result["schedule"] if r["EQP_ID"] == "EQP001"],
        key=lambda r: r["START_TM"],
    )[0]
    assert first["WF_QTY"] == 2
    assert first["PROC_TIME"] == 20
    assert first["LOT_ID"].startswith("LOT_SHORT")


def test_earliest_st_uses_end_time_not_per_wafer_st_only():
    """장당 ST가 같아도 장수×ST가 짧은 LOT이 먼저."""
    env = _minimal_env()
    result = run_inference(env, algorithm="earliest_st", record_history=False)
    eqp1 = sorted(
        [r for r in result["schedule"] if r["EQP_ID"] == "EQP001"],
        key=lambda r: r["START_TM"],
    )
    assert eqp1[0]["PROC_TIME"] <= eqp1[1]["PROC_TIME"]
