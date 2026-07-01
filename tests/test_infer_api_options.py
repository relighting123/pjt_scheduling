"""추론 API – CLI infer 옵션(fetch/db-load) 테스트."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def _minimal_env_data():
    return {
        "eqp_ids": ["EQP01"],
        "oper_ids": ["O1"],
        "prod_keys": ["P1"],
        "sim_end_minutes": 1440,
        "sim_base_time": "2026-06-21 17:00:00",
        "lots": [],
        "plan": [],
        "flow": {},
    }


def test_inference_fetches_db_by_default(client, monkeypatch, tmp_path):
  monkeypatch.setattr("config.DATASET_DIR", tmp_path / "dataset")
  infer_input = tmp_path / "dataset" / "FAC001" / "infer" / "input"
  infer_input.mkdir(parents=True)
  (infer_input / "discrete_arrange.json").write_text("[]", encoding="utf-8")

  fetch_calls = []

  def fake_fetch(**kwargs):
    fetch_calls.append(kwargs)
    return infer_input

  with patch("api.server.fetch_from_db", side_effect=fake_fetch), patch(
    "api.server._load_env_data",
    return_value=_minimal_env_data(),
  ), patch("api.server.run_inference", return_value={
    "schedule": [],
    "stats": {"idle_total": 0, "oper_switches": 0, "prod_switches": 0, "completed_qty": {}},
    "plan": [],
    "history": [],
    "event_log": [],
  }), patch("api.server.save_result"), patch(
    "api.server.SchedulingAgent.load",
    return_value=MagicMock(),
  ):
    res = client.post(
      "/api/inference",
      json={
        "input_folder": "FAC001/infer",
        "algorithm": "minprogress",
      },
    )

  assert res.status_code == 200
  assert len(fetch_calls) == 1
  assert fetch_calls[0]["fac_id"] == "FAC001"
  assert fetch_calls[0]["split"] == "infer"
  meta = res.json()["infer_meta"]
  assert meta["fetched_from_db"] is True
  assert meta["input_folder"] == "FAC001/infer"


def test_inference_nodb_skips_fetch(client, monkeypatch, tmp_path):
  monkeypatch.setattr("config.DATASET_DIR", tmp_path / "dataset")
  infer_input = tmp_path / "dataset" / "FAC001" / "infer" / "input"
  infer_input.mkdir(parents=True)
  (infer_input / "discrete_arrange.json").write_text("[]", encoding="utf-8")

  with patch("api.server.fetch_from_db") as fetch_mock, patch(
    "api.server._load_env_data",
    return_value=_minimal_env_data(),
  ), patch("api.server.run_inference", return_value={
    "schedule": [],
    "stats": {"idle_total": 0, "oper_switches": 0, "prod_switches": 0, "completed_qty": {}},
    "plan": [],
    "history": [],
    "event_log": [],
  }), patch("api.server.save_result"), patch(
    "api.server.SchedulingAgent.load",
    return_value=MagicMock(),
  ):
    res = client.post(
      "/api/inference",
      json={
        "input_folder": "FAC001/infer",
        "algorithm": "minprogress",
        "nodb": True,
      },
    )

  assert res.status_code == 200
  fetch_mock.assert_not_called()
  assert res.json()["infer_meta"]["nodb"] is True
  assert res.json()["infer_meta"]["fetched_from_db"] is False


def test_inference_db_load_after_save(client, monkeypatch, tmp_path):
  monkeypatch.setattr("config.DATASET_DIR", tmp_path / "dataset")
  infer_input = tmp_path / "dataset" / "FAC001" / "infer" / "input"
  infer_input.mkdir(parents=True)
  (infer_input / "discrete_arrange.json").write_text("[]", encoding="utf-8")

  with patch("api.server.fetch_from_db", return_value=infer_input), patch(
    "api.server._load_env_data",
    return_value=_minimal_env_data(),
  ), patch("api.server.run_inference", return_value={
    "schedule": [],
    "stats": {"idle_total": 0, "oper_switches": 0, "prod_switches": 0, "completed_qty": {}},
    "plan": [],
    "history": [],
    "event_log": [],
  }), patch("api.server.save_result"), patch(
    "api.server.load_output_sql_files",
  ) as load_mock, patch(
    "api.server.SchedulingAgent.load",
    return_value=MagicMock(),
  ):
    res = client.post(
      "/api/inference",
      json={
        "input_folder": "FAC001/infer",
        "algorithm": "minprogress",
        "db_load": True,
        "db_alias": "WT_RTS",
        "no_history": True,
      },
    )

  assert res.status_code == 200
  load_mock.assert_called_once()
  _, kwargs = load_mock.call_args
  assert kwargs["db_alias"] == "WT_RTS"
  assert kwargs["include_history"] is False
  assert res.json()["infer_meta"]["db_loaded"] is True
