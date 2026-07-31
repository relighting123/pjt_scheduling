"""train --facid 생략 시 전체 FAC 학습 CLI 검증."""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from config import list_fac_ids


def test_list_fac_ids_empty_when_no_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr("config.DATASET_DIR", tmp_path / "dataset")
    assert list_fac_ids("train") == []


def test_list_fac_ids_returns_only_facs_with_train_data(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "FAC001" / "train" / "20260101070000" / "input").mkdir(parents=True)
    (dataset / "FAC001" / "train" / "20260101070000" / "input" / "discrete_arrange.json").write_text(
        "{}", encoding="utf-8"
    )
    (dataset / "FAC002" / "test" / "20260101070000" / "input").mkdir(parents=True)
    (dataset / "FAC002" / "test" / "20260101070000" / "input" / "discrete_arrange.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.setattr("config.DATASET_DIR", dataset)
    assert list_fac_ids("train") == ["FAC001"]


def test_train_parser_facid_optional():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument(
        "--facid", "--fac-id", "--fac_id", dest="facid", default=None,
        help="공장 ID",
    )
    train_p.add_argument("--all", dest="all_folders", action="store_true")

    args = parser.parse_args(["train"])
    assert args.facid is None
    assert args.all_folders is False

    args = parser.parse_args(["train", "--fac-id", "FAC001"])
    assert args.facid == "FAC001"


def test_cmd_train_without_facid_trains_each_fac(monkeypatch):
    import main

    trained: list[str] = []

    class AgentStub:
        def train(self, env_data, **kwargs):
            trained.append(kwargs["fac_id"])

        def save(self, **kwargs):
            pass

    monkeypatch.setattr(main, "list_fac_ids", lambda split: ["FAC001", "FAC002"])
    monkeypatch.setattr(
        main,
        "list_split_folders",
        lambda fac_id, split: [f"{fac_id}/train/20260101070000"],
    )
    monkeypatch.setattr(main, "_load_many", lambda folders: [{"eqp_ids": [], "lots": [], "prod_keys": [], "oper_ids": []}])
    monkeypatch.setattr(main, "_load_env_data", lambda folder, period_key=None: {"eqp_ids": [], "lots": [], "prod_keys": [], "oper_ids": []})
    monkeypatch.setattr(main, "SchedulingAgent", AgentStub)
    monkeypatch.setattr(
        main,
        "save_training_convergence_report",
        lambda *a, **k: {"json_path": "x", "png_path": None, "verdict": "ok", "note": ""},
    )
    monkeypatch.setattr(main, "run_validation", lambda *a, **k: None)
    monkeypatch.setattr(main, "model_dir_for", lambda fac_id: MagicMock())

    main.cmd_train(fac_id=None)

    assert trained == ["FAC001", "FAC002"]


def test_cmd_train_without_facid_exits_when_no_train_data(monkeypatch):
    import main

    monkeypatch.setattr(main, "list_fac_ids", lambda split: [])

    with pytest.raises(SystemExit) as exc:
        main.cmd_train(fac_id=None)
    assert exc.value.code == 1
