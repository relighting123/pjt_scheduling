"""train CLI / API 기본 데이터 소스(JSON vs DB) 테스트."""
import sys
from unittest.mock import patch

import pytest

from api.server import TrainRequest, _resolve_train_folders
from main import parse_args


def test_train_parse_args_default_json_mode():
    argv = sys.argv
    try:
        sys.argv = ["main.py", "train", "--facid", "FAC001", "--prevcnt", "3"]
        args = parse_args()
        assert args.db is False
        assert not args.db
    finally:
        sys.argv = argv


def test_train_parse_args_db_flag():
    argv = sys.argv
    try:
        sys.argv = ["main.py", "train", "--facid", "FAC001", "--prevcnt", "3", "--db"]
        args = parse_args()
        assert args.db is True
    finally:
        sys.argv = argv


def test_resolve_train_folders_period_range_uses_local_json(tmp_path, monkeypatch):
    monkeypatch.setattr("config.DATASET_DIR", tmp_path)
    for period in ("20260621170000", "20260622170000"):
        inp = tmp_path / "FAC001" / "train" / period / "input"
        inp.mkdir(parents=True)
        (inp / "discrete_arrange.json").write_text("[]", encoding="utf-8")

    req = TrainRequest(
        from_date="20260621170000",
        to_date="20260622170000",
        fac_id="FAC001",
    )
    with patch("data.loader.rule_timekey_query.resolve_collect_periods") as db_call:
        folders = _resolve_train_folders(req)
        db_call.assert_not_called()

    assert folders == [
        "FAC001/train/20260621170000",
        "FAC001/train/20260622170000",
    ]


def test_resolve_train_folders_prevcnt_uses_local_json(tmp_path, monkeypatch):
    monkeypatch.setattr("config.DATASET_DIR", tmp_path)
    period = "20260621170000"
    inp = tmp_path / "FAC001" / "train" / period / "input"
    inp.mkdir(parents=True)
    (inp / "discrete_arrange.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        "api.server.resolve_train_period_range",
        lambda **kwargs: ("20260621170000", "20260621170000"),
    )

    req = TrainRequest(prevcnt=3, fac_id="FAC001")
    with patch("data.loader.rule_timekey_query.resolve_collect_periods") as db_call:
        folders = _resolve_train_folders(req)
        db_call.assert_not_called()

    assert folders == ["FAC001/train/20260621170000"]
