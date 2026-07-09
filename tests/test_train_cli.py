"""train CLI / API 기본 데이터 소스(JSON vs DB) 테스트."""
import sys
from unittest.mock import patch

import pytest

from api.server import TrainRequest, _resolve_train_folders
from main import parse_args


def test_train_parse_args_no_db_flags():
    argv = sys.argv
    try:
        sys.argv = ["main.py", "train", "--facid", "FAC001", "--prevcnt", "3"]
        args = parse_args()
        assert not hasattr(args, "db") or getattr(args, "db", False) is False
        assert not hasattr(args, "nodb") or getattr(args, "nodb", False) is False
    finally:
        sys.argv = argv


def test_infer_parse_args_no_nodb_flag():
    argv = sys.argv
    try:
        sys.argv = ["main.py", "infer", "--facid", "FAC001"]
        args = parse_args()
        assert not hasattr(args, "nodb")
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
    """--prevcnt N: 오늘 날짜 기준 구간이 아니라, 존재하는 train 폴더 중 최근(정렬상 마지막) N개."""
    monkeypatch.setattr("config.DATASET_DIR", tmp_path)
    periods = [
        "20260101170000",
        "20260201170000",
        "20260301170000",
        "20260621170000",
        "20260622170000",
    ]
    for period in periods:
        inp = tmp_path / "FAC001" / "train" / period / "input"
        inp.mkdir(parents=True)
        (inp / "discrete_arrange.json").write_text("[]", encoding="utf-8")

    req = TrainRequest(prevcnt=3, fac_id="FAC001")
    with patch("data.loader.rule_timekey_query.resolve_collect_periods") as db_call:
        folders = _resolve_train_folders(req)
        db_call.assert_not_called()

    assert folders == [
        "FAC001/train/20260301170000",
        "FAC001/train/20260621170000",
        "FAC001/train/20260622170000",
    ]
