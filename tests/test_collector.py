"""collector 디버그 CLI 테스트."""
import textwrap
from pathlib import Path

import pytest

from data.collector import (
    CollectorOptions,
    TrainingDataCollector,
    build_arg_parser,
    collector_options_from_args,
)
from data.loader.fetch import fetch_from_db


def test_collector_debug_args():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--once", "--facid", "FAC001", "--preflight", "-v", "--debug",
    ])
    opts = collector_options_from_args(args)
    assert opts.preflight is True
    assert opts.verbose is True
    assert opts.debug is True


def test_fetch_dry_run_skips_oracle(tmp_path, monkeypatch):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        textwrap.dedent("""
        default: WT_RTS
        WT_RTS:
          user: u
          password: p
          dsn: d
        """),
        encoding="utf-8",
    )
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    for name in (
        "discrete_arrange.sql",
        "abstract_arrange.sql",
        "plan.sql",
        "flow.sql",
        "split.sql",
        "batch_info.sql",
    ):
        (sql_dir / name).write_text("@db:WT_RTS\nSELECT 1 FROM dual", encoding="utf-8")

    monkeypatch.setenv("DB_CONFIG", str(cfg))
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    out = fetch_from_db(
        fac_id="FAC001",
        split="train",
        period="20260621170000",
        lot_cd="LC001",
        dry_run=True,
        verbose=True,
    )
    assert "20260621170000" in str(out)
    assert not (out / "plan.json").exists()


def test_fetch_dry_run_logs_lot_cd_bind(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        "default: WT_RTS\nWT_RTS:\n  user: u\n  password: p\n  dsn: d\n",
        encoding="utf-8",
    )
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    for name in (
        "discrete_arrange.sql",
        "abstract_arrange.sql",
        "plan.sql",
        "flow.sql",
        "split.sql",
        "batch_info.sql",
    ):
        (sql_dir / name).write_text("@db:WT_RTS\nSELECT 1 FROM dual", encoding="utf-8")

    monkeypatch.setenv("DB_CONFIG", str(cfg))
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    fetch_from_db(
        fac_id="FAC001",
        split="train",
        period="20260621170000",
        lot_cd="LC001",
        dry_run=True,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "LOT_CD" in out


def test_collector_lot_cd_arg():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--once", "--facid", "FAC001", "--lot-cd", "LC001",
    ])
    assert args.lot_cd == "LC001"
    assert args.facid == "FAC001"
    collector = TrainingDataCollector(fac_id=args.facid, lot_cd=args.lot_cd)
    assert collector.lot_cd == "LC001"


def test_fetch_error_includes_sql_context(tmp_path, monkeypatch):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        "default: WT_RTS\nWT_RTS:\n  user: u\n  password: p\n  dsn: d\n",
        encoding="utf-8",
    )
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "plan.sql").write_text("@db:WT_RTS\nSELECT 1", encoding="utf-8")
    for name in (
        "discrete_arrange.sql",
        "abstract_arrange.sql",
        "flow.sql",
        "split.sql",
        "batch_info.sql",
    ):
        (sql_dir / name).write_text("@db:WT_RTS\nSELECT 1", encoding="utf-8")

    monkeypatch.setenv("DB_CONFIG", str(cfg))
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    with pytest.raises(RuntimeError) as exc:
        fetch_from_db(fac_id="FAC001", split="train", period="20260621170000")
    msg = str(exc.value)
    assert "discrete_arrange.sql" in msg
    assert "wt_rts" in msg
    assert "period=20260621170000" in msg


def test_collector_preflight_runs_without_db(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "databases.yaml"
    cfg.write_text(
        "default: WT_RTS\nWT_RTS:\n  user: u\n  password: p\n  dsn: d\n",
        encoding="utf-8",
    )
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    for name in (
        "discrete_arrange.sql",
        "abstract_arrange.sql",
        "plan.sql",
        "flow.sql",
        "split.sql",
        "batch_info.sql",
    ):
        (sql_dir / name).write_text("@db:WT_RTS\nSELECT 1", encoding="utf-8")

    monkeypatch.setenv("DB_CONFIG", str(cfg))
    monkeypatch.setattr("config.SQL_DIR", sql_dir)

    collector = TrainingDataCollector(fac_id="FAC001", prevdays=1)
    collector.collect_once(
        options=CollectorOptions(preflight=True, verbose=True),
    )
    out = capsys.readouterr().out
    assert "[preflight]" in out
    assert "WT_RTS" in out or "wt_rts" in out
