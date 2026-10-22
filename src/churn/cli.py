"""Command-line entry point: `churn <command>`. The Makefile wraps these."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

import duckdb

from churn.config import get_settings


def cmd_download(_: argparse.Namespace) -> None:
    from churn.data.download import download

    download(get_settings())


def _labels(con: duckdb.DuckDBPyConnection) -> None:
    from churn.data.labels import build_labels, validate_against_official

    settings = get_settings()
    build_labels(con, settings)
    report = asdict(validate_against_official(con, settings))
    out = settings.data_dir / "label_validation.json"
    out.write_text(json.dumps(report, indent=2))
    logging.info("Label validation vs official KKBox labels: %s", report)


def cmd_data(_: argparse.Namespace) -> None:
    from churn.data.ingest import build_raw_tables

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(settings.db_path)) as con:
        build_raw_tables(con, settings)
        _labels(con)


def cmd_labels(_: argparse.Namespace) -> None:
    """Rebuild labels from the tables already in DuckDB; raw files are not needed."""
    with duckdb.connect(str(get_settings().db_path)) as con:
        _labels(con)


def cmd_clean_raw(_: argparse.Namespace) -> None:
    from churn.data.ingest import clean_raw

    clean_raw(get_settings())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="churn")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download + extract KKBox data").set_defaults(func=cmd_download)
    sub.add_parser("data", help="Build sampled tables + labels").set_defaults(func=cmd_data)
    sub.add_parser("labels", help="Rebuild labels from DuckDB only").set_defaults(func=cmd_labels)
    sub.add_parser(
        "clean-raw", help="Delete data/raw after verifying the DuckDB tables"
    ).set_defaults(func=cmd_clean_raw)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
