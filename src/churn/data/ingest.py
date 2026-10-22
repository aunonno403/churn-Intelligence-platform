"""Raw KKBox files -> sampled, typed DuckDB tables."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb

from churn.config import Settings
from churn.data.logs_filter import filter_logs
from churn.data.sql import render

log = logging.getLogger(__name__)

# transactions_v2 holds March 2017: the renewals that decide the test cohort's labels.
CSV_FILES = ["transactions.csv", "transactions_v2.csv", "members_v3.csv", "train.csv"]
LOG_FILES = ["user_logs.csv", "user_logs_v2.csv"]
TABLES = ["sampled_users", "transactions", "user_logs", "members", "official_labels"]


def _find(raw: Path, name: str) -> Path | None:
    """Locate `name` as a plain CSV or as `name.7z` anywhere under raw/."""
    for candidate in (name, f"{name}.7z"):
        if hit := next(raw.rglob(candidate), None):
            return hit
    return None


def filtered_logs_dir(settings: Settings) -> Path:
    # Keyed by sample + window so changing either never reuses a stale filtered file.
    tag = f"{settings.sample_keep}of{settings.sample_mod}_from{settings.logs_start:%Y%m%d}"
    return settings.raw_dir / f"filtered_logs_{tag}"


def check_raw_files(settings: Settings) -> None:
    # `churn download` deletes each archive only after unpacking it, so a leftover one means
    # its CSV may be truncated. The log archives are meant to stay (they are streamed).
    unfinished = [
        p.name
        for p in settings.raw_dir.rglob("*.7z")
        if p.name.removesuffix(".7z") not in LOG_FILES
    ]
    if unfinished:
        raise RuntimeError(
            f"Unpacking did not finish for {unfinished}. Run `churn download` again."
        )
    missing = [f for f in CSV_FILES if not any(settings.raw_dir.rglob(f))]
    missing += [f for f in LOG_FILES if _find(settings.raw_dir, f) is None]
    if missing:
        raise FileNotFoundError(
            f"Missing raw files under {settings.raw_dir}: {missing}. Run `churn download` first."
        )


def build_raw_tables(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    check_raw_files(settings)
    params = dict(
        raw_dir=settings.raw_dir.as_posix(),
        sample_mod=settings.sample_mod,
        sample_keep=settings.sample_keep,
    )
    con.execute(render("00_raw_tables.sql", settings, **params))

    users = {row[0].encode() for row in con.execute("SELECT msno FROM sampled_users").fetchall()}
    logs_dir = filtered_logs_dir(settings)
    for name in LOG_FILES:
        out = logs_dir / name
        if out.exists():
            log.info("Reusing filtered %s", out)
            continue
        log.info("Filtering %s (the full user_logs pass takes a while)...", name)
        filter_logs(_find(settings.raw_dir, name), out, users, settings.logs_start)
    con.execute(render("01_user_logs.sql", settings, logs_dir=logs_dir.as_posix()))

    for table in TABLES:
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        log.info("  %-16s %12s rows", table, f"{n:,}")


def clean_raw(settings: Settings) -> None:
    """Delete data/raw once DuckDB holds every table. Re-run `churn download` to rebuild."""
    if not settings.db_path.exists():
        raise RuntimeError(f"{settings.db_path} not found; run `churn data` before cleaning.")
    with duckdb.connect(str(settings.db_path), read_only=True) as con:
        present = {r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
        for table in [*TABLES, "labels"]:
            if (
                table not in present
                or not con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            ):
                raise RuntimeError(f"Table {table!r} missing or empty; not deleting raw data.")
    if not settings.raw_dir.exists():
        log.info("Nothing to clean: %s does not exist.", settings.raw_dir)
        return
    size = sum(f.stat().st_size for f in settings.raw_dir.rglob("*") if f.is_file())
    shutil.rmtree(settings.raw_dir)
    log.info("Deleted %s (%.1f GB freed).", settings.raw_dir, size / 1024**3)
