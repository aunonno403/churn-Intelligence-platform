"""Download and unpack the KKBox data via the Kaggle API, keeping disk use low.

Prerequisites: a Kaggle API token (~/.kaggle/access_token) and having accepted the rules at
https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules

Disk: the zip (~8 GB) is deleted once the needed archives are out of it. Small archives are
unpacked to CSV and deleted; the user_logs archives stay compressed and are filtered while
decompressing in `churn data`. Peak ~16 GB during the unzip, ~12 GB after.
`churn clean-raw` removes everything under data/raw once the DuckDB file is built.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import py7zr

from churn.config import Settings

log = logging.getLogger(__name__)

COMPETITION = "kkbox-churn-prediction-challenge"
WANTED = {
    "members_v3.csv",
    "transactions.csv",
    "transactions_v2.csv",
    "user_logs.csv",
    "user_logs_v2.csv",
    "train.csv",
}
STREAMED = {"user_logs.csv", "user_logs_v2.csv"}  # never fully unpacked


def _csv_name(member: str) -> str:
    return Path(member).name.removesuffix(".7z")


def download(settings: Settings) -> None:
    raw = settings.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / f"{COMPETITION}.zip"

    if not zip_path.exists() and not any(raw.rglob("*.7z")):
        # Imported here: importing kaggle authenticates, which needs a token.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        log.info("Downloading %s to %s (~8 GB)...", COMPETITION, raw)
        api.competition_download_files(COMPETITION, path=str(raw), quiet=False)

    if zip_path.exists():
        try:
            zf = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            zip_path.unlink()
            raise RuntimeError(
                f"{zip_path.name} was incomplete and has been deleted. Run `churn download` again."
            ) from None
        with zf:
            members = [m for m in zf.namelist() if _csv_name(m) in WANTED]
            if missing := WANTED - {_csv_name(m) for m in members}:
                raise RuntimeError(
                    f"{zip_path.name} lacks {sorted(missing)}; zip kept, nothing extracted."
                )
            for member in members:  # always re-extract: an earlier run may have stopped mid-file
                log.info("Unzipping %s", member)
                zf.extract(member, raw)
        zip_path.unlink()
        log.info("Deleted %s", zip_path.name)

    for archive in list(raw.rglob("*.7z")):
        if _csv_name(archive.name) in STREAMED:
            continue
        log.info("Extracting %s", archive.name)
        with py7zr.SevenZipFile(archive) as sz:
            sz.extractall(path=archive.parent)
        archive.unlink()
