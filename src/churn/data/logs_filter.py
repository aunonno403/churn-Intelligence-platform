"""Stream the daily listening logs, keeping only sampled users inside the feature lookback.

user_logs.csv is ~30 GB uncompressed. Decompressing the .7z straight through this filter
means the full file never touches disk: only the ~1-2% of rows we keep are written.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import BinaryIO

import py7zr
from py7zr.io import NullIO, Py7zIO, WriterFactory

log = logging.getLogger(__name__)

CHUNK = 64 * 1024 * 1024
PROGRESS_EVERY = 2 * 1024**3


class LineFilter:
    """Incremental CSV filter: rows are `msno,YYYYMMDD,...`; the header always passes."""

    def __init__(self, out: BinaryIO, users: set[bytes], min_date: date) -> None:
        self._out = out
        self._users = users
        self._min_date = min_date.strftime("%Y%m%d").encode()
        self._tail = b""
        self._header_done = False
        self.rows_seen = 0
        self.rows_kept = 0

    def feed(self, chunk: bytes) -> None:
        lines = (self._tail + chunk).split(b"\n")
        self._tail = lines.pop()  # possibly incomplete last line
        self._write(lines)

    def finish(self) -> None:
        if self._tail.strip():
            self._write([self._tail])
        self._tail = b""

    def _write(self, lines: list[bytes]) -> None:
        if not self._header_done and lines:
            self._out.write(lines.pop(0) + b"\n")
            self._header_done = True
        users, min_date = self._users, self._min_date
        kept = []
        for line in lines:
            msno, _, rest = line.partition(b",")
            if msno in users and rest[:8] >= min_date:
                kept.append(line)
        self.rows_seen += len(lines)
        self.rows_kept += len(kept)
        if kept:
            self._out.write(b"\n".join(kept) + b"\n")


class _FilterIO(Py7zIO):
    """py7zr writer that forwards decompressed bytes to a LineFilter instead of a file."""

    def __init__(self, filt: LineFilter, label: str) -> None:
        self._filt, self._label, self._n, self._next_log = filt, label, 0, PROGRESS_EVERY

    def write(self, s: bytes | bytearray) -> int:
        self._filt.feed(bytes(s))
        self._n += len(s)
        if self._n >= self._next_log:
            log.info("  %s: %.0f GB decompressed", self._label, self._n / 1024**3)
            self._next_log += PROGRESS_EVERY
        return len(s)

    def read(self, size: int | None = None) -> bytes:
        return b""

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._n

    def flush(self) -> None:
        pass

    def size(self) -> int:
        return self._n


class _FilterFactory(WriterFactory):
    def __init__(self, filt: LineFilter, label: str) -> None:
        self._filt, self._label = filt, label

    def create(self, filename: str) -> Py7zIO:
        return _FilterIO(self._filt, self._label) if filename.endswith(".csv") else NullIO()


def filter_logs(source: Path, out: Path, users: set[bytes], min_date: date) -> None:
    """Filter a user_logs .csv or .csv.7z into `out`. Writes atomically via a .partial file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    with tmp.open("wb") as fh:
        filt = LineFilter(fh, users, min_date)
        if source.suffix == ".7z":
            with py7zr.SevenZipFile(source) as sz:
                sz.extractall(factory=_FilterFactory(filt, source.name))
        else:
            with source.open("rb") as src:
                while chunk := src.read(CHUNK):
                    filt.feed(chunk)
        filt.finish()
    tmp.replace(out)
    log.info("  %s: kept %s of %s rows", source.name, f"{filt.rows_kept:,}", f"{filt.rows_seen:,}")
