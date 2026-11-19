"""File identity for the two sources: are we looking at the same bytes as last time?"""

import hashlib
import io
import os
from pathlib import Path

import polars as pl

from .values import StrPath

WINDOW = 1 << 20
DIGEST_BYTES = 16
# Part of every hash ever recorded. The project's old name stays here for
# that reason alone; changing it would change every recorded hash.
PREFIX = b"cinemetrics-headtail-1MiB-v1"


def _read_exactly(fh: io.FileIO, n: int) -> bytes:
    parts: list[bytes] = []
    while n > 0:
        chunk = fh.read(n)
        if not chunk:
            break
        parts.append(chunk)
        n -= len(chunk)
    return b"".join(parts)


def _new_digest(size: int) -> "hashlib.blake2b":
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    h.update(PREFIX)
    h.update(b"%d\n" % size)
    return h


def size_and_hash(path: StrPath) -> tuple[int, str] | tuple[None, None]:
    try:
        size = os.path.getsize(path)
        h = _new_digest(size)
        with open(path, "rb", buffering=0) as fh:
            if size <= 2 * WINDOW:
                while chunk := fh.read(WINDOW):
                    h.update(chunk)
            else:
                h.update(_read_exactly(fh, WINDOW))
                fh.seek(-WINDOW, os.SEEK_END)
                h.update(_read_exactly(fh, WINDOW))
    except OSError:
        return None, None
    return size, h.hexdigest()


NOAUDIO_SUFFIX = ".m4a.noaudio"
NOAUDIO_CONTENT = b"no audio stream in the source\n"


def noaudio_path_for(m4a: Path) -> Path:
    """The marker beside where a file's m4a would be, saying its source has
    no audio stream."""
    return m4a.with_name(m4a.stem + NOAUDIO_SUFFIX)


def audio_path_for(m4a: Path) -> Path | None:
    """The m4a if it is there, the marker if that is, and None when the
    extraction has not reached the file."""
    if m4a.exists():
        return m4a
    marker = noaudio_path_for(m4a)
    return marker if marker.exists() else None


HASH_COLS = ["mp4_bytes", "mp4_hash", "m4a_bytes", "m4a_hash"]

UNRECORDED_VERDICTS = ("mp4 not recorded", "m4a not recorded")

BLOCKING_VERDICTS = ("mp4 changed", "m4a changed", "mp4 gone", "m4a gone", *UNRECORDED_VERDICTS)


def check_sources(meta_path: str | Path, mp4_path: Path, m4a: Path) -> list[tuple[str, str]]:
    df = pl.read_parquet(meta_path)
    if "mp4_hash" not in df.columns:
        return [("not recorded", "metadata.parquet predates the hash columns")]

    stored = dict(zip(HASH_COLS, df.select(HASH_COLS).row(0), strict=True))
    out: list[tuple[str, str]] = []
    audio = audio_path_for(m4a) or m4a
    for kind, path in (("mp4", mp4_path), ("m4a", audio)):
        was_bytes, was_hash = stored[f"{kind}_bytes"], stored[f"{kind}_hash"]
        now_bytes, now_hash = size_and_hash(path)
        if was_hash is None:
            if now_hash is None:
                detail = f"row stores no {kind}_hash, and {path} is not there"
            else:
                detail = (
                    f"row stores no {kind}_hash; {path} is there now, "
                    f"{now_bytes:,} bytes, {now_hash}"
                )
            out.append((f"{kind} not recorded", detail))
            continue
        if now_hash is None:
            out.append((f"{kind} gone", f"cannot read {path}"))
        elif now_hash != was_hash:
            delta = (
                "same size" if now_bytes == was_bytes else f"{was_bytes:,} -> {now_bytes:,} bytes"
            )
            out.append((f"{kind} changed", f"{was_hash} -> {now_hash}, {delta}"))
    return out
