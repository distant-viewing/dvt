"""Audio times onto the mp4's own timeline.

Every audio annotator works from the m4a, or from a stem cut from it, and
measures time from its first sample. That sample need not sit at the start
of the mp4: an audio stream may begin after the picture, and the extraction
keeps that start, so the m4a carries it. Each audio table is written here
with that start added to its time columns, so every table under
annotations/ is on the one timeline the video tables use, and the amount
added is kept in the parquet's key-value metadata under AUDIO_START so a
table says which clock it is on.
"""

from pathlib import Path

import polars as pl

from .annotators import stamp
from .tools import ffprobe
from .tools.values import StrPath

AUDIO_START = "audio_start"

# The time columns of each audio table.
TIME_COLUMNS = {
    "words.parquet": ("start", "end"),
    "audioseg_intervals.parquet": ("start", "end"),
    "audioseg_timeline.parquet": ("time",),
    "praat.parquet": ("time",),
    "laughter.parquet": ("start", "end"),
    "laughter_labels.parquet": ("start", "end"),
}


def start(mp4_path: StrPath, m4a_path: StrPath) -> float:
    """Where the m4a's first sample sits on the mp4's timeline. ffmpeg wrote
    the m4a from where the mp4 begins, so its own start is counted from
    there. Zero for nearly every file."""
    return ffprobe.start(mp4_path) + ffprobe.start(m4a_path, "audio")


def shifted(df: pl.DataFrame, table: str, by: float) -> pl.DataFrame:
    """The table with `by` added to its time columns."""
    cols = [k for k in TIME_COLUMNS[table] if k in df.columns]
    return df.with_columns([pl.col(k) + by for k in cols]) if by and cols else df


def write(df: pl.DataFrame, name: str, path: Path, start: float) -> None:
    """Writes one audio table on the mp4's clock: `start` is added to its
    time columns, the annotator's stamp goes on, and the amount is recorded
    in the file's metadata."""
    start = float(start)
    stamp(shifted(df, path.name, start), name).write_parquet(
        path, metadata={AUDIO_START: repr(start)}
    )


def recorded(path: Path) -> float | None:
    """The start an audio table was written with, or None for a table
    written before this module existed. Every such table on the cluster
    was shifted once, in September 2026, so it is on the mp4's clock too;
    the amount was recorded then only where it was not zero."""
    value = pl.read_parquet_metadata(path).get(AUDIO_START)
    return None if value is None else float(value)
