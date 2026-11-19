"""Reads the streams, duration, chapters and frame times of a media file with ffprobe."""

import json
import subprocess
import time
from typing import cast

from .values import Raw, StrPath, float_or_none, mapping

TEXT_SUBTITLE_CODECS = frozenset(
    {
        "subrip",
        "ass",
        "ssa",
        "mov_text",
        "webvtt",
        "text",
        "srt",
    }
)


TIMEOUT = 120
TRIES = 1
FRAME_TIMEOUT = 900


class FFProbeError(RuntimeError):
    pass


SHOW = {
    "format": "-show_format",
    "stream": "-show_streams",
    "chapter": "-show_chapters",
    "program": "-show_programs",
}


def run(path: StrPath, *flags: str, timeout: int = TIMEOUT, tries: int = TRIES) -> Raw:
    """ffprobe with the given flags, retried with a backoff. Raises on final failure."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", *flags, str(path)]
    err = "not run"
    for attempt in range(1, tries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return cast("Raw", json.loads(r.stdout or "{}"))  # ffprobe writes an object
            err = r.stderr.strip() or f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            err = f"timed out after {timeout}s"
        except Exception as e:
            err = str(e)
        if attempt < tries:
            time.sleep(2 * attempt)
    raise FFProbeError(f"{err} ({tries} attempt{'s' if tries > 1 else ''})")


def probe(
    path: StrPath,
    entries: str = "format:stream:chapter",
    *,
    timeout: int = TIMEOUT,
    tries: int = TRIES,
) -> Raw:
    return run(path, *[SHOW[w] for w in entries.split(":")], timeout=timeout, tries=tries)


def streams(
    path: StrPath, kind: str | None = None, *, timeout: int = TIMEOUT, tries: int = TRIES
) -> list[Raw]:
    out: list[Raw] = probe(path, timeout=timeout, tries=tries).get("streams", [])
    return [s for s in out if s.get("codec_type") == kind] if kind else out


def duration(path: StrPath, *, timeout: int = TIMEOUT, tries: int = TRIES) -> float | None:
    report = probe(path, "format", timeout=timeout, tries=tries)
    d: object = report.get("format", {}).get("duration")
    return float_or_none(d)


def chapters(path: StrPath, *, timeout: int = TIMEOUT, tries: int = TRIES) -> list[Raw]:
    found: list[Raw] = probe(path, "chapter", timeout=timeout, tries=tries).get("chapters", [])
    return found


def start(
    path: StrPath, kind: str | None = None, *, timeout: int = TIMEOUT, tries: int = TRIES
) -> float:
    """Where the file begins on its own timeline, or its first stream of one
    kind, in seconds. A stream need not begin where the file does: video
    often starts a few frames in, and audio now and then seconds in."""
    found: list[Raw] = (
        streams(path, kind, timeout=timeout, tries=tries)[:1]
        if kind
        else [probe(path, "format", timeout=timeout, tries=tries).get("format", {})]
    )
    first = mapping(found[0]) if found else None
    begins = float_or_none(first.get("start_time")) if first else None
    if begins is None:
        raise FFProbeError(f"no start time for {kind or 'the file'} in {path}")
    return begins


def frame_times(path: StrPath, timeout: int = FRAME_TIMEOUT) -> list[float]:
    """The time each video frame is shown, in seconds on the file's own
    timeline and in the order the frames are shown, so the nth value belongs
    to the nth frame a decoder hands over.

    No picture is decoded, but ffprobe reads every packet of the file to list
    them, so this costs a full read. dvt.tools.mp4.frame_times gives the same times
    from an mp4's index and comes here only for a file it cannot handle."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise FFProbeError(f"timed out after {timeout}s") from e
    if r.returncode != 0:
        raise FFProbeError(r.stderr.strip() or f"exit {r.returncode}")
    fields = [line.split(",")[0] for line in r.stdout.splitlines() if line.strip()]
    untimed = sum(f in ("", "N/A") for f in fields)
    if untimed or not fields:
        raise FFProbeError(f"{untimed} of {len(fields)} video packets carry no time in {path}")
    return sorted(float(f) for f in fields)
