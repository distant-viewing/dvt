"""Chapters and subtitles: the two annotations that are already in the file."""

import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from .tools import ffprobe
from .tools.values import Raw, StrPath

# A table's column types, as polars takes them.
Schema = Mapping[str, pl.DataType | type[pl.DataType]]
# One subtitle cue: when it starts and ends, in seconds, and its text.
Cue = tuple[float, float, str]

TEXT_SUBTITLE_CODECS = ffprobe.TEXT_SUBTITLE_CODECS

PROBE_TIMEOUT = 120
PROBE_TRIES = 3

EXTRACT_TIMEOUT = 1200

CHAPTER_SCHEMA = {
    "fname": pl.Utf8,
    "cid": pl.Int64,
    "title": pl.Utf8,
    "t0": pl.Float64,
    "t1": pl.Float64,
    "present": pl.Boolean,
}

SUBTITLE_SCHEMA = {
    "fname": pl.Utf8,
    "sid": pl.Int64,
    "stream": pl.Int64,
    "lang": pl.Utf8,
    "codec": pl.Utf8,
    "forced": pl.Boolean,
    "default": pl.Boolean,
    "eid": pl.Int64,
    "t0": pl.Float64,
    "t1": pl.Float64,
    "text": pl.Utf8,
    "present": pl.Boolean,
}


def _sentinel(fname: str, schema: Schema) -> pl.DataFrame:
    row: dict[str, list[object]] = {k: [None] for k in schema}
    row["fname"] = [fname]
    row["present"] = [False]
    return pl.DataFrame(row, schema=schema)


def _obj(value: Raw | None) -> Raw:
    """A json object as ffprobe gave it, or an empty one for a null or a
    missing field."""
    return value or {}


def _probe(ip: StrPath, *flags: str) -> Raw | None:
    try:
        return ffprobe.run(ip, *flags, timeout=PROBE_TIMEOUT, tries=PROBE_TRIES)
    except ffprobe.FFProbeError as e:
        print(f"  [ffprobe] {Path(ip).name}: {e}")
        return None


def _seconds(chapter: Raw, key: str) -> float | None:
    text = chapter.get(f"{key}_time")
    if text is not None:
        try:
            return round(float(text), 3)
        except (TypeError, ValueError):
            pass
    ticks, base = chapter.get(key), chapter.get("time_base")
    if ticks is None or not base:
        return None
    try:
        num, _, den = base.partition("/")
        return round(float(ticks) * float(num) / float(den or 1), 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def create_chapters(ip: StrPath, od: Path) -> None:
    out = od / "chapters.parquet"
    if out.exists():
        print(f"[{od.name}:chapters] Already exists")
        return

    if not Path(ip).exists():
        print(f"[{od.name}:chapters] Missing film at {ip} — skipped")
        return

    t0 = time.monotonic()
    probe = _probe(ip, "-show_chapters")
    if probe is None:
        print(f"[{od.name}:chapters] ffprobe failed — skipped")
        return

    chapters = probe.get("chapters", [])
    if chapters:
        df = pl.DataFrame(
            {
                "fname": [od.name] * len(chapters),
                "cid": [ch.get("id") for ch in chapters],
                "title": [_obj(ch.get("tags")).get("title") for ch in chapters],
                "t0": [_seconds(ch, "start") for ch in chapters],
                "t1": [_seconds(ch, "end") for ch in chapters],
                "present": [True] * len(chapters),
            },
            schema=CHAPTER_SCHEMA,
        )
    else:
        df = _sentinel(od.name, CHAPTER_SCHEMA)

    od.mkdir(parents=True, exist_ok=True)
    from .annotators import stamp

    stamp(df, "chapters").write_parquet(out)

    if chapters:
        print(f"[{od.name}:chapters] {len(chapters)} chapters ({time.monotonic() - t0:.1f}s)")
    else:
        print(f"[{od.name}:chapters] none in the container — recorded as absent")


SRT_TIME = re.compile(
    r"^\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _srt_seconds(h: str, m: str, s: str, ms: str) -> float:
    return round(int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000, 3)


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    lines, i = text.splitlines(), 0
    while i < len(lines):
        m = SRT_TIME.match(lines[i])
        if m is None:
            i += 1
            continue
        t0 = _srt_seconds(*m.group(1, 2, 3, 4))
        t1 = _srt_seconds(*m.group(5, 6, 7, 8))
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i])
            i += 1
        cues.append((t0, t1, "\n".join(body)))
    return cues


def _extract_srt(ip: StrPath, streams: Sequence[int], tmp: Path) -> dict[int, str]:
    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(ip), *args],
            capture_output=True,
            text=True,
            timeout=EXTRACT_TIMEOUT,
        )

    args: list[str] = []
    for sid in streams:
        args += ["-map", f"0:s:{sid}", "-f", "srt", str(tmp / f"{sid}.srt")]
    try:
        r = run(args)
    except subprocess.TimeoutExpired:
        r = None

    got: dict[int, str] = {}
    for sid in streams:
        p = tmp / f"{sid}.srt"
        if p.exists() and p.stat().st_size:
            got[sid] = p.read_text(errors="replace")

    if r is not None and r.returncode == 0:
        return got

    for sid in streams:
        if sid in got:
            continue
        try:
            run(["-map", f"0:s:{sid}", "-f", "srt", str(tmp / f"{sid}.srt")])
        except subprocess.TimeoutExpired:
            continue
        p = tmp / f"{sid}.srt"
        if p.exists() and p.stat().st_size:
            got[sid] = p.read_text(errors="replace")
    return got


def create_subtitles(ip: StrPath, od: Path) -> None:
    out = od / "subtitles.parquet"
    if out.exists():
        print(f"[{od.name}:subtitles] Already exists")
        return

    if not Path(ip).exists():
        print(f"[{od.name}:subtitles] Missing film at {ip} — skipped")
        return

    t0 = time.monotonic()
    probe = _probe(ip, "-show_streams")
    if probe is None:
        print(f"[{od.name}:subtitles] ffprobe failed — skipped")
        return

    subs = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
    text_subs = {
        i: s for i, s in enumerate(subs) if (s.get("codec_name") or "") in TEXT_SUBTITLE_CODECS
    }

    frames: list[pl.DataFrame] = []
    n_cues = 0
    if text_subs:
        with tempfile.TemporaryDirectory(prefix="subs_") as td:
            got = _extract_srt(ip, sorted(text_subs), Path(td))
        for sid in sorted(got):
            cues = parse_srt(got[sid])
            if not cues:
                continue
            st = text_subs[sid]
            tags = _obj(st.get("tags"))
            disp = _obj(st.get("disposition"))
            n_cues += len(cues)
            frames.append(
                pl.DataFrame(
                    {
                        "fname": [od.name] * len(cues),
                        "sid": [sid] * len(cues),
                        "stream": [st.get("index")] * len(cues),
                        "lang": [tags.get("language")] * len(cues),
                        "codec": [st.get("codec_name")] * len(cues),
                        "forced": [bool(disp.get("forced"))] * len(cues),
                        "default": [bool(disp.get("default"))] * len(cues),
                        "eid": list(range(len(cues))),
                        "t0": [c[0] for c in cues],
                        "t1": [c[1] for c in cues],
                        "text": [c[2] for c in cues],
                        "present": [True] * len(cues),
                    },
                    schema=SUBTITLE_SCHEMA,
                )
            )

    df = pl.concat(frames) if frames else _sentinel(od.name, SUBTITLE_SCHEMA)

    od.mkdir(parents=True, exist_ok=True)
    from .annotators import stamp

    stamp(df, "subtitles").write_parquet(out)

    if frames:
        print(
            f"[{od.name}:subtitles] {n_cues} cues over {len(frames)} of "
            f"{len(subs)} track(s) ({time.monotonic() - t0:.1f}s)"
        )
    else:
        skipped = len(subs) - len(text_subs)
        why = f"{skipped} bitmap track(s), no text" if skipped else "no subtitle tracks"
        print(f"[{od.name}:subtitles] {why} — recorded as absent")
