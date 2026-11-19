"""What ffprobe says about one video file: the facts the inventory keeps a row of."""

import json
import subprocess
import sys
import time
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import TypedDict, cast

from . import language
from .ffprobe import TEXT_SUBTITLE_CODECS
from .values import Raw, float_or_none, int_or_none, mapping, mappings, text

FPS_MIN, FPS_MAX = 1.0, 121.0


STANDARD_NAMES = {
    (720, 480): "SD",
    (1280, 720): "HD",
    (1920, 1080): "FHD",
    (3840, 2160): "UHD",
}


PROBE_TIMEOUT = 120
PROBE_TRIES = 3

Stream = Mapping[str, object]


class Meta(TypedDict):
    """The inventory's row for one file, as extract_meta makes it."""

    duration_sec: float | None
    fps: float | None
    fps_r: float | None
    video_codec: str | None
    audio_codec: str | None
    width: int | None
    height: int | None
    sar: str
    dar: str | None
    field_order: str | None
    video_bitrate_kbps: float | None
    audio_bitrate_kbps: float | None
    audio_streams: int
    audio_channels: int | None
    audio_channels_max: int | None
    audio_layout: str | None
    audio_sample_rate: int | None
    audio_profile: str | None
    audio_lang: str | None
    audio_lang_mixed: bool | None
    audio_track: int | None
    audio_tracks: str | None
    audio_start_sec: float | None
    audio_duration_sec: float | None
    subtitle_streams: int
    subtitle_text_streams: int
    subtitle_text_samples_max: int | None
    subtitle_tracks: str | None
    chapters: int
    chapter_starts: str | None


def probe_video(filepath: Path) -> Raw | None:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format", "-show_chapters",
        str(filepath),
    ]  # fmt: skip
    err = "not run"
    for attempt in range(1, PROBE_TRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
            if result.returncode == 0:
                return cast("Raw", json.loads(result.stdout))  # ffprobe writes an object
            err = result.stderr.strip() or f"exit {result.returncode}"
        except subprocess.TimeoutExpired:
            err = f"timed out after {PROBE_TIMEOUT}s"
        except Exception as e:
            err = str(e)
        if attempt < PROBE_TRIES:
            time.sleep(2 * attempt)
        else:
            print(f"  [ERROR] {filepath.name}: {err} ({PROBE_TRIES} attempts)", file=sys.stderr)
    return None


def parse_fps(fps_str: str) -> float | None:
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return round(int(num) / int(den), 4) if int(den) != 0 else None
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return None


def parse_ratio(text: str | None) -> float | None:
    if not text:
        return None
    for sep in (":", "/"):
        if sep in text:
            a, b = text.split(sep, 1)
            try:
                num, den = float(a), float(b)
            except ValueError:
                return None
            return num / den if den else None
    try:
        return float(text)
    except ValueError:
        return None


def chapter_start(chapter: Stream) -> float | None:
    start_time = chapter.get("start_time")
    if start_time is None:
        ticks, base = chapter.get("start"), parse_ratio(text(chapter.get("time_base")))
        if ticks is None or not base:
            return None
        seconds = float_or_none(ticks)
        return round(seconds * base, 3) if seconds is not None else None
    seconds = float_or_none(start_time)
    return round(seconds, 3) if seconds is not None else None


def pick_fps(video: Stream, duration: float | None) -> tuple[float | None, float | None]:
    fps_r = parse_fps(text(video.get("r_frame_rate")) or "")

    for key in ("avg_frame_rate", "r_frame_rate"):
        val = text(video.get(key))
        if not val or val in ("0/0", "0"):
            continue
        fps = parse_fps(val)
        if fps and FPS_MIN <= fps <= FPS_MAX:
            return fps, fps_r

    nb = video.get("nb_frames")
    if nb and duration:
        frames = int_or_none(nb)
        if frames is None:
            return None, fps_r
        fps = round(frames / duration, 4)
        if FPS_MIN <= fps <= FPS_MAX:
            return fps, fps_r
    return None, fps_r


def _tag(stream: Stream | None, name: str) -> str | None:
    if not stream:
        return None
    tags = mapping(stream.get("tags"))
    return (text(tags.get(name)) if tags else None) or None


def _int_field(stream: Stream | None, key: str) -> int | None:
    return int_or_none(stream.get(key)) if stream is not None else None


def _seconds(stream: Stream, key: str) -> float | None:
    seconds = float_or_none(stream.get(key))
    return round(seconds, 3) if seconds is not None else None


def _bitrate_kbps(stream: Stream | None) -> float | None:
    val = stream.get("bit_rate") if stream else None
    if val:
        bits = int_or_none(val)
        if bits is not None:
            return round(bits / 1000, 1)
    return None


def _clean(field: str) -> str:
    return field.replace(":", "").replace("|", "")


def _track_field(stream: Stream) -> str:
    lang = _clean(_tag(stream, "language") or "und")
    chan = _int_field(stream, "channels") or ""
    br = _bitrate_kbps(stream)
    return f"{lang}:{chan}:{br:.0f}" if br else f"{lang}:{chan}:"


def _sub_field(stream: Stream) -> str:
    lang = _clean(_tag(stream, "language") or "und")
    codec = _clean(text(stream.get("codec_name")) or "unknown")
    disp = mapping(stream.get("disposition")) or {}
    n = _int_field(stream, "nb_frames")
    return (
        f"{lang}:{codec}:{n if n is not None else ''}"
        f":{1 if disp.get('forced') else 0}"
        f":{1 if disp.get('default') else 0}"
    )


def extract_meta(probe: Mapping[str, object]) -> Meta:
    streams = mappings(probe.get("streams", []))
    fmt = mapping(probe.get("format", {})) or {}

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    # The audio_* columns describe the stream the extraction takes for the
    # m4a, while audio_lang is the first stream's tag, which sets the
    # language whichever stream is taken.
    track = language.pick_stream(audios) if audios else None
    audio = audios[track] if track is not None else None
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    chapters = mappings(probe.get("chapters", []))

    duration = None
    for src in [video, audio, fmt]:
        raw = src.get("duration") if src else None
        if raw:
            seconds = float_or_none(raw)
            if seconds is not None:
                duration = round(seconds, 3)
                break

    fps, fps_r = pick_fps(video, duration) if video else (None, None)

    video_codec = text(video.get("codec_name")) if video else None
    audio_codec = text(audio.get("codec_name")) if audio else None
    width = int_or_none(video.get("width")) if video and video.get("width") else None
    height = int_or_none(video.get("height")) if video and video.get("height") else None

    sar = text(video.get("sample_aspect_ratio")) if video else None
    if not sar or sar.startswith("0:"):
        sar = "1:1"

    dar = text(video.get("display_aspect_ratio")) if video else None
    if (not dar or dar.startswith("0:")) and width and height:
        num, den = ([*sar.split(":"), "1"])[:2]
        try:
            f = Fraction(width * int(num), height * int(den))
            dar = f"{f.numerator}:{f.denominator}"
        except (ValueError, ZeroDivisionError):
            dar = None

    field_order = text(video.get("field_order")) if video else None

    video_br = _bitrate_kbps(video)
    if video_br is None and fmt.get("bit_rate"):
        bits = int_or_none(fmt.get("bit_rate"))
        if bits is not None:
            video_br = round(bits / 1000, 1)

    def is_text(stream: Stream) -> bool:
        return (text(stream.get("codec_name")) or "") in TEXT_SUBTITLE_CODECS

    channel_counts = [c for c in (_int_field(s, "channels") for s in audios) if c]
    text_samples = [
        n for n in (_int_field(s, "nb_frames") for s in subs if is_text(s)) if n is not None
    ]
    chapter_times = [t for t in (chapter_start(ch) for ch in chapters) if t is not None]

    return {
        "duration_sec": duration,
        "fps": fps,
        "fps_r": fps_r,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "width": width,
        "height": height,
        "sar": sar,
        "dar": dar,
        "field_order": field_order,
        "video_bitrate_kbps": video_br,
        "audio_bitrate_kbps": _bitrate_kbps(audio),
        "audio_streams": len(audios),
        "audio_channels": _int_field(audio, "channels"),
        "audio_channels_max": max(channel_counts) if channel_counts else None,
        "audio_layout": text(audio.get("channel_layout")) if audio else None,
        "audio_sample_rate": _int_field(audio, "sample_rate"),
        "audio_profile": text(audio.get("profile")) if audio else None,
        "audio_lang": language.normalise(_tag(audios[0], "language")) if audios else None,
        "audio_lang_mixed": language.tags_differ(audios) if audios else None,
        "audio_track": track,
        "audio_tracks": "|".join(_track_field(st) for st in audios) or None,
        "audio_start_sec": _seconds(audio, "start_time") if audio else None,
        "audio_duration_sec": _seconds(audio, "duration") if audio else None,
        "subtitle_streams": len(subs),
        "subtitle_text_streams": sum(1 for st in subs if is_text(st)),
        "subtitle_text_samples_max": max(text_samples, default=None),
        "subtitle_tracks": "|".join(_sub_field(st) for st in subs) or None,
        "chapters": len(chapters),
        "chapter_starts": "|".join(f"{t:.3f}" for t in chapter_times) or None,
    }
