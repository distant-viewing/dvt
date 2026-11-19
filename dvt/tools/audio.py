"""Takes the sound out of a video file as an m4a, or leaves a marker saying it has none."""

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from . import language
from .filehash import NOAUDIO_CONTENT, NOAUDIO_SUFFIX
from .values import Raw, int_or_none

REENCODE_KBPS_MIN = 128
REENCODE_KBPS_MAX = 320
REENCODE_KBPS_DEFAULT = 192

PROBE_TIMEOUT = 120


def probe_audio_streams(path: Path) -> list[Raw]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-select_streams",
            "a",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip() or result.returncode}")
    found: list[Raw] = json.loads(result.stdout).get("streams", [])
    return found


def channels_of(stream: Mapping[str, object]) -> int:
    return int_or_none(stream.get("channels")) or 0


def encode_args(stream: Mapping[str, object]) -> tuple[list[str], str]:
    if stream.get("codec_name") == "aac":
        return ["-c:a", "copy"], "copy"

    bits = int_or_none(stream.get("bit_rate"))
    kbps = round(bits / 1000) if bits is not None else REENCODE_KBPS_DEFAULT
    kbps = min(REENCODE_KBPS_MAX, max(REENCODE_KBPS_MIN, kbps))
    codec = stream.get("codec_name") or "unknown"
    return ["-c:a", "aac", "-b:a", f"{kbps}k"], f"{codec} -> aac {kbps}k"


def extract_audio(input_path: Path, output_path: Path) -> str:
    streams = probe_audio_streams(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not streams:
        marker = output_path.with_name(output_path.stem + NOAUDIO_SUFFIX)
        tmp_marker = marker.with_name(f".{marker.name}.part")
        tmp_marker.write_bytes(NOAUDIO_CONTENT)
        tmp_marker.replace(marker)
        return "no audio stream"

    track = language.pick_stream(streams)
    stream = streams[track]
    codec_args, how = encode_args(stream)
    desc = f"a:{track} {channels_of(stream)}ch {stream.get('sample_rate', '?')}Hz {how}"

    tmp_path = output_path.with_name(f".{output_path.stem}.part{output_path.suffix}")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-map",
                f"0:a:{track}",
                "-map_chapters",
                "-1",
                *codec_args,
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit code {result.returncode}):\n{result.stderr}")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(output_path)
    return desc
