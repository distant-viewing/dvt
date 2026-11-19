import shutil
import subprocess
from pathlib import Path

import pytest

from dvt.tools import audio, filehash


@pytest.mark.parametrize(
    ("stream", "args", "how"),
    [
        ({"codec_name": "aac"}, ["-c:a", "copy"], "copy"),
        ({"codec_name": "ac3", "bit_rate": "640000"}, ["-c:a", "aac", "-b:a", "320k"],
         "ac3 -> aac 320k"),
        ({"codec_name": "mp2", "bit_rate": "64000"}, ["-c:a", "aac", "-b:a", "128k"],
         "mp2 -> aac 128k"),
        ({"bit_rate": "N/A"}, ["-c:a", "aac", "-b:a", "192k"], "unknown -> aac 192k"),
    ],
)  # fmt: skip
def test_encode_args(stream: dict[str, object], args: list[str], how: str) -> None:
    assert audio.encode_args(stream) == (args, how)


@pytest.mark.parametrize(("value", "n"), [(6, 6), ("2", 2), (None, 0), ("x", 0)])
def test_channels_of(value: object, n: int) -> None:
    assert audio.channels_of({"channels": value}) == n
    assert audio.channels_of({}) == 0


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def make(path: Path, *audio_inputs: str) -> Path:
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:duration=1"]
    for source in audio_inputs:
        cmd += ["-f", "lavfi", "-i", source]
    cmd += [arg for i in range(len(audio_inputs) + 1) for arg in ("-map", str(i))]
    subprocess.run([*cmd, "-c:v", "libx264", "-c:a", "ac3", "-y", str(path)], check=True)
    return path


@needs_ffmpeg
def test_extract_takes_the_film_stream(tmp_path: Path) -> None:
    src = make(tmp_path / "a.mkv", "sine=duration=1", "sine=duration=1:sample_rate=48000")
    out = tmp_path / "out" / "a.m4a"
    desc = audio.extract_audio(src, out)
    assert desc.startswith("a:0 1ch ")
    assert desc.endswith("ac3 -> aac 128k")
    assert out.exists()
    assert sorted(p.name for p in out.parent.iterdir()) == ["a.m4a"]


@needs_ffmpeg
def test_extract_from_a_silent_file_leaves_the_marker(tmp_path: Path) -> None:
    src = make(tmp_path / "a.mkv")
    out = tmp_path / "a.m4a"
    assert audio.extract_audio(src, out) == "no audio stream"
    assert filehash.noaudio_path_for(out).read_bytes() == filehash.NOAUDIO_CONTENT
    assert not out.exists()


def test_probe_of_nothing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        audio.probe_audio_streams(tmp_path / "missing.mkv")
