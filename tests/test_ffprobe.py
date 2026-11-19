import shutil
import subprocess
from pathlib import Path

import pytest

from dvt.tools import ffprobe

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


@pytest.fixture
def film(tmp_path: Path) -> Path:
    path = tmp_path / "a.mkv"
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:01,000\nHi\n")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=duration=3", "-i", str(srt),
         "-map", "0", "-map", "1", "-map", "2", "-c:v", "libx264", "-c:a", "aac", "-c:s", "srt",
         "-output_ts_offset", "0.5", "-y", str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_streams_and_kinds(film: Path) -> None:
    kinds = [s["codec_type"] for s in ffprobe.streams(film)]
    assert kinds == ["video", "audio", "subtitle"]
    assert [s["codec_name"] for s in ffprobe.streams(film, "audio")] == ["aac"]


@needs_ffmpeg
def test_duration_and_start(film: Path) -> None:
    duration = ffprobe.duration(film)
    assert duration is not None
    assert 3.0 <= duration <= 3.6
    assert ffprobe.start(film) == pytest.approx(0.5, abs=0.05)
    assert ffprobe.start(film, "video") == pytest.approx(0.5, abs=0.05)
    assert ffprobe.chapters(film) == []


@needs_ffmpeg
def test_frame_times(film: Path) -> None:
    times = ffprobe.frame_times(film)
    assert len(times) == 75
    assert times == sorted(times)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ffprobe.FFProbeError, match="1 attempt"):
        ffprobe.probe(tmp_path / "missing.mkv")
    with pytest.raises(ffprobe.FFProbeError):
        ffprobe.frame_times(tmp_path / "missing.mkv")


@needs_ffmpeg
def test_start_of_a_kind_the_file_lacks(film: Path) -> None:
    with pytest.raises(ffprobe.FFProbeError, match="no start time for data"):
        ffprobe.start(film, "data")
