import shutil
import subprocess
from pathlib import Path

import pytest

from dvt.tools import ffprobe, mp4

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def make(path: Path, *args: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=2",
         *args, "-y", str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
@pytest.mark.parametrize(
    "args",
    [
        ("-c:v", "libx264", "-bf", "0"),
        ("-c:v", "libx264", "-bf", "2"),  # display order differs from decode order
        ("-c:v", "libx264", "-output_ts_offset", "1.5"),  # an empty edit in front
        ("-c:v", "mpeg4", "-r", "24000/1001"),
    ],
)
def test_index_times_match_ffprobe(tmp_path: Path, args: tuple[str, ...]) -> None:
    path = make(tmp_path / "a.mp4", *args)
    from_index = mp4.index_times(path)
    from_packets = ffprobe.frame_times(path)
    assert len(from_index) == len(from_packets)
    assert from_index == pytest.approx(from_packets, abs=1e-6)


@needs_ffmpeg
def test_frame_times_of_a_non_mp4_go_to_ffprobe(tmp_path: Path) -> None:
    path = make(tmp_path / "a.mkv", "-c:v", "libx264")
    assert len(mp4.frame_times(path)) == 50


@needs_ffmpeg
def test_fragmented_mp4_is_unhandled(tmp_path: Path) -> None:
    path = make(tmp_path / "a.mp4", "-c:v", "libx264", "-movflags", "frag_keyframe+empty_moov")
    with pytest.raises(mp4.Unhandled):
        mp4.index_times(path)
    assert len(mp4.frame_times(path)) == 50


def test_not_an_mp4(tmp_path: Path) -> None:
    path = tmp_path / "a.mp4"
    path.write_bytes(b"\x00\x00\x00\x08free" * 3)
    with pytest.raises(mp4.Unhandled, match="no moov"):
        mp4.index_times(path)


def test_box_smaller_than_its_header(tmp_path: Path) -> None:
    path = tmp_path / "a.mp4"
    path.write_bytes(b"\x00\x00\x00\x04free")
    with pytest.raises(mp4.Unhandled, match="size 4"):
        mp4.index_times(path)
