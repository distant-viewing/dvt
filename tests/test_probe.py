import contextlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fixtures import EMPTY, FILM, SILENT
from hypothesis import given
from hypothesis import strategies as st

from dvt.tools import probe

# What extract_meta returned for the fixtures before the package took up the
# strict checks; the inventory is built from these rows, so they must not move.
FILM_META = {
    "duration_sec": 5400.25, "fps": 23.976, "fps_r": 23.976, "video_codec": "h264",
    "audio_codec": "dts", "width": 1440, "height": 1080, "sar": "4:3", "dar": "16:9",
    "field_order": "progressive", "video_bitrate_kbps": 4500.0, "audio_bitrate_kbps": 1536.0,
    "audio_streams": 3, "audio_channels": 6, "audio_channels_max": 6,
    "audio_layout": "5.1(side)", "audio_sample_rate": 48000, "audio_profile": "DTS",
    "audio_lang": "fra", "audio_lang_mixed": True, "audio_track": 1,
    "audio_tracks": "fre:2:192|fra:6:1536|eng:2:128", "audio_start_sec": 0.0,
    "audio_duration_sec": None, "subtitle_streams": 2, "subtitle_text_streams": 1,
    "subtitle_text_samples_max": 1200,
    "subtitle_tracks": "eng:subrip:1200:0:1|und:hdmv_pgs_subtitle::1:0", "chapters": 4,
    "chapter_starts": "0.000|90.090",
}  # fmt: skip

NOTHING = {
    "duration_sec": None, "fps": None, "fps_r": None, "video_codec": None,
    "audio_codec": None, "width": None, "height": None, "sar": "1:1", "dar": None,
    "field_order": None, "video_bitrate_kbps": None, "audio_bitrate_kbps": None,
    "audio_streams": 0, "audio_channels": None, "audio_channels_max": None,
    "audio_layout": None, "audio_sample_rate": None, "audio_profile": None,
    "audio_lang": None, "audio_lang_mixed": None, "audio_track": None, "audio_tracks": None,
    "audio_start_sec": None, "audio_duration_sec": None, "subtitle_streams": 0,
    "subtitle_text_streams": 0, "subtitle_text_samples_max": None, "subtitle_tracks": None,
    "chapters": 0, "chapter_starts": None,
}  # fmt: skip

SILENT_META = {
    **NOTHING, "duration_sec": 10.01, "fps": 29.97, "fps_r": 29.97,
    "video_codec": "mpeg2video", "width": 720, "height": 480, "dar": "3:2",
    "video_bitrate_kbps": 8000.0,
}  # fmt: skip


@pytest.mark.parametrize(
    ("report", "meta"), [(FILM, FILM_META), (SILENT, SILENT_META), (EMPTY, NOTHING)]
)
def test_extract_meta(report: dict[str, object], meta: dict[str, object]) -> None:
    assert probe.extract_meta(report) == meta


@pytest.mark.parametrize(
    ("text", "fps"),
    [("24000/1001", 23.976), ("25", 25.0), ("30/0", None), ("x", None), ("", None)],
)
def test_parse_fps(text: str, fps: float | None) -> None:
    assert probe.parse_fps(text) == fps


@pytest.mark.parametrize(
    ("text", "ratio"),
    [("16:9", 16 / 9), ("1/1000", 0.001), ("2", 2.0), ("4:0", None), ("a:b", None),
     (None, None), ("", None)],
)  # fmt: skip
def test_parse_ratio(text: str | None, ratio: float | None) -> None:
    assert probe.parse_ratio(text) == ratio


@pytest.mark.parametrize(
    ("chapter", "start"),
    [({"start_time": "12.3456"}, 12.346), ({"start": 1500, "time_base": "1/1000"}, 1.5),
     ({"start": 1500}, None), ({"start_time": None, "start": "x", "time_base": "1/2"}, None),
     ({}, None)],
)  # fmt: skip
def test_chapter_start(chapter: dict[str, object], start: float | None) -> None:
    assert probe.chapter_start(chapter) == start


def test_pick_fps_falls_back_to_frame_count() -> None:
    video = {"avg_frame_rate": "0/0", "r_frame_rate": "90000/1", "nb_frames": "250"}
    assert probe.pick_fps(video, 10.0) == (25.0, 90000.0)


def test_pick_fps_gives_up() -> None:
    assert probe.pick_fps({"r_frame_rate": "1000/1"}, None) == (None, 1000.0)


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=25,
)


@given(json_values)
def test_extract_meta_survives_odd_streams(value: object) -> None:
    """Whatever ffprobe puts in a stream, the row is still made or the error
    is one of the kinds the caller already expects from bad data."""
    for report in ({"streams": [value]}, {"streams": [{"codec_type": "video", "x": value}]}):
        with contextlib.suppress(AttributeError, TypeError, KeyError, ValueError, IndexError):
            probe.extract_meta(report)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


@needs_ffmpeg
def test_probe_video_on_a_real_file(tmp_path: Path) -> None:
    mkv = tmp_path / "a.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
         "-f", "lavfi", "-i", "sine=duration=2", "-c:v", "libx264", "-c:a", "aac",
         "-metadata:s:a:0", "language=ger", "-y", str(mkv)],
        check=True,
    )  # fmt: skip
    report = probe.probe_video(mkv)
    assert report is not None
    meta = probe.extract_meta(report)
    assert (meta["width"], meta["height"], meta["fps"]) == (320, 240, 25.0)
    assert meta["audio_lang"] == "deu"
    assert json.dumps(meta)  # every value is one the parquet writer can take


def test_probe_video_on_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr(probe.time, "sleep", waits.append)
    assert probe.probe_video(tmp_path / "missing.mkv") is None
    assert waits == [2, 4]  # three tries, backing off between them
