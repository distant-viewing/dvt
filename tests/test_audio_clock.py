from pathlib import Path

import polars as pl
import pytest
from conftest import make_video, needs_ffmpeg

from dvt import audio_clock


def test_shifted() -> None:
    df = pl.DataFrame({"start": [0.0, 1.0], "end": [0.5, 2.0], "word": ["a", "b"]})
    out = audio_clock.shifted(df, "words.parquet", 0.25)
    assert out.rows() == [(0.25, 0.75, "a"), (1.25, 2.25, "b")]
    assert audio_clock.shifted(df, "words.parquet", 0.0) is df
    praat = pl.DataFrame({"f0": [1.0]})
    assert audio_clock.shifted(praat, "praat.parquet", 1.0).equals(praat)


def test_write_and_recorded(tmp_path: Path) -> None:
    path = tmp_path / "words.parquet"
    audio_clock.write(pl.DataFrame({"start": [1.0], "end": [2.0]}), "words", path, 0.5)
    df = pl.read_parquet(path)
    assert df.rows() == [(1.5, 2.5, 1)]
    assert audio_clock.recorded(path) == 0.5
    pl.DataFrame({"start": [1.0]}).write_parquet(tmp_path / "old.parquet")
    assert audio_clock.recorded(tmp_path / "old.parquet") is None


@needs_ffmpeg
def test_start(tmp_path: Path) -> None:
    from dvt.tools.audio import extract_audio

    video = make_video(tmp_path / "v.mp4")
    m4a = tmp_path / "v.m4a"
    extract_audio(video, m4a)
    assert audio_clock.start(video, m4a) == pytest.approx(0.0, abs=0.05)
