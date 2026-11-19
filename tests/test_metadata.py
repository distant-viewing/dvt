from pathlib import Path

import polars as pl
import pytest
from conftest import make_video, needs_ffmpeg

from dvt import metadata
from dvt.source import Source
from dvt.tools import filehash
from dvt.tools.audio import extract_audio


@needs_ffmpeg
def test_create_meta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = make_video(tmp_path / "Film.mp4", seconds=6, bars=(20, 20, 0, 0), lang="ger")
    s = Source.under(video, tmp_path / "out")
    extract_audio(s.video, s.audio)
    s.out.mkdir(parents=True)  # video.process_mp4 makes it
    metadata.create_meta(s)
    row = pl.read_parquet(s.out / "metadata.parquet").row(0, named=True)
    assert (row["n_frames"], row["fps"], row["img_width"], row["img_height"]) == (
        60,
        10.0,
        160,
        160,
    )
    assert (row["crop_top"], row["crop_bottom"], row["crop_left"], row["crop_right"]) == (
        20, 20, 0, 0)  # fmt: skip
    assert row["crop_usable"]
    assert row["crop_stable"]
    assert row["crop_frac"] == pytest.approx(40 / 160)
    assert row["ar_known"]
    assert row["display_ar"] == pytest.approx(160 / 120)
    assert row["ar_nearest"] == 1.33
    assert (row["mp4_bytes"], row["mp4_hash"]) == filehash.size_and_hash(video)
    assert (row["m4a_bytes"], row["m4a_hash"]) == filehash.size_and_hash(s.audio)
    assert (row["language_code"], row["language_declared"]) == ("de", "deu")
    assert row["video_start"] == pytest.approx(0.0, abs=0.05)
    assert row["audio_start"] == pytest.approx(0.0, abs=0.05)
    assert row["version"] == 1
    out = capsys.readouterr().out
    assert "[Film:metadata] crop t/b/l/r=20/20/0/0 (25.0% bar)" in out
    assert "[Film:metadata] aspect 1.333 (nearest 1.33, 0.003 away)" in out
    assert "[Film:metadata] language de from the tag deu, per its container" in out
    metadata.create_meta(s)
    assert capsys.readouterr().out == "[Film:metadata] Already exists\n"


@needs_ffmpeg
def test_a_film_with_no_sound(tmp_path: Path) -> None:
    video = make_video(tmp_path / "Silent.mp4", audio=False)
    s = Source.under(video, tmp_path / "out")
    extract_audio(s.video, s.audio)
    s.out.mkdir(parents=True)  # video.process_mp4 makes it
    metadata.create_meta(s)
    row = pl.read_parquet(s.out / "metadata.parquet").row(0, named=True)
    assert row["audio_start"] is None
    assert row["m4a_bytes"] == len(filehash.NOAUDIO_CONTENT)
    assert row["language_code"] is None


@needs_ffmpeg
def test_no_sound_taken_out(tmp_path: Path) -> None:
    s = Source.under(make_video(tmp_path / "Film.mp4"), tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="no audio recorded"):
        metadata.create_meta(s)
    assert not (s.out / "metadata.parquet").exists()


@needs_ffmpeg
def test_detect_crop_of_junk(tmp_path: Path) -> None:
    junk = tmp_path / "j.mp4"
    junk.write_text("no")
    assert not metadata.detect_crop(junk, 160, 120)["crop_usable"]
