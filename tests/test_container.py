"""Chapters and subtitles, the two annotations already in the file."""

from pathlib import Path

import polars as pl
import pytest
from conftest import make_video, needs_ffmpeg, stamp
from hypothesis import given
from hypothesis import strategies as st

from dvt import container


@pytest.mark.parametrize(
    ("chapter", "expected"),
    [
        ({"start_time": "12.3456"}, 12.346),
        ({"start_time": "x", "start": 900, "time_base": "1/90"}, 10.0),
        ({"start": 12345, "time_base": "1/1000"}, 12.345),
        ({"start": 5, "time_base": "2"}, 10.0),
        ({"start": 5, "time_base": "1/0"}, None),
        ({"start": 5}, None),
        ({}, None),
    ],
)
def test_seconds(chapter: dict[str, object], expected: float | None) -> None:
    assert container._seconds(chapter, "start") == expected  # pyright: ignore[reportPrivateUsage]


def test_parse_srt() -> None:
    text = (
        "1\n00:00:01,500 --> 00:00:02,000\nHello\nthere\n\n"
        "garbage\n"
        "2\n01:02:03.4 --> 01:02:04,05\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nLast\n"
    )
    assert container.parse_srt(text) == [
        (1.5, 2.0, "Hello\nthere"),
        (3723.4, 3724.05, ""),
        (5.0, 6.0, "Last"),
    ]


cue = st.tuples(st.integers(0, 10**7), st.integers(0, 10**7),
                st.text(st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp")),
                        min_size=1).filter(lambda t: t.strip() == t and t != ""))  # fmt: skip


@given(st.lists(cue, max_size=6))
def test_parse_srt_reads_what_srt_writes(cues: list[tuple[int, int, str]]) -> None:
    text = "".join(f"{i + 1}\n{stamp(a / 1000)} --> {stamp(b / 1000)}\n{t}\n\n"
                   for i, (a, b, t) in enumerate(cues))  # fmt: skip
    assert container.parse_srt(text) == [(a / 1000, b / 1000, t) for a, b, t in cues]


@needs_ffmpeg
def test_chapters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = make_video(tmp_path / "v.mp4", chapters=[(0, 1.5, "One"), (1.5, 4, "Two")])
    od = tmp_path / "Film"
    container.create_chapters(video, od)
    df = pl.read_parquet(od / "chapters.parquet")
    assert df.select("fname", "cid", "title", "t0", "t1", "present").rows() == [
        ("Film", 0, "One", 0.0, 1.5, True),
        ("Film", 1, "Two", 1.5, 4.0, True),
    ]
    assert "[Film:chapters] 2 chapters" in capsys.readouterr().out


@needs_ffmpeg
def test_no_chapters_or_subtitles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = make_video(tmp_path / "v.mp4")
    od = tmp_path / "Film"
    container.create_chapters(video, od)
    container.create_subtitles(video, od)
    for table, schema in (("chapters", container.CHAPTER_SCHEMA),
                          ("subtitles", container.SUBTITLE_SCHEMA)):  # fmt: skip
        df = pl.read_parquet(od / f"{table}.parquet")
        assert df.drop("version").schema == pl.Schema(schema)
        assert df.select("fname", "present").rows() == [("Film", False)]
    out = capsys.readouterr().out
    assert "[Film:chapters] none in the container — recorded as absent" in out
    assert "[Film:subtitles] no subtitle tracks — recorded as absent" in out
    container.create_chapters(video, od)
    container.create_subtitles(video, od)
    assert capsys.readouterr().out.count("Already exists") == 2


@needs_ffmpeg
def test_subtitles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = make_video(tmp_path / "v.mp4",
                       subtitles=[(0.5, 1.5, "Bonjour"), (2.0, 3.25, "Salut\nami")])  # fmt: skip
    od = tmp_path / "Film"
    container.create_subtitles(video, od)
    df = pl.read_parquet(od / "subtitles.parquet")
    assert df.select("sid", "stream", "lang", "codec", "forced", "default", "eid", "t0", "t1",
                     "text", "present").rows() == [
        (0, 2, "fra", "mov_text", False, True, 0, 0.5, 1.5, "Bonjour", True),
        (0, 2, "fra", "mov_text", False, True, 1, 2.0, 3.25, "Salut\nami", True),
    ]  # fmt: skip
    assert "[Film:subtitles] 2 cues over 1 of 1 track(s)" in capsys.readouterr().out


def test_unreadable(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                    monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.setattr(container, "PROBE_TRIES", 1)
    junk = tmp_path / "v.mp4"
    junk.write_text("no")
    container.create_chapters(junk, tmp_path / "Film")
    container.create_subtitles(junk, tmp_path / "Film")
    container.create_chapters(tmp_path / "none.mp4", tmp_path / "Film")
    out = capsys.readouterr().out
    assert "[Film:chapters] ffprobe failed — skipped" in out
    assert "[Film:subtitles] ffprobe failed — skipped" in out
    assert "[Film:chapters] Missing film at" in out
    assert not (tmp_path / "Film").exists()
