"""The parts that say what runs and where: the annotator list, the model
directory, a Source, the language a film is in, and the command line."""

import logging
import os
import time
import warnings
from pathlib import Path

import polars as pl
import pytest
from conftest import make_video, needs_ffmpeg

from dvt import align_paths, annotators, cli, config, language, log, noise, source
from dvt.annotators import ANNOTATORS
from dvt.source import Source

# ---- annotators ---------------------------------------------------------------


def test_metadata_comes_first() -> None:
    assert next(iter(ANNOTATORS)) == "metadata"


def test_every_requirement_is_an_annotator() -> None:
    for est in ANNOTATORS.values():
        assert est.side in annotators.SIDES
        assert est.source in annotators.SOURCES
        assert set(est.requires) <= set(ANNOTATORS)


def test_sides() -> None:
    assert annotators.side_of("words") == annotators.CPU_AUDIO
    assert annotators.names_for_side(annotators.GPU_AUDIO) == ["audioseg"]
    assert annotators.names_for_side(annotators.CPU_AUDIO) == ["words", "praat", "laughter"]
    assert annotators.ANNOTATORS["laughter"].requires == ("audioseg", "words", "praat")
    assert annotators.names_for_side(annotators.CPU) == [
        "metadata", "letterbox", "chroma", "chapters", "subtitles", "frames", "shots",
        "frames_r720"]  # fmt: skip
    assert annotators.outputs_of("tracks") == ("tracks.parquet", "tracks_frames.parquet")


def test_stamp() -> None:
    df = annotators.stamp(pl.DataFrame({"a": [1, 2]}), "chroma")
    assert df.schema["version"] == pl.Int32
    assert df["version"].to_list() == [1, 1]


# ---- config -------------------------------------------------------------------


@pytest.fixture
def fresh_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_models", None)
    monkeypatch.setattr(config, "_tmp", None)
    monkeypatch.delenv(config.MODELS_ENV, raising=False)
    monkeypatch.setenv("TORCH_HOME", "unset")


def test_configure(fresh_config: None, tmp_path: Path) -> None:
    config.configure(tmp_path / "m", tmp_path / "t")
    assert config.models() == tmp_path / "m"
    assert config.tmp() == tmp_path / "t"
    assert os.environ["TORCH_HOME"] == str(tmp_path / "m" / "torch")
    assert config.whisper() == tmp_path / "m" / "whisper-large-v3-ct2"
    assert config.nltk_data() == tmp_path / "m" / "nltk_data"
    assert config.separator() == tmp_path / "m" / "audio-separator"


def test_models_from_the_environment(
    fresh_config: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.MODELS_ENV, str(tmp_path))
    assert config.models() == tmp_path
    assert config.tmp().name == "dvt"


def test_models_by_default(fresh_config: None) -> None:
    assert config.models() == config.DEFAULT_MODELS
    assert os.environ["TORCH_HOME"] == str(config.DEFAULT_MODELS / "torch")


# ---- source -------------------------------------------------------------------


@needs_ffmpeg
def test_source_under(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in" / "Film_1950_Feature.mp4", lang="fre")
    s = Source.under(video, tmp_path / "out")
    assert s.name == "Film_1950_Feature"
    assert s.out == tmp_path / "out" / "annotations" / "Film_1950_Feature"
    assert s.audio == tmp_path / "out" / "audio" / "Film_1950_Feature.m4a"
    assert s.frames == tmp_path / "out" / "frames" / "Film_1950_Feature"
    assert s.speech == tmp_path / "out" / "stems" / "speech" / "Film_1950_Feature.flac"
    assert s.stem("drums") == tmp_path / "out" / "stems" / "drums" / "Film_1950_Feature.flac"
    assert s.sound is None
    assert s.container is not None
    assert (s.container["width"], s.container["audio_lang"]) == (160, "fra")


def test_source_of_an_unreadable_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_text("no")
    assert Source.under(junk, tmp_path).container is None


def test_sound(tmp_path: Path) -> None:
    s = Source(video=tmp_path / "v.mp4", out=tmp_path / "o", audio=tmp_path / "a.m4a",
               frames=tmp_path / "f", stems=str(tmp_path / source.STEM))  # fmt: skip
    assert s.sound is None
    (tmp_path / "a.m4a").write_bytes(b"x")
    assert s.sound == tmp_path / "a.m4a"


# ---- language -----------------------------------------------------------------


def src(container: dict[str, object] | None) -> Source:
    p = Path("x")
    return Source(video=p, out=Path("o/Film"), audio=p, frames=p, stems="s", container=container)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (None, (None, None, "no audio stream, per its container")),
        ("eng", ("en", "eng", "en from the tag eng, per its container")),
        ("fre", ("fr", "fra", "fr from the tag fra, per its container")),
        ("zxx", (None, "zxx", "no linguistic content (zxx), per its container")),
        ("und", (None, "und", "language not yet decided (und), per its container")),
        ("tha", (None, "tha", "th from the tag tha, which whisperx has no aligner for")),
        (
            "yid",
            (
                None,
                "yid",
                "tagged yid, which is not a language this pipeline knows, per its container",
            ),
        ),
    ],
)
def test_from_tag(tag: str | None, expected: tuple[str | None, str | None, str]) -> None:
    got = language.from_tag(tag, "its container")
    assert (got.code, got.declared, got.note) == expected


def test_resolve() -> None:
    assert language.resolve(src(None)).note == "nothing is known of the container of Film"
    assert language.resolve(src({"audio_lang": "deu"})).code == "de"
    assert language.resolve(src({})).note == "no audio stream, per its container"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (None, (None, None, "metadata.parquet records no audio language for this film")),
        ("unk", (None, "unk", "unk per the curated table, in metadata.parquet")),
        ("fr", ("fr", "fr", "fr per the curated table, per metadata.parquet")),
        ("yi", (None, "yi", "yi per the curated table, which whisperx has no aligner for")),
        ("ita", ("it", "ita", "it from the tag ita, per metadata.parquet")),
    ],
)
def test_from_metadata(declared: str | None, expected: tuple[str | None, str | None, str]) -> None:
    got = language.from_metadata(declared)
    assert (got.code, got.declared, got.note) == expected


def test_aligners() -> None:
    assert "en" in language.alignable()
    assert align_paths.resolve_align_model("en")[0] == align_paths.TORCHAUDIO
    assert align_paths.resolve_align_model("ja")[0] == align_paths.HUGGINGFACE
    assert align_paths.resolve_align_model("xx", "some/model") == ("huggingface", "some/model")
    with pytest.raises(ValueError, match="no default aligner"):
        align_paths.resolve_align_model("xx")


def test_align_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_models", tmp_path)
    assert align_paths.align_model_dir("en") is None
    assert align_paths.align_model_dir("ja") == str(tmp_path)


# ---- log and noise ------------------------------------------------------------


def test_log(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    od = Path("out/Film")
    log.log_status(od, "frames", "Already exists")
    monkeypatch.setattr(time, "monotonic", lambda: 3725.0)
    log.log_finished(od, "frames", 0.0)
    assert capsys.readouterr().out == (
        "[Film:frames] Already exists\n[Film:frames] Finished (01:02:05)\n"
    )


def test_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TBM_NOISE", raising=False)
    with warnings.catch_warnings():
        noise.silence()
        assert logging.getLogger("insightface").level == logging.ERROR
    monkeypatch.setenv("TBM_NOISE", "1")
    noise.silence()


def test_quiet_partial_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TBM_NOISE", raising=False)
    lg = logging.getLogger("transformers.modeling_utils")
    with noise.quiet_partial_load():
        drop = lg.filters[-1]
        keep = logging.LogRecord("x", logging.INFO, "", 0, "LOAD REPORT: MISSING a", None, None)
        lose = logging.LogRecord("x", logging.INFO, "", 0, "LOAD REPORT: UNEXPECTED", None, None)
        other = logging.LogRecord("x", logging.INFO, "", 0, "hello", None, None)
        assert isinstance(drop, logging.Filter)
        assert [drop.filter(r) for r in (keep, lose, other)] == [True, False, True]
    assert drop not in lg.filters


def test_quiet_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TBM_NOISE", raising=False)
    logging.getLogger("whisperx.asr").setLevel(logging.DEBUG)
    noise.quiet_loggers()
    assert logging.getLogger("whisperx.asr").level == logging.WARNING


# ---- cli ----------------------------------------------------------------------


def test_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["metadata", "cpu", "metadata.parquet"]
    assert "after shots, face, yolo" in out[list(ANNOTATORS).index("tracks")]


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (["--annotators", "frames,nope"], "a path and --out are both needed"),
        ([], "a path and --out are both needed"),
        (["missing.mp4", "--out", "o", "--models", "m"], "FileNotFoundError: missing.mp4"),
    ],
)
def test_usage_errors(argv: list[str], said: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert said in capsys.readouterr().err


def test_no_models_on_the_command_line(
    fresh_config: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    argv = [str(video), "--out", str(tmp_path), "--models", str(tmp_path / "m")]
    with pytest.raises(SystemExit):
        cli.main([*argv, "--annotators", "frames,nope"])
    assert "not annotators of a video: nope" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([*argv, "--annotators", "metadata,frames,yolo"])
    err = capsys.readouterr().err
    assert f"no model for frames, yolo in {tmp_path / 'm'}" in err
    assert "`dvt models fetch` downloads them" in err
