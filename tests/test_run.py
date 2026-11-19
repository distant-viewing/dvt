"""run calls the annotators in their order and skips what has no input.
The annotators themselves are stood in for, so no model is loaded."""

import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from dvt import run
from dvt.source import STEM, Source

Call = tuple[str, tuple[object, ...]]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Stand-ins for the modules run imports, each recording its calls."""
    seen: list[Call] = []

    def record(name: str) -> Callable[..., None]:
        def f(*args: object) -> None:
            seen.append((name, args))

        return f

    def none() -> list[str]:
        return []

    def words(stem: Path, od: Path, start: float) -> None:
        seen.append(("words", (stem, od, start)))

    fakes: dict[str, dict[str, Callable[..., None] | Callable[[], list[str]]]] = {
        "dvt.video": {"process_mp4": record("video"), "check_imports": record("check")},
        "dvt.audioseg": {"process_audioseg": record("audioseg")},
        "dvt.words": {"process_m4a": words},
        "dvt.praat": {"process_praat": record("praat")},
        "dvt.laughter": {"process_laughter": record("laughter"), "models_missing": none},
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(module, k, v)
        monkeypatch.setitem(sys.modules, name, module)

    def start(mp4: Path, m4a: Path) -> float:
        return 0.25

    monkeypatch.setattr(run.audio_clock, "start", start)
    return seen


def source(tmp_path: Path) -> Source:
    return Source(video=tmp_path / "v.mp4", out=tmp_path / "Film", audio=tmp_path / "v.m4a",
                  frames=tmp_path / "frames", stems=str(tmp_path / STEM / "v.flac"))  # fmt: skip


def test_of_kind() -> None:
    assert run.of_kind(["words", "frames", "metadata", "praat"], run.VIDEO) == [
        "metadata",
        "frames",
    ]
    assert run.of_kind(["laughter", "words", "praat"], run.SPEECH) == ["words", "praat", "laughter"]


def test_load(calls: list[Call]) -> None:
    run.load(["shots", "audioseg", "words", "praat", "metadata"])
    assert calls == [("check", (["metadata", "shots"],))]


def test_load_needs_the_laughter_weights(
    calls: list[Call], monkeypatch: pytest.MonkeyPatch
) -> None:
    def absent() -> list[str]:
        return ["laughter_net.pt"]

    monkeypatch.setattr(sys.modules["dvt.laughter"], "models_missing", absent)
    with pytest.raises(SystemExit, match=r"needs laughter_net\.pt under models/laughter"):
        run.load(["words", "laughter"])


def test_everything(tmp_path: Path, calls: list[Call]) -> None:
    s = source(tmp_path)
    s.audio.write_bytes(b"m4a")
    s.speech.parent.mkdir(parents=True)
    s.speech.write_bytes(b"flac")
    run.annotate(s)
    assert [c[0] for c in calls] == ["video", "audioseg", "words", "praat", "laughter"]
    assert calls[0][1][1] == [n for n in run.ANNOTATORS if run.ANNOTATORS[n].source == "mp4"]
    assert calls[1][1] == (s, 0.25)
    assert calls[2][1] == (s.speech, s.out, 0.25)
    assert calls[4][1] == (s.speech, s.video, s.out, 0.25)


def test_no_sound(tmp_path: Path, calls: list[Call], capsys: pytest.CaptureFixture[str]) -> None:
    s = source(tmp_path)
    run.annotate(s, ["audioseg", "words"])
    out = capsys.readouterr().out
    assert "[Film:audio] no m4a yet — its sound has not been taken out — skipped" in out
    assert "[Film:words] No speech stem — skipped" in out
    (tmp_path / "v.m4a.noaudio").write_bytes(b"")
    run.annotate(s, ["audioseg"])
    assert "[Film:audio] no audio in the source — skipped" in capsys.readouterr().out
    assert calls == []


def test_a_stem_without_its_m4a(tmp_path: Path, calls: list[Call],
                                capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    s = source(tmp_path)
    s.speech.parent.mkdir(parents=True)
    s.speech.write_bytes(b"flac")
    run.annotate(s, ["words", "praat"])
    assert "[Film:words+praat] No m4a beside the stem — skipped" in capsys.readouterr().out
    assert calls == []


def test_extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = source(tmp_path)

    def extract_audio(video: Path, audio: Path) -> str:
        return "made"

    monkeypatch.setattr(run.extraction, "extract_audio", extract_audio)
    assert run.extract(s) == "made"
    s.audio.write_bytes(b"m4a")
    assert run.extract(s) is None
