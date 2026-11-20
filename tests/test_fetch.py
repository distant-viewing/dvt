"""dvt models: which models are in the model directory, and the fetch of the
ones that are not. Nothing is downloaded here; each model's own fetch is
replaced by one that writes the files it would."""

import dataclasses
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from dvt import cli, config, fetch, pins
from dvt.annotators import ANNOTATORS


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "_models", None)
    monkeypatch.setenv("TORCH_HOME", "unset")
    monkeypatch.delenv(fetch.OFFLINE, raising=False)
    config.configure(tmp_path / "m")
    return tmp_path / "m"


def put(root: Path, name: str) -> None:
    for p in fetch.MODELS[name].paths:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).touch()


@pytest.fixture
def fetched(models: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Every model's fetch replaced by one that writes its paths and says so."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def standing_in(name: str) -> fetch.Model:
        def fetch_it(languages: Sequence[str]) -> None:
            calls.append((name, tuple(languages)))
            put(models, name)

        return dataclasses.replace(fetch.MODELS[name], fetch=fetch_it)

    monkeypatch.setattr(fetch, "MODELS", {name: standing_in(name) for name in fetch.MODELS})
    return calls


def test_every_model_belongs_to_an_annotator() -> None:
    assert set(fetch.MODELS) <= set(ANNOTATORS)
    assert {m.extra for m in fetch.MODELS.values()} == {"video", "face", "yolo", "audio"}


def test_hub_dir() -> None:
    repo = "facebook/dinov2-large"
    assert fetch.hub_dir(repo) == f"models--facebook--dinov2-large/snapshots/{pins.HUB[repo]}"


def test_missing(models: Path) -> None:
    assert fetch.missing() == list(fetch.MODELS)
    assert fetch.missing(["metadata", "shots", "tracks"]) == []
    put(models, "face")
    assert "face" not in fetch.missing()
    # A model is there when every one of its paths is.
    (models / fetch.MODELS["laughter"].paths[0]).parent.mkdir(parents=True)
    (models / fetch.MODELS["laughter"].paths[0]).touch()
    assert fetch.missing(["face", "laughter", "chroma"]) == ["laughter"]


def test_fetch_takes_only_what_is_missing(
    models: Path, fetched: list[tuple[str, tuple[str, ...]]], capsys: pytest.CaptureFixture[str]
) -> None:
    put(models, "frames")
    assert fetch.fetch(["frames", "yolo", "words", "metadata"], ["en", "fr"]) == ["yolo", "words"]
    assert fetched == [("yolo", ("en", "fr")), ("words", ("en", "fr"))]
    assert "[yolo] note: trained from an ultralytics model" in capsys.readouterr().out
    assert fetch.fetch(["frames", "yolo", "words"]) == []


def test_fetch_names_the_extra_that_is_not_installed(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def absent(languages: Sequence[str]) -> None:
        raise ModuleNotFoundError("No module named 'insightface'", name="insightface")

    held = dataclasses.replace(fetch.MODELS["face"], fetch=absent)
    monkeypatch.setattr(fetch, "MODELS", {"face": held})
    with pytest.raises(SystemExit, match=r"face: insightface is not installed.*dvt\[face\]"):
        fetch.fetch()


def test_command_loads_no_library_of_an_extra() -> None:
    code = (
        "import sys; import dvt.cli; "
        "sys.exit(any(m in sys.modules for m in ('torch', 'cv2', 'transformers')))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_command_says_which_library_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def absent(path: Path, out: Path, names: list[str] | None) -> None:
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(cli, "annotate", absent)
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "film.mp4"), "--out", str(tmp_path), "--models", str(tmp_path)])
    assert "torch is not installed" in capsys.readouterr().err


def test_main(
    models: Path,
    fetched: list[tuple[str, tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    there = ["--models", str(models)]
    assert cli.main(["models", "check", *there, "--annotators", "frames,dino2"]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[:2] == ["frames        missing", "dino2         missing"]
    assert f"2 missing from {models}" in out

    assert cli.main(["models", "fetch", *there, "--annotators", "frames,words"]) == 0
    assert fetched == [("frames", ("en",)), ("words", ("en",))]
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "frames        there",
        "words         there",
    ]
    assert cli.main(["models", "check", *there, "--annotators", "frames,words"]) == 0

    fetched.clear()
    args = ["models", "fetch", *there, "--language", "fr", "--language", "de"]
    assert cli.main([*args, "--annotators", "laughter"]) == 0
    assert fetched == [("laughter", ("fr", "de"))]

    with pytest.raises(SystemExit):
        cli.main(["models", "fetch", *there, "--annotators", "metadata"])
    assert "no model is fetched for: metadata" in capsys.readouterr().err
    monkeypatch.setenv(fetch.OFFLINE, "1")
    with pytest.raises(SystemExit):
        cli.main(["models", "fetch", *there])
    assert "HF_HUB_OFFLINE=1 is set" in capsys.readouterr().err
