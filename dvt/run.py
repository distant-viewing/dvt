"""Runs annotators over one source, each after what it reads from."""

import importlib
from collections.abc import Collection, Iterable

from .tools import audio as extraction
from .tools import filehash

# The settings every annotator runs under, before anything else is imported.
# isort: off
from . import runtime as runtime

# isort: on
from . import audio_clock
from .annotators import ANNOTATORS
from .source import Source

VIDEO, SEPARATION, SPEECH = "mp4", "audioseg", "stems"


def of_kind(names: Collection[str], kind: str) -> list[str]:
    return [n for n in ANNOTATORS if n in names and ANNOTATORS[n].source == kind]


def load(names: Collection[str]) -> None:
    """Imports what the named annotators need, before the first file, so a
    package missing from the environment stops the run and not each file
    in turn."""
    if of_kind(names, VIDEO):
        from .video import check_imports

        check_imports(of_kind(names, VIDEO))
    if of_kind(names, SEPARATION):
        importlib.import_module(".audioseg", __package__)
    if "words" in names:
        importlib.import_module(".words", __package__)
    if "praat" in names:
        importlib.import_module(".praat", __package__)
    if "laughter" in names:
        from .laughter import models_missing

        absent = models_missing()
        if absent:
            raise SystemExit(f"the laughter stage needs {', '.join(absent)} under models/laughter")


def extract(source: Source) -> str | None:
    """Takes the sound out of the video, as an m4a or a marker that says
    there is none, unless one of the two is already there."""
    if source.sound is not None:
        return None
    return extraction.extract_audio(source.video, source.audio)


def annotate(source: Source, names: Iterable[str] | None = None) -> None:
    """Runs the named annotators over one source, all of them when none are
    named. The video annotators come first, in the order ANNOTATORS lists
    them, then the separation of the sound, then what reads the speech
    stem. An annotator whose input is missing says so and is skipped."""
    names = list(ANNOTATORS) if names is None else list(names)
    od = source.out

    video = of_kind(names, VIDEO)
    if video:
        from .video import process_mp4

        process_mp4(source, video)

    # Where the sound begins on the video's timeline; every audio table is
    # written on that clock.
    start = audio_clock.start(source.video, source.audio) if source.audio.exists() else None

    if of_kind(names, SEPARATION):
        if not source.audio.exists():
            settled = filehash.noaudio_path_for(source.audio).exists()
            why = (
                "no audio in the source"
                if settled
                else "no m4a yet — its sound has not been taken out"
            )
            print(f"[{od.name}:audio] {why} — skipped")
        else:
            from .audioseg import process_audioseg

            process_audioseg(source, start)

    speech = of_kind(names, SPEECH)
    if speech:
        stem = source.speech
        if not stem.exists() or start is None:
            why = "No speech stem" if not stem.exists() else "No m4a beside the stem"
            print(f"[{od.name}:{'+'.join(speech)}] {why} — skipped")
            return
        if "words" in speech:
            from .words import process_m4a

            process_m4a(stem, od, start=start)
        if "praat" in speech:
            from .praat import process_praat

            process_praat(stem, od, start)
        if "laughter" in speech:
            from .laughter import process_laughter

            process_laughter(stem, source.video, od, start)
