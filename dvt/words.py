"""Transcribes the separated speech stem into per-film parquets."""

import sys
import time
from collections.abc import Callable, Collection
from pathlib import Path

# The settings every annotator runs under, before anything that loads a
# model; and the libraries' own noise turned down before whisperx loads.
# isort: off
from . import runtime as runtime

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from polars import col as c

from .noise import quiet_loggers, silence

silence()

import nltk  # noqa: E402
import whisperx  # noqa: E402
from silero_vad import get_speech_timestamps, load_silero_vad  # noqa: E402

# isort: on
from . import audio_clock, config  # noqa: E402
from .align_paths import align_model_dir  # noqa: E402
from .language import from_metadata as resolve_language  # noqa: E402

quiet_loggers()

Audio = NDArray[np.float32]
Stage = Callable[[Path, Audio, float], None]


def _use_models() -> None:
    """Points nltk at the model directory, once."""
    path = str(config.nltk_data())
    if path not in nltk.data.path:
        nltk.data.path.append(path)


DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 8
SAMPLING_RATE = 16000

SPEECH_PAD = 0.03


def log_status(od: Path, step_name: str, status: str) -> None:
    print(f"[{od.name}:{step_name}] {status}")


def log_finished(od: Path, step_name: str, t0: float) -> None:
    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"[{od.name}:{step_name}] Finished ({h:02d}:{m:02d}:{s:02d})")


def clip_to_speech(
    w_start: float, w_end: float, speech_segs: list[dict[str, float]], pad: float = SPEECH_PAD
) -> tuple[float, float]:
    overlaps = [s for s in speech_segs if s["end"] > w_start and s["start"] < w_end]
    if not overlaps:
        return w_start, w_end
    new_start = max(w_start, min(s["start"] for s in overlaps) - pad)
    new_end = min(w_end, max(s["end"] for s in overlaps) + pad)
    return new_start, new_end


def create_words(od: Path, audio: Audio, start: float) -> None:
    words_path = od / "words.parquet"
    if words_path.exists():
        log_status(od, "words", "Already exists")
        return

    t0 = time.monotonic()

    words_schema = {"word": pl.String, "start": pl.Float64, "end": pl.Float64, "score": pl.Float64}

    meta_path = od / "metadata.parquet"
    if not meta_path.exists():
        raise ValueError(
            f"{od.name}: words needs metadata.parquet for the declared "
            f"language. Run the metadata stage first."
        )
    if "language_declared" not in pl.read_parquet_schema(meta_path):
        raise ValueError(
            f"{od.name}: metadata.parquet predates the language columns. "
            f"Delete it and rerun the metadata stage."
        )
    declared = pl.read_parquet(meta_path, columns=["language_declared"])["language_declared"][0]
    language = resolve_language(declared)
    if language.code is None:
        audio_clock.write(pl.DataFrame(schema=words_schema), "words", words_path, start)
        log_status(od, "words", f"Skipped — {language.note}")
        log_finished(od, "words", t0)
        return

    vad_model = load_silero_vad()
    speech_segs = get_speech_timestamps(
        torch.as_tensor(audio),
        vad_model,
        sampling_rate=SAMPLING_RATE,
        return_seconds=True,
    )
    if not speech_segs:
        audio_clock.write(pl.DataFrame(schema=words_schema), "words", words_path, start)
        log_status(od, "words", "No speech detected — empty words.parquet")
        log_finished(od, "words", t0)
        return

    log_status(od, "words", f"Transcribing as {language.note}")
    model = whisperx.load_model(
        str(config.whisper()),
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=language.code,
    )
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language.code)

    align_model, align_metadata = whisperx.load_align_model(
        language_code=language.code,
        device=DEVICE,
        model_dir=align_model_dir(language.code),
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    words = [word for segment in aligned["segments"] for word in segment["words"]]
    words_df = pl.DataFrame(words, schema=words_schema).select(c.word, c.start, c.end, c.score)

    starts: list[float | None] = []
    ends: list[float | None] = []
    for word in words_df.iter_rows(named=True):
        w_start, w_end = word["start"], word["end"]
        if w_start is None or w_end is None:
            starts.append(w_start)
            ends.append(w_end)
            continue
        new_start, new_end = clip_to_speech(w_start, w_end, speech_segs)
        starts.append(new_start)
        ends.append(new_end)

    words_df = words_df.with_columns(
        pl.Series("start", starts),
        pl.Series("end", ends),
    )

    audio_clock.write(words_df, "words", words_path, start)
    log_finished(od, "words", t0)


STAGES: dict[str, Stage] = {
    "words": create_words,
}


def process_m4a(
    ip: Path, od: Path, annotators: Collection[str] | None = None, start: float | None = None
) -> None:
    """`start` is where the sound begins on the mp4's timeline, from
    audio_clock.start; every table is written on that clock."""
    if start is None:
        raise ValueError(f"{od.name}: process_m4a needs the audio start; see audio_clock")
    stages = {name: fn for name, fn in STAGES.items() if annotators is None or name in annotators}
    if not stages:
        return

    od.mkdir(parents=True, exist_ok=True)
    _use_models()
    print(f"Working on {od.name} (audio).")

    audio = whisperx.load_audio(str(ip))
    for stage in stages.values():
        stage(od, audio, start)

    print(f"Finished {od.name} (audio).")


if __name__ == "__main__":
    process_m4a(Path(sys.argv[1]), Path(sys.argv[2]), start=float(sys.argv[3]))
