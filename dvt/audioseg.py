"""Separates the source audio into stems and writes the speech intervals and timeline."""

import hashlib
import logging
import os
import random
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

# The settings every annotator runs under, before anything that loads a model.
# isort: off
from . import runtime as runtime

# isort: on
import scipy.ndimage as sn
import scipy.signal as ss
import soundfile as sf
import torch
from audio_separator.separator import Separator

from . import audio_clock, config

if TYPE_CHECKING:
    from .source import Source

Floats = NDArray[np.float32]

SR = 16_000
FRAME = 400
HOP = 160
SMOOTH = 5
MIX_HOP_MS = 50

STEM_CONFIG: dict[str, dict[str, int]] = {
    "speech": {"enter": 20, "exit": 15, "gap": 200, "min": 100},
    "other_vocal": {"enter": 20, "exit": 15, "gap": 200, "min": 100},
    "bass": {"enter": 20, "exit": 15, "gap": 500, "min": 500},
    "drums": {"enter": 20, "exit": 15, "gap": 500, "min": 500},
    "other": {"enter": 20, "exit": 15, "gap": 500, "min": 500},
}

SHIFT_SEED = 0

TIMELINE_STEMS = ("bass", "drums", "other", "other_vocal", "speech")

OUT_EVENTS = "audioseg_intervals.parquet"
OUT_TIMELINE = "audioseg_timeline.parquet"
EVENT_SCHEMA = {"stem": pl.String, "start": pl.Float64, "end": pl.Float64, "confidence": pl.Float64}

# The separator's models want at least this much. Shorter audio is padded
# with silence to this length, separated, and its stems cut back, so every
# file with audio gets its five stems: the cpuaudio side reads the speech
# stem, and a file without one can never finish there.
MIN_SEPARABLE_SEC = 30.0

SEPARATOR_SR = 44_100

SEPARATE_CHUNK_SEC = 300.0
SEPARATE_OVERLAP_SEC = 10.0

MIN_CHUNK_SEC = 60.0


def _shift_seed(stem: str) -> int:
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") ^ SHIFT_SEED


@contextmanager
def _seeded_random(seed: int) -> Generator[None]:
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _to_separator_input(src: Path, dst: Path) -> Path:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-ac",
            "2",
            "-ar",
            str(SEPARATOR_SR),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not decode {src.name} to stereo "
            f"(exit code {result.returncode}):\n{result.stderr.strip()}"
        )
    return dst


def _pad_short(wav: Path) -> int | None:
    """Pads a wav shorter than MIN_SEPARABLE_SEC with silence to that length,
    in place, and returns the frames it had; None when it was long enough and
    is untouched. _trim_stems cuts the stems back to what was returned."""
    info = sf.info(str(wav))
    want = int(MIN_SEPARABLE_SEC * info.samplerate)
    if info.frames >= want:
        return None
    data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    padded = np.zeros((want, data.shape[1]), dtype=np.float32)
    padded[: len(data)] = data
    sf.write(str(wav), padded, rate, subtype="PCM_16")
    return info.frames


def _trim_stems(stem_paths: dict[str, Path], frames: int) -> None:
    for path in stem_paths.values():
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(str(path), data[:frames], rate, subtype="PCM_16")


def _load_mono_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    del data
    if sr != SR:
        g = gcd(SR, sr)
        mono = ss.resample_poly(mono, SR // g, sr // g).astype(np.float32)
    return mono


def _rms_db(waveform: np.ndarray) -> np.ndarray:
    n_frames = (len(waveform) - FRAME) // HOP + 1
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(n_frames, FRAME),
        strides=(waveform.strides[0] * HOP, waveform.strides[0]),
    )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return (20.0 * np.log10(rms + 1e-10)).astype(np.float32)


STEM_CODEC = "flac"


def _save_stems(stem_paths: dict[str, Path], source: "Source") -> None:
    """Every stem the separator made, all five, kept as flac beside the
    tables made from them. The four besides speech are read by the papers
    and the site, so a run over a file always writes the whole set."""
    for stem, wav in stem_paths.items():
        dst = source.stem(stem)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.stem}.part{dst.suffix}")
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-c:a", STEM_CODEC, str(tmp)],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg could not encode the {stem} stem of {source.name} "
                    f"(exit code {result.returncode}):\n{result.stderr.strip()}"
                )
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dst)


def _hysteresis(db: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    mask = np.zeros(len(db), dtype=bool)
    state = False
    for i, v in enumerate(db):
        if not state and v >= enter:
            state = True
        elif state and v < exit_:
            state = False
        mask[i] = state
    return mask


def _bridge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out, n, i = mask.copy(), len(mask), 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _remove_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out, n, i = mask.copy(), len(mask), 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _audio_to_db(audio_path: Path) -> tuple[np.ndarray, float, float] | None:
    """None for a stem shorter than one frame, which has no level to measure."""
    wav = _load_mono_16k(audio_path)
    if len(wav) < FRAME:
        return None
    db = sn.median_filter(_rms_db(wav), size=SMOOTH)
    floor = float(np.percentile(db, 10))
    dynamic_range = max(float(np.percentile(db, 99)) - floor, 1e-6)
    return db, floor, dynamic_range


def _detect_intervals(
    db: Floats, floor: float, dynamic_range: float, cfg: dict[str, int]
) -> list[tuple[float, float, float]]:
    mask = _hysteresis(db, floor + cfg["enter"], floor + cfg["exit"])
    mask = _bridge_gaps(mask, max(1, round(cfg["gap"] / 10)))
    mask = _remove_short(mask, max(1, round(cfg["min"] / 10)))

    intervals: list[tuple[float, float, float]] = []
    n, i = len(mask), 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            conf = float(np.clip((np.mean(db[start:i]) - floor) / dynamic_range, 0.0, 1.0))
            intervals.append((start * HOP / SR, i * HOP / SR, conf))
        else:
            i += 1
    return intervals


def _stem_timeline(stem_paths: dict[str, Path], hop_ms: int = MIX_HOP_MS) -> pl.DataFrame:
    hop_samples = round(SR * hop_ms / 1000)
    names = list(stem_paths)

    envelopes: list[Floats] = []
    for path in stem_paths.values():
        wav = _load_mono_16k(path)
        frames = len(wav) // hop_samples
        envelopes.append(
            (wav[: frames * hop_samples].reshape(frames, hop_samples) ** 2).mean(axis=1)
        )
        del wav

    n_frames = min(len(e) for e in envelopes)
    energy = np.stack([e[:n_frames] for e in envelopes])
    del envelopes

    total = energy.sum(axis=0)
    fractions = energy / np.clip(total, 1e-10, None)

    data = {
        "time": np.arange(n_frames) * hop_ms / 1000,
        "energy_db": 10.0 * np.log10(total + 1e-10),
    }
    for i, name in enumerate(names):
        data[name] = fractions[i]
        data[f"{name}_db"] = 10.0 * np.log10(energy[i] + 1e-10)

    return pl.DataFrame(data).sort("time")


def _make_separator(work_dir: Path) -> Separator:
    return Separator(
        output_dir=str(work_dir),
        model_file_dir=str(config.separator()),
        log_level=logging.INFO if os.environ.get("TBM_NOISE") else logging.WARNING,
        output_format="WAV",
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        invert_using_spec=False,
        sample_rate=SEPARATOR_SR,
        use_autocast=False,
        chunk_duration=None,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.5,
            "batch_size": 1,
            "enable_denoise": True,
        },
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
        demucs_params={
            "segment_size": "Default",
            "shifts": 2,
            "overlap": 0.25,
            "segments_enabled": True,
        },
    )


def _pick(
    files: list[str], marker: str, work: Path, context: str, *, without: bool = False
) -> Path:
    hits = [f for f in files if (marker.lower() in f.lower()) != without]
    if not hits:
        raise RuntimeError(
            f"the separator returned no output "
            f"{'without' if without else 'matching'} '{marker}' for {context}; "
            f"it returned {len(files)} file(s): {files}. An empty list means the "
            "separation itself failed — Separator.separate() catches every "
            "per-file exception, logs it at ERROR and returns [], so the cause "
            "is in the job log above this traceback."
        )
    return work / hits[0]


def _chunk_spans(total: int, chunk: int, overlap: int, minimum: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        stop = min(start + chunk, total)
        if total - stop < minimum:
            stop = total
        spans.append((start, stop))
        if stop >= total:
            return spans
        start = stop - overlap


class _Separates(Protocol):
    def separate(self, audio_file_path: str) -> list[str]: ...


def _separate_chunked(
    sep: _Separates,
    seed: int,
    src: Path,
    work: Path,
    label: str,
    wanted: dict[str, tuple[str, bool]],
    context: str,
) -> dict[str, Path]:
    info = sf.info(str(src))
    rate = info.samplerate
    total = info.frames

    if rate != SEPARATOR_SR:
        raise RuntimeError(
            f"{src.name} is at {rate} Hz, not the {SEPARATOR_SR} Hz the "
            f"separator is configured for; the silence fill in _separate_chunked "
            f"assumes a chunk comes back at the rate it went in"
        )

    overlap = int(SEPARATE_OVERLAP_SEC * rate)
    spans = _chunk_spans(total, int(SEPARATE_CHUNK_SEC * rate), overlap, int(MIN_CHUNK_SEC * rate))

    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[:, None]
    fade_out = 1.0 - fade_in

    out_paths = {name: work / f"{label}_{name}.wav" for name in wanted}
    writers = {
        name: sf.SoundFile(str(path), "w", samplerate=rate, channels=2, subtype="PCM_16")
        for name, path in out_paths.items()
    }
    tails: dict[str, Floats | None] = dict.fromkeys(wanted)
    chunk_dir = work / f"{label}_chunks"
    chunk_dir.mkdir(exist_ok=True)

    try:
        for i, (start, stop) in enumerate(spans):
            last = stop >= total
            data, _ = sf.read(
                str(src), start=start, frames=stop - start, dtype="float32", always_2d=True
            )

            audible = bool(data.any())
            produced: list[str] = []
            if audible:
                piece = chunk_dir / f"chunk_{i:04d}.wav"
                sf.write(str(piece), data, rate, subtype="PCM_16")

                with _seeded_random(seed):
                    produced = sep.separate(str(piece))
                piece.unlink(missing_ok=True)
            del data

            silence = None

            for name, (marker, without) in wanted.items():
                arr = None
                if audible:
                    found = _pick(produced, marker, work, context, without=without)

                    if found.exists():
                        arr, _ = sf.read(str(found), dtype="float32", always_2d=True)
                        found.unlink()

                if arr is None:
                    if silence is None:
                        silence = np.zeros((stop - start, 2), dtype=np.float32)
                    arr = silence

                tail = tails[name]
                if tail is None:
                    body = arr if last else arr[: len(arr) - overlap]
                else:
                    writers[name].write(tail * fade_out + arr[:overlap] * fade_in)
                    body = arr[overlap:] if last else arr[overlap : len(arr) - overlap]
                writers[name].write(body)
                tails[name] = None if last else arr[len(arr) - overlap :].copy()
    finally:
        for writer in writers.values():
            writer.close()

    for name, path in out_paths.items():
        got = sf.info(str(path)).frames
        if got != total:
            raise RuntimeError(
                f"stitched {label}/{name} for {context} is {got} frames against "
                f"the {total} that went in, over {len(spans)} chunk(s)"
            )
    return out_paths


def process_audioseg(source: "Source", start: float | None = None) -> None:
    """Separates the source's m4a into its stems and writes the tables
    made from them. `start` is where the sound begins on the mp4's timeline, from
    audio_clock.start; the intervals and timeline are written on that clock."""
    if start is None:
        raise ValueError(f"{source.name}: process_audioseg needs the audio start; see audio_clock")
    input_audio = source.audio
    od = source.out
    od.mkdir(parents=True, exist_ok=True)

    tables = (od / OUT_EVENTS).exists() and (od / OUT_TIMELINE).exists()

    # The separation is the costly part and the stems are its product. A file
    # with a speech stem and both tables is left alone, though its other
    # four stems may be missing. A file with all five stems and no tables has
    # the tables made from the stems. Any other file is separated, every
    # stem written over what was there, and its tables made.
    stems = {name: source.stem(name) for name in TIMELINE_STEMS}
    if tables and stems["speech"].exists():
        print(f"[{od.name}:audioseg] Already exists")
        return
    if all(p.exists() for p in stems.values()):
        print(f"[{od.name}:audioseg] Tables from the five stems already kept")
        _write_tables(stems, od, start)
        return

    config.tmp().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(config.tmp()), prefix=f"{input_audio.stem}_"
    ) as work_str:
        work = Path(work_str)
        sep = _make_separator(work)
        device = sep.torch_device
        if device is None or device.type != "cuda":
            raise RuntimeError(
                f"audio_separator fell back to {device.type if device else 'no device'} for "
                f"{input_audio.name} — torch.cuda.is_available() is "
                f"{torch.cuda.is_available()} in this process. It logs that "
                "fallback at INFO, which _make_separator's WARNING log level "
                "hides, so a CPU run otherwise finishes ~50x slower with no "
                "trace in the job log."
            )
        seed = _shift_seed(input_audio.stem)

        mix_wav = _to_separator_input(input_audio, work / f"{input_audio.stem}_stereo.wav")
        kept = _pad_short(mix_wav)

        sep.load_model("htdemucs_ft.yaml")
        demucs = _separate_chunked(
            sep,
            seed,
            mix_wav,
            work,
            "demucs",
            {
                "bass": ("(Bass)", False),
                "drums": ("(Drums)", False),
                "other": ("(Other)", False),
                "vocals": ("(Vocals)", False),
            },
            context=input_audio.name,
        )
        stem_paths = {name: demucs[name] for name in ("bass", "drums", "other")}

        sep.load_model("vocals_mel_band_roformer.ckpt")
        mel = _separate_chunked(
            sep,
            seed,
            demucs["vocals"],
            work,
            "mel",
            {
                "other_vocal": ("(other)", False),
                "speech": ("(other)", True),
            },
            context=input_audio.name,
        )
        stem_paths["other_vocal"] = mel["other_vocal"]
        stem_paths["speech"] = mel["speech"]

        assert tuple(stem_paths) == TIMELINE_STEMS, (
            f"stem order {tuple(stem_paths)} != TIMELINE_STEMS {TIMELINE_STEMS}"
        )

        if kept is not None:
            _trim_stems(stem_paths, kept)
            print(
                f"[{od.name}:audioseg] Audio is {kept / SEPARATOR_SR:.1f}s; "
                f"separated padded to {MIN_SEPARABLE_SEC:.0f}s and cut back"
            )

        _save_stems(stem_paths, source)
        print(f"[{od.name}:audioseg] Kept {len(stem_paths)} stems beside {source.speech}")

        _write_tables(stem_paths, od, start)


def _write_tables(stem_paths: dict[str, Path], od: Path, start: float) -> None:
    """The intervals and timeline of one file from its five stems, written on
    the mp4's clock: `start` is the audio start, never an interval's."""
    event_rows: list[dict[str, str | float]] = []
    for name, path in stem_paths.items():
        measured = _audio_to_db(path)
        if measured is None:
            continue
        db, floor, dynamic_range = measured
        for s, e, conf in _detect_intervals(db, floor, dynamic_range, STEM_CONFIG[name]):
            event_rows.append({"stem": name, "start": s, "end": e, "confidence": conf})

    df_events = (
        pl.DataFrame(event_rows, schema=EVENT_SCHEMA)
        if event_rows
        else pl.DataFrame(schema=EVENT_SCHEMA)
    ).sort(["stem", "start"])

    df_timeline = _stem_timeline(stem_paths)

    audio_clock.write(df_events, "audioseg", od / OUT_EVENTS, start)
    audio_clock.write(df_timeline, "audioseg", od / OUT_TIMELINE, start)
    print(
        f"[{od.name}:audioseg] Saved {len(df_events)} intervals, {len(df_timeline)} timeline rows"
    )
