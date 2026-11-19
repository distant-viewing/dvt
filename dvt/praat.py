"""Phonetic measurement of the separated speech stem, frame by frame."""

import sys
import time
from pathlib import Path

import numpy as np
import parselmouth
import polars as pl
import soundfile as sf
from numpy.typing import NDArray
from parselmouth.praat import call as praat_call

from . import audio_clock

Floats = NDArray[np.float64]


TIME_STEP = 0.01

PITCH_FLOOR = 75.0
PITCH_CEILING = 400.0

FORMANT_CEILING = 5500.0
FORMANT_COUNT = 5
FORMANTS_KEPT = 4

SPECTRUM_WINDOW = 0.025
SPECTRUM_MAX_HZ = 8000.0

MIN_VOICED_SEC = 0.06

JITTER_ARGS = (0.0001, 0.02, 1.3)
SHIMMER_ARGS = (0.0001, 0.02, 1.3, 1.6)

BLOCK_SEC = 600.0
MARGIN_SEC = 5.0

MIN_BLOCK_SEC = 2.0

SCHEMA = {
    "time": pl.Float64,
    "f0_hz": pl.Float32,
    "intensity_db": pl.Float32,
    "hnr_db": pl.Float32,
    **{f"f{n}_hz": pl.Float32 for n in range(1, FORMANTS_KEPT + 1)},
    **{f"b{n}_hz": pl.Float32 for n in range(1, FORMANTS_KEPT + 1)},
    "cog_hz": pl.Float32,
    "spectral_sd_hz": pl.Float32,
    "spectral_skewness": pl.Float32,
    "spectral_kurtosis": pl.Float32,
    "spectral_tilt_db_per_khz": pl.Float32,
    "jitter_local": pl.Float32,
    "shimmer_local": pl.Float32,
}


def log_status(od: Path, status: str) -> None:
    print(f"[{od.name}:praat] {status}")


def _sample_on(grid: Floats, obj: parselmouth.TimeFrameSampled, values: Floats) -> Floats:
    idx = np.rint((grid - obj.t1) / obj.dt).astype(np.int64)
    out = np.full(len(grid), np.nan)
    ok = (idx >= 0) & (idx < len(values))
    out[ok] = values[idx[ok]]
    return out


def _formant_tracks(formant: parselmouth.Formant, grid: Floats) -> tuple[Floats, Floats]:
    n = FORMANTS_KEPT
    freqs = np.empty((len(grid), n))
    bws = np.empty((len(grid), n))
    for j, t in enumerate(grid):
        for k in range(n):
            freqs[j, k] = formant.get_value_at_time(k + 1, t)
            bws[j, k] = formant.get_bandwidth_at_time(k + 1, t)
    return freqs, bws


def _spectral_moments(sound: parselmouth.Sound, grid: Floats) -> dict[str, Floats]:
    sg = sound.to_spectrogram(
        window_length=SPECTRUM_WINDOW,
        time_step=TIME_STEP,
        maximum_frequency=SPECTRUM_MAX_HZ,
    )
    power = sg.values
    freqs = np.arange(len(power)) * sg.dy + sg.y1

    total = power.sum(axis=0)
    quiet = total <= 0.0
    safe = np.where(quiet, 1.0, total)

    cog = (freqs[:, None] * power).sum(axis=0) / safe
    dev = freqs[:, None] - cog
    m2 = (dev**2 * power).sum(axis=0) / safe
    sd = np.sqrt(m2)
    sd_safe = np.where(sd <= 0.0, 1.0, sd)
    m3 = (dev**3 * power).sum(axis=0) / safe
    m4 = (dev**4 * power).sum(axis=0) / safe

    khz = freqs / 1000.0
    db = 10.0 * np.log10(np.maximum(power, 1e-20))
    khz_c = khz - khz.mean()
    tilt = (khz_c[:, None] * (db - db.mean(axis=0))).sum(axis=0) / (khz_c**2).sum()

    out = {
        "cog_hz": cog,
        "spectral_sd_hz": sd,
        "spectral_skewness": m3 / sd_safe**3,
        "spectral_kurtosis": m4 / sd_safe**4 - 3.0,
        "spectral_tilt_db_per_khz": tilt,
    }
    for key in out:
        out[key] = np.where(quiet, np.nan, out[key])
    return {k: _sample_on(grid, sg, v) for k, v in out.items()}


def _voiced_runs(f0: Floats, grid: Floats) -> list[tuple[int, int]]:
    voiced = np.isfinite(f0)
    if not voiced.any():
        return []
    edges = np.diff(voiced.astype(np.int8))
    starts = [int(i) for i in np.flatnonzero(edges == 1) + 1]
    stops = [int(i) for i in np.flatnonzero(edges == -1) + 1]
    if voiced[0]:
        starts.insert(0, 0)
    if voiced[-1]:
        stops.append(len(voiced))
    least = max(2, int(round(MIN_VOICED_SEC / TIME_STEP)))
    return [(a, b) for a, b in zip(starts, stops, strict=True) if b - a >= least]


def _number(value: object) -> float:
    """What a Praat query gave, which is a number for the queries made here."""
    if not isinstance(value, float):
        raise TypeError(f"Praat gave {value!r} where a number was expected")
    return value


def _voice_quality(sound: parselmouth.Sound, f0: Floats, grid: Floats) -> dict[str, Floats]:
    jitter = np.full(len(grid), np.nan)
    shimmer = np.full(len(grid), np.nan)
    runs = _voiced_runs(f0, grid)
    if not runs:
        return {"jitter_local": jitter, "shimmer_local": shimmer}

    points = praat_call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    if not isinstance(points, parselmouth.Data):
        raise TypeError(f"Praat gave {points!r} where a PointProcess was expected")
    for a, b in runs:
        t0, t1 = float(grid[a]), float(grid[b - 1]) + TIME_STEP
        jitter[a:b] = _number(praat_call(points, "Get jitter (local)", t0, t1, *JITTER_ARGS))
        shimmer[a:b] = _number(
            praat_call([sound, points], "Get shimmer (local)", t0, t1, *SHIMMER_ARGS)
        )
    return {"jitter_local": jitter, "shimmer_local": shimmer}


def _block_frames(sound: parselmouth.Sound, grid: Floats) -> dict[str, Floats]:
    pitch = sound.to_pitch_ac(
        time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING
    )
    f0_frames = pitch.selected_array["frequency"].copy()
    f0_frames[f0_frames == 0.0] = np.nan
    f0 = _sample_on(grid, pitch, f0_frames)

    intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=TIME_STEP)
    harmonicity = sound.to_harmonicity_cc(time_step=TIME_STEP, minimum_pitch=PITCH_FLOOR)
    formant = sound.to_formant_burg(
        time_step=TIME_STEP, max_number_of_formants=FORMANT_COUNT, maximum_formant=FORMANT_CEILING
    )
    freqs, bws = _formant_tracks(formant, grid)

    hnr = _sample_on(grid, harmonicity, harmonicity.values[0])
    hnr[hnr <= -199.0] = np.nan

    out = {
        "f0_hz": f0,
        "intensity_db": _sample_on(grid, intensity, intensity.values[0]),
        "hnr_db": hnr,
    }
    for k in range(FORMANTS_KEPT):
        out[f"f{k + 1}_hz"] = freqs[:, k]
        out[f"b{k + 1}_hz"] = bws[:, k]
    out.update(_spectral_moments(sound, grid))
    out.update(_voice_quality(sound, f0, grid))
    return out


def _read_block(path: Path, rate: int, lo: float, hi: float) -> parselmouth.Sound:
    start = int(round(lo * rate))
    frames = int(round(hi * rate)) - start
    data, sr = sf.read(str(path), start=start, frames=frames, dtype="float64", always_2d=True)
    return parselmouth.Sound(data.mean(axis=1), sampling_frequency=sr)


def _blocks(duration: float) -> list[tuple[float, float]]:
    """Blocks of BLOCK_SEC from the start. A tail shorter than MIN_BLOCK_SEC
    is too short to measure alone, so the block before it takes it in."""
    edges = [float(a) for a in np.arange(0.0, duration, BLOCK_SEC)]
    if len(edges) > 1 and duration - edges[-1] < MIN_BLOCK_SEC:
        edges.pop()
    ends = [*edges[1:], duration] if edges else []
    return list(zip(edges, ends, strict=True))


def process_praat(stem: Path, od: Path, start: float | None = None) -> None:
    """`start` is where the sound begins on the mp4's timeline, from
    audio_clock.start; the frames are written on that clock."""
    if start is None:
        raise ValueError(f"{od.name}: process_praat needs the audio start; see audio_clock")
    od.mkdir(parents=True, exist_ok=True)
    out_path = od / "praat.parquet"
    if out_path.exists():
        log_status(od, "Already exists")
        return

    t0 = time.monotonic()
    info = sf.info(str(stem))
    duration = info.frames / info.samplerate

    if duration < MIN_BLOCK_SEC:
        audio_clock.write(pl.DataFrame(schema=SCHEMA), "praat", out_path, start)
        log_status(od, f"Stem is {duration:.1f}s, too short to analyse — wrote empty praat.parquet")
        return

    log_status(od, f"Measuring {duration / 60:.1f} min of speech stem")
    columns: dict[str, list[Floats]] = {name: [] for name in SCHEMA}
    for a, b in _blocks(duration):
        lo = max(0.0, a - MARGIN_SEC)
        hi = min(duration, b + MARGIN_SEC)
        block = _read_block(stem, info.samplerate, lo, hi)

        grid = np.arange(a - lo, b - lo, TIME_STEP)
        frames = _block_frames(block, grid)
        columns["time"].append(grid + lo)
        for name, values in frames.items():
            columns[name].append(values)

    df = pl.DataFrame(
        {name: np.concatenate(parts) for name, parts in columns.items()},
        schema=SCHEMA,
    ).with_columns(pl.col(pl.Float32).fill_nan(None))
    audio_clock.write(df, "praat", out_path, start)

    voiced = df["f0_hz"].is_not_null().sum()
    elapsed = time.monotonic() - t0
    log_status(
        od,
        f"Saved {df.height} frames, {voiced} voiced "
        f"({100 * voiced / max(df.height, 1):.1f}%), "
        f"{duration / max(elapsed, 1e-9):.0f}x realtime",
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <speech_stem.flac> <output_dir> <audio_start_sec>")
        sys.exit(1)
    process_praat(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))
