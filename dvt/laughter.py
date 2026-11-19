"""Laughter heard between lines, from the speech stem: the cascade of
2026_laugh in projects-2026 (24_cascade.py), which labels every moment music, speech,
laughter or other, each taking what the ones before it left.

  music     bass and drums take more than MUSIC of the audioseg timeline
  speech    the words, a word stretched past LONG_WORD left out, padded and
            with short gaps closed; where praat finds voicing at a good HNR;
            and every run of voicing that touches a word
  laughter  a candidate the classifier keeps: a stretch outside speech and
            music where the stem stands MIN_RISE above a local floor, kept
            from MIN_LEN, scored on its measures and on the network's mean
            and top probability of laughter over it
  other     the rest, silence among it

Laughter under speech is speech, and laughter over music is music. The
pieces of one laugh are joined across a gap under MERGE seconds with no
speech or music in it.

The stage reads words.parquet, praat.parquet and audioseg_timeline.parquet,
the speech stem, the mix (the video's first audio stream, as the classifier
was fitted on; the m4a is not always the same samples), and the weights under
models()/laughter: laughter_net.pt and laughter_classifier.json, written by
2026_laugh/28_export.py. It writes laughter.parquet, one row per laugh
(start, end, pieces, p, the classifier's top score over its pieces), and
laughter_labels.parquet, the runs of each label. Both are on the mp4's clock.

The detector was trained on American sitcoms with an audience or a laugh
track. On other material it will call crowd noise, applause and effects
between lines laughter.
"""

import functools
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from scipy import ndimage, signal

from . import audio_clock, config
from .laughter_net import LaughterNet
from .tools import probejson

Floats = NDArray[np.float64]
Floats32 = NDArray[np.float32]
Mask = NDArray[np.bool_]
Labelled = NDArray[np.int32 | np.intp]

SR = 16000
STEP = 0.01  # seconds, praat's step: the masks and levels
SMOOTH = 0.25  # seconds, the stem level

PAD = 0.1  # seconds either side of a word
GAP = 0.5  # seconds, gaps between words closed
LONG_WORD = 1.0  # seconds, a longer word is not taken as speech
VOICED_WIN = 0.5  # seconds
VOICED_SHARE = 0.5  # of VOICED_WIN voiced makes speech
HNR_MIN = 6.0  # dB, mean HNR of the voiced frames there
VOICE_GAP = 0.05  # seconds, unvoiced within a run of voicing

MUSIC = 0.3  # share of bass and drums in the timeline
HEAD = 0.15  # seconds, a candidate's start against what follows
FLOOR_PCT = 20  # percentile of the level outside speech
FLOOR_WIN = 60.0  # seconds, centred
FLOOR_MIN = 5.0  # seconds outside speech in the window, at least
MIN_RISE = 6.0  # dB above the floor
CLOSE = 0.3  # seconds
MIN_LEN = 0.5  # seconds
SILENT_DB = -100.0  # a frame of the stem below this is no sound

N_FFT = 1024
WIN = 736  # samples, 46 ms
HOP = 160  # samples, 10 ms
N_MELS = 128
NET_STEP = 0.02  # seconds, the network's frame
MEL_CHUNK = 65536  # spectrogram frames made at a time
LEVEL_EVERY = 50  # spectrogram frames, for the episode's level
NET_SILENT = -90.0  # dB, frames below are not used for the level

MERGE = 1.0  # seconds
LABELS = ("other", "music", "speech", "laughter")

NET_FILE = "laughter_net.pt"
CLASSIFIER_FILE = "laughter_classifier.json"
INPUTS = ("words.parquet", "praat.parquet", "audioseg_timeline.parquet")

LAUGHS_SCHEMA = {"start": pl.Float64, "end": pl.Float64, "pieces": pl.Int64, "p": pl.Float64}
LABELS_SCHEMA = {"start": pl.Float64, "end": pl.Float64, "label": pl.String}


def log_status(od: Path, msg: str) -> None:
    print(f"[{od.name}:laughter] {msg}", flush=True)


# ---- the classifier ------------------------------------------------------------


@dataclass(frozen=True)
class Tree:
    feature: NDArray[np.int64]
    threshold: Floats
    missing_left: Mask
    left: NDArray[np.int64]
    right: NDArray[np.int64]
    leaf: Mask
    value: Floats


@dataclass(frozen=True)
class Classifier:
    features: tuple[str, ...]
    cut: float
    baseline: float
    trees: tuple[Tree, ...]

    def score(self, x: Floats) -> Floats:
        """p for each row of x, whose columns are `features`; a missing value
        is NaN."""
        raw = np.full(len(x), self.baseline)
        for t in self.trees:
            for r in range(len(x)):
                k = 0
                while not t.leaf[k]:
                    v = x[r, t.feature[k]]
                    go_left = bool(t.missing_left[k]) if np.isnan(v) else bool(v <= t.threshold[k])
                    k = int(t.left[k] if go_left else t.right[k])
                raw[r] += t.value[k]
        p: Floats = 1 / (1 + np.exp(-raw))
        return p


def _numbers(obj: object, key: str) -> Floats:
    values = [probejson.as_float(v) for v in probejson.items(probejson.get(obj, key))]
    if not values or any(v is None for v in values):
        raise ValueError(f"{CLASSIFIER_FILE}: {key} is not a list of numbers")
    return np.array(values, dtype=np.float64)


def _flags(obj: object, key: str) -> Mask:
    values = probejson.items(probejson.get(obj, key))
    flags = [v for v in values if isinstance(v, bool)]
    if not values or len(flags) != len(values):
        raise ValueError(f"{CLASSIFIER_FILE}: {key} is not a list of true and false")
    return np.array(flags, dtype=bool)


def _number(obj: object, key: str) -> float:
    value = probejson.as_float(probejson.get(obj, key))
    if value is None:
        raise ValueError(f"{CLASSIFIER_FILE}: {key} is not a number")
    return value


def parse_classifier(text: str) -> Classifier:
    """The classifier as 28_export.py writes it."""
    obj = probejson.load(text)
    names = [probejson.as_str(v) for v in probejson.items(probejson.get(obj, "features"))]
    features = tuple(n for n in names if n is not None)
    if not features or len(features) != len(names):
        raise ValueError(f"{CLASSIFIER_FILE}: features is not a list of names")
    trees: list[Tree] = []
    for t in probejson.items(probejson.get(obj, "trees")):
        trees.append(
            Tree(
                feature=_numbers(t, "feature").astype(np.int64),
                threshold=_numbers(t, "threshold"),
                missing_left=_flags(t, "missing_left"),
                left=_numbers(t, "left").astype(np.int64),
                right=_numbers(t, "right").astype(np.int64),
                leaf=_flags(t, "leaf"),
                value=_numbers(t, "value"),
            )
        )
    if not trees:
        raise ValueError(f"{CLASSIFIER_FILE}: no trees")
    return Classifier(features, _number(obj, "cut"), _number(obj, "baseline"), tuple(trees))


@functools.cache
def classifier() -> Classifier:
    return parse_classifier((config.models() / "laughter" / CLASSIFIER_FILE).read_text())


@functools.cache
def network() -> LaughterNet:
    net = LaughterNet()
    net.load_state_dict(torch.load(config.models() / "laughter" / NET_FILE))
    net.eval()
    return net


# ---- levels and masks ----------------------------------------------------------


def decode(path: Path) -> Floats32:
    """The first audio stream, mono at SR, as ffmpeg gives it."""
    out = ["-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *out], capture_output=True, check=True
    ).stdout
    return np.frombuffer(raw, np.float32)


def level(y: Floats32) -> Floats32:
    """The level in dB, one value per STEP, smoothed over SMOOTH."""
    hop = int(STEP * SR)
    n = len(y) // hop
    db = 10 * np.log10((y[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return np.asarray(ndimage.uniform_filter1d(db, int(SMOOTH / STEP)), dtype=np.float32)


def frames(spans: Iterable[tuple[float, float]], n: int) -> Mask:
    on = np.zeros(n, bool)
    for a, b in spans:
        on[max(0, int(a / STEP)) : max(0, int(b / STEP))] = True
    return on


def close(on: Mask, sec: float) -> Mask:
    closed: Mask = ndimage.binary_closing(on, np.ones(int(sec / STEP) | 1, dtype=bool)) | on
    return closed


def praat_index(praat: pl.DataFrame, n: int) -> NDArray[np.int64]:
    return np.clip((praat["time"].to_numpy() / STEP).round().astype(np.int64), 0, n - 1)


def by_praat(praat: pl.DataFrame, n: int) -> Mask:
    """Speech where half of VOICED_WIN is voiced at a mean HNR of HNR_MIN."""
    i = praat_index(praat, n)
    voiced = np.zeros(n)
    hnr = np.full(n, np.nan)
    voiced[i] = praat["f0_hz"].is_not_null().to_numpy()
    hnr[i] = praat["hnr_db"].to_numpy()
    k = int(VOICED_WIN / STEP)
    share: Floats = ndimage.uniform_filter1d(voiced, k)
    weighted: Floats = ndimage.uniform_filter1d(voiced * np.nan_to_num(hnr), k)
    mean_hnr = weighted / np.maximum(share, 1e-9)
    return close((share >= VOICED_SHARE) & (mean_hnr >= HNR_MIN), GAP)


def speech_mask(words: pl.DataFrame, praat: pl.DataFrame, n: int) -> Mask:
    """The words, a word stretched past LONG_WORD left out, padded and with
    gaps under GAP closed; praat's speech; and every run of voicing that
    touches a word, unpadded, in whole."""
    spans = [
        (float(a), float(b))
        for a, b in words.select("start", "end").drop_nulls().rows()
        if float(b) - float(a) <= LONG_WORD
    ]
    on_word = frames(spans, n)
    padded = frames(((a - PAD, b + PAD) for a, b in spans), n)
    voiced = np.zeros(n, bool)
    voiced[praat_index(praat, n)] = praat["f0_hz"].is_not_null().to_numpy()
    runs_of: Labelled
    runs_of, _ = ndimage.label(close(voiced, VOICE_GAP))
    touching = np.unique(runs_of[on_word & (runs_of > 0)])
    return close(padded, GAP) | by_praat(praat, n) | np.isin(runs_of, touching[touching > 0])


def shares(timeline: pl.DataFrame, n: int) -> dict[str, Floats]:
    """Each stem's share of the audio from the audioseg timeline, per STEP."""
    tl = timeline.sort("time")
    t = tl["time"].to_numpy()
    i = np.clip(np.searchsorted(t, np.arange(n) * STEP), 0, len(t) - 1)
    return {
        "music": (tl["bass"] + tl["drums"]).to_numpy()[i],
        "other": tl["other"].to_numpy()[i],
        "speech": tl["speech"].to_numpy()[i],
    }


def local_floor(lv: Floats32, quiet: Mask) -> Floats:
    """FLOOR_PCT of the level where quiet, within FLOOR_WIN, every second."""
    n, half, every = len(lv), int(FLOOR_WIN / 2 / STEP), int(1 / STEP)
    whole = float(np.percentile(lv[quiet], FLOOR_PCT))
    centres = np.arange(0, n, every)
    vals: list[float] = []
    for c in centres:
        s = slice(max(0, int(c) - half), int(c) + half)
        v = lv[s][quiet[s]]
        vals.append(float(np.percentile(v, FLOOR_PCT)) if len(v) * STEP >= FLOOR_MIN else whole)
    floor: Floats = np.interp(np.arange(n), centres, vals)
    return floor


# ---- the candidates ------------------------------------------------------------

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "start",
    "end",
    "peak_db",
    "over_speech_db",
    "residual_db",
    "music",
    "other",
    "speech",
    "peak_at",
    "swell_db",
    "after_speech",
)
CANDIDATE_SCHEMA = pl.Schema({k: pl.Float64() for k in CANDIDATE_COLUMNS})


def candidates(
    stem_level: Floats32,
    mix_level: Floats32,
    words: pl.DataFrame,
    praat: pl.DataFrame,
    timeline: pl.DataFrame,
) -> pl.DataFrame:
    """The stretches outside speech and music where the stem stands MIN_RISE
    above its local floor, with the measures the classifier reads. They are
    rounded as 2026_laugh wrote them, since the classifier was fitted on
    the rounded values."""
    n = min(len(stem_level), len(mix_level))
    lv, mix = stem_level[:n], mix_level[:n]
    talk = speech_mask(words, praat, n)
    share = shares(timeline, n)
    music = share["music"] > MUSIC
    sounding = lv > SILENT_DB
    quiet = ~talk & ~music & sounding
    if not quiet.any() or not (talk & sounding).any():
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)
    floor = local_floor(lv, quiet)
    speech_db = float(np.median(lv[talk & sounding]))
    residual = mix - lv
    speech_residual = float(np.median(residual[talk & sounding]))
    on = close(quiet & (lv > floor + MIN_RISE), CLOSE) & ~talk & ~music
    labelled: Labelled
    labelled, _ = ndimage.label(on)
    head = int(HEAD / STEP)
    rows: list[dict[str, float | None]] = []
    for found in ndimage.find_objects(labelled):
        s = found[0]
        a, b = s.start * STEP, s.stop * STEP
        if b - a < MIN_LEN:
            continue
        seg = lv[s]
        before = np.flatnonzero(talk[: s.start])
        rows.append(
            {
                "start": round(a, 2),
                "end": round(b, 2),
                "peak_db": round(float(seg.max()), 1),
                "over_speech_db": round(float(seg.max() - speech_db), 1),
                "residual_db": round(float(residual[s].mean() - speech_residual), 1),
                "music": round(float(share["music"][s].mean()), 3),
                "other": round(float(share["other"][s].mean()), 3),
                "speech": round(float(share["speech"][s].mean()), 3),
                "peak_at": round(float(np.argmax(seg) * STEP), 2),
                "swell_db": round(float(seg[head:].max() - seg[:head].max()), 1),
                "after_speech": round(a - float(before[-1] * STEP), 2) if len(before) else None,
            }
        )
    return pl.DataFrame(rows, schema=CANDIDATE_SCHEMA)


# ---- the network ---------------------------------------------------------------


def mel_filters(n_fft: int = N_FFT, n: int = N_MELS) -> Floats:
    def to_mel(hz: Floats) -> Floats:
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(m: Floats) -> Floats:
        return 700 * (10 ** (m / 2595) - 1)

    edges = to_hz(np.linspace(to_mel(np.array(60.0)), to_mel(np.array(SR / 2)), n + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n, len(bins)))
    for i in range(n):
        lo, c, hi = edges[i : i + 3]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


def mel(y: Floats32, chunk: int = MEL_CHUNK) -> NDArray[np.float16]:
    """The log-mel spectrogram, one row per HOP, the window centred on the
    middle of its HOP, in dB as float16, as the network was trained on. Its
    frames are independent, so it is made `chunk` frames at a time, which
    gives the same numbers in memory that does not grow with the file."""
    edge = (WIN - HOP) // 2
    padded = np.pad(y, (edge, edge + HOP))
    n = 1 + (len(padded) - WIN) // HOP if len(padded) >= WIN else 0
    fb = mel_filters()
    parts: list[NDArray[np.float16]] = []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        z: NDArray[np.complex64]
        _, _, z = signal.stft(
            padded[a * HOP : (b - 1) * HOP + WIN], SR, window="hann", nperseg=WIN,
            noverlap=WIN - HOP, nfft=N_FFT, boundary=None, padded=False,
        )  # fmt: skip
        parts.append((10 * np.log10(fb @ (np.abs(z) ** 2) + 1e-10)).T.astype(np.float16))
    return np.concatenate(parts) if parts else np.zeros((0, N_MELS), np.float16)


def laughter_p(spec: NDArray[np.float16], net: LaughterNet) -> Floats32:
    """The network's probability of laughter every NET_STEP, the spectrogram
    first set to the stem's median level."""
    spec = spec[: len(spec) // 2 * 2]
    every = spec[::LEVEL_EVERY].astype(np.float32).mean(1)
    loud = every[every > NET_SILENT]
    if not len(loud):
        return np.zeros(len(spec) // 2, np.float32)
    at = float(np.median(loud))
    return net.laughter(spec, at)


def with_network(c: pl.DataFrame, p: Floats32) -> pl.DataFrame:
    """The candidates with their length and the network's mean and top p."""
    mean: list[float] = []
    top: list[float] = []
    for a, b in c.select("start", "end").rows():
        i = int(a / NET_STEP)
        j = max(int(b / NET_STEP), i + 1)
        mean.append(float(p[i:j].mean()) if i < len(p) else np.nan)
        top.append(float(p[i:j].max()) if i < len(p) else np.nan)
    return c.with_columns(
        len=pl.col("end") - pl.col("start"),
        net_mean=pl.Series(mean, dtype=pl.Float64),
        net_max=pl.Series(top, dtype=pl.Float64),
    )


def kept(c: pl.DataFrame, model: Classifier) -> pl.DataFrame:
    """The candidates the classifier scores at its cut or above, with p."""
    if not c.height:
        return c.with_columns(p=pl.lit(None, pl.Float64))
    x = c.select(model.features).to_numpy().astype(np.float64)
    return c.with_columns(p=pl.Series(model.score(x))).filter(pl.col("p") >= model.cut)


# ---- the cascade ---------------------------------------------------------------


def assemble(
    laughs: pl.DataFrame, talk: Mask, music: Mask, merge: float = MERGE
) -> tuple[NDArray[np.int8], Floats]:
    """Each moment's label as an index into LABELS, and the classifier's
    score where laughter was kept."""
    n = len(talk)
    laugh = np.zeros(n, bool)
    p = np.zeros(n)
    for a, b, q in laughs.select("start", "end", "p").rows():
        i, j = int(round(float(a) / STEP)), int(round(float(b) / STEP))
        laugh[i:j] = True
        p[i:j] = np.maximum(p[i:j], float(q))
    blocked = talk | music
    gaps: Labelled
    gaps, _ = ndimage.label(~laugh)
    for found in ndimage.find_objects(gaps):
        s = found[0]
        if (
            s.start > 0
            and s.stop < n
            and (s.stop - s.start) * STEP < merge
            and not blocked[s].any()
        ):
            laugh[s] = True
    code = np.zeros(n, np.int8)
    code[laugh & ~talk & ~music] = 3
    code[talk & ~music] = 2
    code[music] = 1
    return code, p


def runs(code: NDArray[np.int8]) -> pl.DataFrame:
    edges = np.flatnonzero(np.diff(code)) + 1
    a = np.r_[0, edges]
    b = np.r_[edges, len(code)]
    if not len(code):
        return pl.DataFrame(schema=LABELS_SCHEMA)
    return pl.DataFrame(
        {
            "start": np.round(a * STEP, 2),
            "end": np.round(b * STEP, 2),
            "label": [LABELS[int(code[i])] for i in a],
        },
        schema=LABELS_SCHEMA,
    )


def laughs_of(code: NDArray[np.int8], p: Floats) -> pl.DataFrame:
    found: Labelled
    found, _ = ndimage.label(code == 3)
    rows: list[dict[str, float | int]] = []
    for sl in ndimage.find_objects(found):
        s = sl[0]
        pieces: int = ndimage.label(p[s] > 0)[1]
        rows.append(
            {
                "start": round(s.start * STEP, 2),
                "end": round(s.stop * STEP, 2),
                "pieces": int(pieces),
                "p": round(float(p[s].max()), 3),
            }
        )
    return pl.DataFrame(rows, schema=LAUGHS_SCHEMA)


@dataclass(frozen=True)
class Inputs:
    """What the cascade reads for one file, its times on the stem's clock."""

    stem: Floats32
    mix: Floats32
    words: pl.DataFrame
    praat: pl.DataFrame
    timeline: pl.DataFrame


def cascade(
    inputs: Inputs, model: Classifier, net: LaughterNet
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The runs of each label and the laughs, on the stem's clock."""
    stem_level = level(inputs.stem)
    n = len(stem_level)
    c = candidates(stem_level, level(inputs.mix), inputs.words, inputs.praat, inputs.timeline)
    if c.height:
        c = with_network(c, laughter_p(mel(inputs.stem), net))
    c = kept(c, model)
    talk = speech_mask(inputs.words, inputs.praat, n)
    music = shares(inputs.timeline, n)["music"] > MUSIC
    code, p = assemble(c, talk, music)
    return runs(code), laughs_of(code, p)


def on_stem_clock(df: pl.DataFrame, columns: tuple[str, ...], start: float) -> pl.DataFrame:
    """A table read from the annotations, its times moved back from the mp4's
    clock to the stem's."""
    return df.with_columns([pl.col(k) - start for k in columns]) if start else df


def process_laughter(stem: Path, video: Path, od: Path, start: float | None = None) -> None:
    """`stem` is the speech stem, `video` the mp4 whose first audio stream is
    the mix, and `start` where the sound begins on the mp4's timeline, from
    audio_clock.start."""
    if start is None:
        raise ValueError(f"{od.name}: process_laughter needs the audio start; see audio_clock")
    laughs_path, labels_path = od / "laughter.parquet", od / "laughter_labels.parquet"
    if laughs_path.exists() and labels_path.exists():
        log_status(od, "Already exists")
        return
    missing = [name for name in INPUTS if not (od / name).exists()]
    if missing:
        log_status(od, f"No {', '.join(missing)} — skipped")
        return
    timeline = pl.read_parquet(od / "audioseg_timeline.parquet")
    if not {"bass", "drums", "other", "speech"} <= set(timeline.columns):
        log_status(od, "audioseg_timeline.parquet has no stem shares — skipped")
        return

    t0 = time.monotonic()
    inputs = Inputs(
        stem=decode(stem),
        mix=decode(video),
        words=on_stem_clock(pl.read_parquet(od / "words.parquet"), ("start", "end"), start),
        praat=on_stem_clock(pl.read_parquet(od / "praat.parquet"), ("time",), start),
        timeline=on_stem_clock(timeline, ("time",), start),
    )
    labels, laughs = cascade(inputs, classifier(), network())
    audio_clock.write(labels, "laughter", labels_path, start)
    audio_clock.write(laughs, "laughter", laughs_path, start)
    seconds = float((laughs["end"] - laughs["start"]).sum()) if laughs.height else 0.0
    log_status(
        od,
        f"Saved {laughs.height} laughs, {seconds:.0f}s, in {time.monotonic() - t0:.0f}s",
    )


def models_missing() -> list[str]:
    """The laughter weights that are not in the model directory."""
    where = config.models() / "laughter"
    return [name for name in (NET_FILE, CLASSIFIER_FILE) if not (where / name).exists()]
