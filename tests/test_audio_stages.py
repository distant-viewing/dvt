"""The audio stages, with the trained models stood in for: praat on made-up
tones, audioseg with a fake separator, and words with a fake whisperx and
voice detector, so what surrounds each model is what is tested."""

import itertools
import random
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf
import torch
from conftest import needs_ffmpeg
from hypothesis import given
from hypothesis import strategies as st

from dvt import align_paths, audioseg, config, praat, stage_language, words
from dvt.audio_clock import AUDIO_START
from dvt.source import Source


def tone(seconds: float, hz: float = 150.0, rate: int = 16_000, voiced: float = 1.0) -> np.ndarray:
    """A sawtooth at hz for the first `voiced` share of the time, then silence."""
    t = np.arange(int(seconds * rate)) / rate
    wave = 0.3 * (2.0 * ((t * hz) % 1.0) - 1.0)
    wave[t >= seconds * voiced] = 0.0
    return wave.astype(np.float32)


# ---- praat ----------------------------------------------------------------------


def test_process_praat(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stem = tmp_path / "speech.flac"
    sf.write(str(stem), tone(3.0, voiced=0.5), 16_000)
    od = tmp_path / "Film"
    praat.process_praat(stem, od, start=0.5)
    df = pl.read_parquet(od / "praat.parquet")
    assert df.drop("version").schema == pl.Schema(praat.SCHEMA)
    assert df.height == 300
    assert df["time"][0] == pytest.approx(0.5)
    assert df["time"][-1] == pytest.approx(0.5 + 2.99)
    assert pl.read_parquet_metadata(od / "praat.parquet")[AUDIO_START] == "0.5"
    voiced = df.filter(pl.col("f0_hz").is_not_null())
    assert voiced.height > 100
    assert voiced["f0_hz"].median() == pytest.approx(150.0, rel=0.02)
    assert voiced["time"].to_numpy().max() < 0.5 + 1.6
    assert (
        df.filter(pl.col("time") > 2.5)["f0_hz"].null_count()
        == df.filter(pl.col("time") > 2.5).height
    )
    assert voiced["jitter_local"].is_not_null().any()
    assert df["cog_hz"].is_not_null().any()
    assert "300 frames" in capsys.readouterr().out
    praat.process_praat(stem, od, start=0.5)
    assert capsys.readouterr().out == "[Film:praat] Already exists\n"


def test_praat_short_stem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stem = tmp_path / "speech.flac"
    sf.write(str(stem), tone(1.0), 16_000)
    praat.process_praat(stem, tmp_path / "Film", start=0.0)
    df = pl.read_parquet(tmp_path / "Film" / "praat.parquet")
    assert df.height == 0
    assert df.drop("version").schema == pl.Schema(praat.SCHEMA)
    assert "too short to analyse" in capsys.readouterr().out


def test_praat_needs_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs the audio start"):
        praat.process_praat(tmp_path / "s.flac", tmp_path / "Film")


def test_praat_measures_the_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stem = tmp_path / "speech.flac"
    sf.write(str(stem), tone(5.0), 16_000)
    monkeypatch.setattr(praat, "BLOCK_SEC", 2.0)
    monkeypatch.setattr(praat, "MARGIN_SEC", 0.5)
    praat.process_praat(stem, tmp_path / "Film", start=0.0)
    df = pl.read_parquet(tmp_path / "Film" / "praat.parquet")
    assert df["time"].to_list() == pytest.approx([i / 100 for i in range(500)])
    assert df.filter(pl.col("time") > 4.0)["f0_hz"].is_not_null().sum() > 80


def test_praat_blocks_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stem = tmp_path / "speech.flac"
    sf.write(str(stem), np.concatenate([tone(3.0), tone(3.0, hz=200.0)]), 16_000)
    monkeypatch.setattr(praat, "BLOCK_SEC", 2.0)
    monkeypatch.setattr(praat, "MARGIN_SEC", 0.5)
    praat.process_praat(stem, tmp_path / "Film", start=0.0)
    df = pl.read_parquet(tmp_path / "Film" / "praat.parquet")
    assert df["time"].to_list() == pytest.approx([i / 100 for i in range(600)])
    early = df.filter(pl.col("time").is_between(0.5, 2.0))["f0_hz"].median()
    late = df.filter(pl.col("time").is_between(3.5, 5.5))["f0_hz"].median()
    assert (early, late) == (pytest.approx(150.0, rel=0.02), pytest.approx(200.0, rel=0.02))


@given(st.floats(min_value=0.0, max_value=5000.0))
def test_praat_blocks_cover(duration: float) -> None:
    blocks = praat._blocks(duration)  # pyright: ignore[reportPrivateUsage]
    if duration == 0.0:
        assert blocks == []
        return
    assert blocks[0][0] == 0.0
    assert blocks[-1][1] == duration
    for (_, b), (c, _) in itertools.pairwise(blocks):
        assert b == c
    assert all(b - a <= praat.BLOCK_SEC + praat.MIN_BLOCK_SEC for a, b in blocks)
    if len(blocks) > 1:
        assert blocks[-1][1] - blocks[-1][0] >= praat.MIN_BLOCK_SEC


@given(st.lists(st.booleans(), max_size=60))
def test_voiced_runs(voiced: list[bool]) -> None:
    f0 = np.where(np.array(voiced, dtype=bool), 120.0, np.nan)
    grid = np.zeros(len(voiced))
    runs = praat._voiced_runs(f0, grid)  # pyright: ignore[reportPrivateUsage]
    least = max(2, round(praat.MIN_VOICED_SEC / praat.TIME_STEP))
    expected: list[tuple[int, int]] = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            if j - i >= least:
                expected.append((i, j))
            i = j
        else:
            i += 1
    assert runs == expected
    assert all(type(a) is int and type(b) is int for a, b in runs)


# ---- audioseg: the activity masks ------------------------------------------------


def runs_of(mask: list[bool], value: bool) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        if mask[i] == value:
            out.append((i, j))
        i = j
    return out


def test_hysteresis() -> None:
    db = np.array([0, 25, 18, 16, 14, 30, 10], dtype=np.float32)
    assert audioseg._hysteresis(db, 20, 15).tolist() == [  # pyright: ignore[reportPrivateUsage]
        False, True, True, True, False, True, False,
    ]  # fmt: skip


@given(st.lists(st.booleans(), max_size=60), st.integers(1, 6))
def test_bridge_gaps(mask: list[bool], gap: int) -> None:
    out = audioseg._bridge_gaps(np.array(mask, dtype=bool), gap).tolist()  # pyright: ignore[reportPrivateUsage]
    for a, b in runs_of(mask, False):
        inner = a > 0 and b < len(mask)
        assert all(out[a:b]) == (inner and b - a <= gap)


@given(st.lists(st.booleans(), max_size=60), st.integers(1, 6))
def test_remove_short(mask: list[bool], least: int) -> None:
    out = audioseg._remove_short(np.array(mask, dtype=bool), least).tolist()  # pyright: ignore[reportPrivateUsage]
    for a, b in runs_of(mask, True):
        assert out[a:b] == [b - a >= least] * (b - a)
    assert not any(o and not m for o, m in zip(out, mask, strict=True))


def test_detect_intervals() -> None:
    db = np.full(300, -60.0, dtype=np.float32)
    db[50:150] = -20.0
    db[155:170] = -30.0
    db[250:255] = -20.0
    cfg = {"enter": 20, "exit": 15, "gap": 200, "min": 100}
    got = audioseg._detect_intervals(db, -60.0, 40.0, cfg)  # pyright: ignore[reportPrivateUsage]
    assert got == [(0.5, 1.7, pytest.approx((100 * 40 + 15 * 30) / 120 / 40))]
    cfg["gap"] = 40
    got = audioseg._detect_intervals(db, -60.0, 40.0, cfg)  # pyright: ignore[reportPrivateUsage]
    assert got == [(0.5, 1.5, 1.0), (1.55, 1.7, 0.75)]


@given(
    st.integers(0, 5000), st.integers(10, 400), st.integers(0, 9), st.integers(0, 200)
)  # fmt: skip
def test_chunk_spans(total: int, chunk: int, overlap: int, minimum: int) -> None:
    spans = audioseg._chunk_spans(total, chunk, overlap, minimum)  # pyright: ignore[reportPrivateUsage]
    assert spans[0][0] == 0
    assert spans[-1][1] == total
    for (_, b), (c, _) in itertools.pairwise(spans):
        assert c == b - overlap
    assert all(b - a <= chunk + minimum for a, b in spans)


def test_rms_db() -> None:
    t = np.arange(16_000) / 16_000
    wave = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    db = audioseg._rms_db(wave)  # pyright: ignore[reportPrivateUsage]
    assert len(db) == (16_000 - audioseg.FRAME) // audioseg.HOP + 1
    assert db.dtype == np.float32
    assert float(np.median(db)) == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.01)


def test_shift_seed() -> None:
    assert audioseg._shift_seed("Film") == audioseg._shift_seed("Film")  # pyright: ignore[reportPrivateUsage]
    assert audioseg._shift_seed("Film") != audioseg._shift_seed("Other")  # pyright: ignore[reportPrivateUsage]
    state = random.getstate()
    with audioseg._seeded_random(3):  # pyright: ignore[reportPrivateUsage]
        first = random.random()
    with audioseg._seeded_random(3):  # pyright: ignore[reportPrivateUsage]
        assert random.random() == first
    assert random.getstate() == state


def test_pick(tmp_path: Path) -> None:
    files = ["a_(Vocals).wav", "a_(Other).wav"]
    pick = audioseg._pick  # pyright: ignore[reportPrivateUsage]
    assert pick(files, "(vocals)", tmp_path, "x") == tmp_path / files[0]
    assert pick(files, "(Vocals)", tmp_path, "x", without=True) == tmp_path / files[1]
    with pytest.raises(RuntimeError, match="returned 0 file"):
        pick([], "(Bass)", tmp_path, "x")


def test_pad_and_trim(tmp_path: Path) -> None:
    wav = tmp_path / "mix.wav"
    sf.write(str(wav), np.full((1000, 2), 0.25, dtype=np.float32), 8000, subtype="PCM_16")
    assert audioseg._pad_short(wav) == 1000  # pyright: ignore[reportPrivateUsage]
    assert sf.info(str(wav)).frames == audioseg.MIN_SEPARABLE_SEC * 8000
    assert audioseg._pad_short(wav) is None  # pyright: ignore[reportPrivateUsage]
    audioseg._trim_stems({"x": wav}, 1000)  # pyright: ignore[reportPrivateUsage]
    data, _ = sf.read(str(wav), always_2d=True)
    assert data.shape == (1000, 2)
    assert data == pytest.approx(0.25, abs=1e-4)


# ---- audioseg: separation with a fake separator ------------------------------------

DEMUCS = {"(Bass)": 0.1, "(Drums)": 0.2, "(Other)": 0.3, "(Vocals)": 0.4}
MEL = {"(other)": 0.25, "(vocals)": 0.75}


class FakeSeparator:
    """Writes each stem as the input scaled by a fixed share, as the
    separator names its outputs."""

    def __init__(self, work: Path, device: str = "cuda") -> None:
        self.work = work
        self.torch_device = torch.device(device)
        self.shares = DEMUCS
        self.calls = 0

    def load_model(self, name: str) -> None:
        self.shares = DEMUCS if name.startswith("htdemucs") else MEL

    def separate(self, audio_file_path: str) -> list[str]:
        self.calls += 1
        path = audio_file_path
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        names: list[str] = []
        for marker, share in self.shares.items():
            name = f"{Path(path).stem}_{marker}.wav"
            sf.write(str(self.work / name), data * share, rate, subtype="PCM_16")
            names.append(name)
        return names


def test_separate_chunked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audioseg, "SEPARATE_CHUNK_SEC", 1.0)
    monkeypatch.setattr(audioseg, "SEPARATE_OVERLAP_SEC", 0.1)
    monkeypatch.setattr(audioseg, "MIN_CHUNK_SEC", 0.3)
    rate = audioseg.SEPARATOR_SR
    t = np.arange(int(3.5 * rate)) / rate
    mix = np.stack([np.sin(2 * np.pi * 220 * t), np.cos(2 * np.pi * 330 * t)], axis=1) * 0.5
    mix[int(0.85 * rate) : int(2.85 * rate)] = 0.0
    src = tmp_path / "mix.wav"
    sf.write(str(src), mix.astype(np.float32), rate, subtype="PCM_16")
    sep = FakeSeparator(tmp_path)
    wanted = {"bass": ("(Bass)", False), "vocals": ("(Vocals)", False)}
    out = audioseg._separate_chunked(sep, 1, src, tmp_path, "d", wanted, "ctx")  # pyright: ignore[reportPrivateUsage]
    assert sep.calls == 2  # four spans, two of them silent
    for name, share in (("bass", 0.1), ("vocals", 0.4)):
        got, _ = sf.read(str(out[name]), dtype="float32", always_2d=True)
        assert got.shape == mix.shape
        assert np.abs(got - mix * share).max() < 2e-3
    assert not list((tmp_path / "d_chunks").iterdir())


def test_separate_chunked_wants_the_rate(tmp_path: Path) -> None:
    src = tmp_path / "mix.wav"
    sf.write(str(src), np.zeros((100, 2), dtype=np.float32), 8000)
    with pytest.raises(RuntimeError, match="not the 44100 Hz"):
        audioseg._separate_chunked(FakeSeparator(tmp_path), 0, src, tmp_path, "d", {}, "c")  # pyright: ignore[reportPrivateUsage]


def audio_source(tmp_path: Path, seconds: float = 4.0) -> Source:
    rate = 22_050
    wav = tone(seconds, rate=rate, voiced=0.6)
    wav[int(seconds * rate * 0.8) :] = 0.2 * np.sin(np.arange(int(seconds * rate * 0.2)) / 3)
    audio = tmp_path / "audio" / "Film.wav"
    audio.parent.mkdir(parents=True)
    sf.write(str(audio), wav, rate)
    return Source(
        video=tmp_path / "Film.mp4",
        out=tmp_path / "annotations" / "Film",
        audio=audio,
        frames=tmp_path / "frames" / "Film",
        stems=str(tmp_path / "stems" / "{stem}" / "Film.flac"),
    )


@needs_ffmpeg
def test_process_audioseg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    s = audio_source(tmp_path)
    monkeypatch.setattr(config, "_tmp", tmp_path / "tmp")
    monkeypatch.setattr(audioseg, "_make_separator", FakeSeparator)
    audioseg.process_audioseg(s, start=0.25)
    out = capsys.readouterr().out
    assert "Audio is 4.0s; separated padded to 30s and cut back" in out
    assert "Kept 5 stems" in out
    for name in audioseg.TIMELINE_STEMS:
        assert sf.info(str(s.stem(name))).frames == 4 * audioseg.SEPARATOR_SR
    speech, _ = sf.read(str(s.speech), always_2d=True)
    mix, _ = sf.read(str(s.stem("other")), always_2d=True)
    assert np.abs(speech).max() == pytest.approx(np.abs(mix).max() * 0.75 * 0.4 / 0.3, rel=0.01)

    events = pl.read_parquet(s.out / audioseg.OUT_EVENTS)
    timeline = pl.read_parquet(s.out / audioseg.OUT_TIMELINE)
    assert events.drop("version").schema == pl.Schema(audioseg.EVENT_SCHEMA)
    assert events["stem"].to_list() == sorted(events["stem"].to_list())
    assert events["start"].min() == pytest.approx(0.25)
    assert timeline.columns[:3] == ["time", "energy_db", "bass"]
    assert timeline.height == 80
    assert timeline["time"][0] == pytest.approx(0.25)
    shares = timeline.select(audioseg.TIMELINE_STEMS).sum_horizontal()
    # Silence has no shares to give out.
    assert all(x == 0.0 or abs(x - 1.0) < 1e-3 for x in shares)
    assert 0 < shares.to_list().count(0.0) < 20

    audioseg.process_audioseg(s, start=0.25)
    assert capsys.readouterr().out == "[Film:audioseg] Already exists\n"
    (s.out / audioseg.OUT_EVENTS).unlink()
    audioseg.process_audioseg(s, start=0.25)
    assert "Tables from the five stems already kept" in capsys.readouterr().out
    assert pl.read_parquet(s.out / audioseg.OUT_EVENTS).equals(events)


@needs_ffmpeg
def test_audioseg_wants_a_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = audio_source(tmp_path)
    monkeypatch.setattr(config, "_tmp", tmp_path / "tmp")

    def on_cpu(work: Path) -> FakeSeparator:
        return FakeSeparator(work, "cpu")

    monkeypatch.setattr(audioseg, "_make_separator", on_cpu)
    with pytest.raises(RuntimeError, match="fell back to cpu"):
        audioseg.process_audioseg(s, start=0.0)
    with pytest.raises(ValueError, match="needs the audio start"):
        audioseg.process_audioseg(s)


def test_make_separator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(config, "_models", tmp_path)
    monkeypatch.setattr(audioseg, "Separator", Recorder)
    audioseg._make_separator(tmp_path / "w")  # pyright: ignore[reportPrivateUsage]
    assert seen["model_file_dir"] == str(tmp_path / "audio-separator")
    assert seen["sample_rate"] == audioseg.SEPARATOR_SR
    assert seen["demucs_params"] == {
        "segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True,
    }  # fmt: skip


# ---- words ---------------------------------------------------------------------------


@given(
    st.floats(0, 100), st.floats(0, 5),
    st.lists(st.tuples(st.floats(0, 100), st.floats(0, 5)), max_size=6),
)  # fmt: skip
def test_clip_to_speech(start: float, length: float, segs: list[tuple[float, float]]) -> None:
    speech = [{"start": a, "end": a + n} for a, n in segs]
    new_start, new_end = words.clip_to_speech(start, start + length, speech)
    assert start <= new_start
    assert new_end <= start + length
    if not any(s["end"] > start and s["start"] < start + length for s in speech):
        assert (new_start, new_end) == (start, start + length)


class FakeWhisper:
    """Stands in for whisperx: three words, the second unaligned."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load_audio(self, path: str) -> np.ndarray:
        self.calls.append("load_audio")
        return np.zeros(16_000 * 5, dtype=np.float32)

    def load_model(self, path: str, device: str, compute_type: str, language: str) -> "FakeWhisper":
        self.calls.append(f"load_model {language}")
        return self

    def transcribe(self, audio: np.ndarray, batch_size: int, language: str) -> dict[str, object]:
        return {"segments": [{"text": "a b c"}]}

    def load_align_model(
        self, language_code: str, device: str, model_dir: str | None
    ) -> tuple[str, dict[str, str]]:
        self.calls.append(f"align {language_code} {model_dir}")
        return "aligner", {"language": language_code}

    def align(self, segments: object, *args: object, **kwargs: object) -> dict[str, object]:
        return {
            "segments": [
                {"words": [
                    {"word": "a", "start": 0.5, "end": 1.5, "score": 0.9},
                    {"word": "b"},
                ]},
                {"words": [{"word": "c", "start": 3.0, "end": 4.0, "score": 0.8}]},
            ]
        }  # fmt: skip


@pytest.fixture
def whisper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeWhisper:
    fake = FakeWhisper()
    monkeypatch.setattr(words, "whisperx", fake)

    def timestamps(audio: torch.Tensor, model: str, **kwargs: object) -> list[dict[str, float]]:
        return [{"start": 1.0, "end": 1.2}, {"start": 3.4, "end": 5.0}]

    monkeypatch.setattr(words, "load_silero_vad", lambda: "vad")
    monkeypatch.setattr(words, "get_speech_timestamps", timestamps)
    monkeypatch.setattr(config, "_models", tmp_path / "models")
    return fake


def write_meta(od: Path, declared: str | None) -> None:
    od.mkdir(parents=True, exist_ok=True)
    schema = {"language_declared": pl.String}
    pl.DataFrame({"language_declared": [declared]}, schema=schema).write_parquet(
        od / "metadata.parquet"
    )


def test_process_m4a(
    tmp_path: Path, whisper: FakeWhisper, capsys: pytest.CaptureFixture[str]
) -> None:
    od = tmp_path / "Film"
    write_meta(od, "fra")
    words.process_m4a(tmp_path / "Film.m4a", od, start=2.0)
    df = pl.read_parquet(od / "words.parquet")
    assert df.drop("version").schema == pl.Schema(
        {"word": pl.String, "start": pl.Float64, "end": pl.Float64, "score": pl.Float64}
    )
    assert df["word"].to_list() == ["a", "b", "c"]
    assert df["start"].to_list() == [pytest.approx(2.97), None, pytest.approx(5.37)]
    assert df["end"].to_list() == [pytest.approx(3.23), None, 6.0]
    assert whisper.calls[:2] == ["load_audio", "load_model fr"]
    assert (
        whisper.calls[2]
        == f"align fr {None if 'fr' in align_paths.DEFAULT_ALIGN_MODELS_TORCH else config.models()}"
    )
    assert str(config.nltk_data()) in words.nltk.data.path
    out = capsys.readouterr().out
    assert out.startswith("Working on Film (audio).")
    assert "[Film:words] Transcribing as" in out
    words.process_m4a(tmp_path / "Film.m4a", od, start=2.0)
    assert "[Film:words] Already exists" in capsys.readouterr().out


def test_words_skips(tmp_path: Path, whisper: FakeWhisper, monkeypatch: pytest.MonkeyPatch,
                     capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    audio = np.zeros(10, dtype=np.float32)
    od = tmp_path / "A"
    write_meta(od, None)
    words.create_words(od, audio, 0.0)
    assert pl.read_parquet(od / "words.parquet").height == 0
    assert "Skipped" in capsys.readouterr().out
    od = tmp_path / "B"
    write_meta(od, "en")

    def silent(*args: object, **kwargs: object) -> list[dict[str, float]]:
        return []

    monkeypatch.setattr(words, "get_speech_timestamps", silent)
    words.create_words(od, audio, 0.0)
    assert pl.read_parquet(od / "words.parquet").height == 0
    assert "No speech detected" in capsys.readouterr().out
    assert whisper.calls == []


def test_words_wants_metadata(tmp_path: Path) -> None:
    audio = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="Run the metadata stage first"):
        words.create_words(tmp_path, audio, 0.0)
    pl.DataFrame({"n_frames": [1]}).write_parquet(tmp_path / "metadata.parquet")
    with pytest.raises(ValueError, match="predates the language columns"):
        words.create_words(tmp_path, audio, 0.0)


def test_process_m4a_choices(tmp_path: Path, whisper: FakeWhisper) -> None:
    words.process_m4a(tmp_path / "x.m4a", tmp_path / "Film", ["praat"], start=0.0)
    assert not (tmp_path / "Film").exists()
    with pytest.raises(ValueError, match="needs the audio start"):
        words.process_m4a(tmp_path / "x.m4a", tmp_path / "Film")


# ---- the aligners --------------------------------------------------------------------


def test_resolve_align_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_models", tmp_path)
    torch_lang = next(iter(align_paths.DEFAULT_ALIGN_MODELS_TORCH))
    hf_lang = next(iter(align_paths.DEFAULT_ALIGN_MODELS_HF))
    assert align_paths.resolve_align_model(torch_lang)[0] == align_paths.TORCHAUDIO
    assert align_paths.resolve_align_model(hf_lang) == (
        align_paths.HUGGINGFACE, align_paths.DEFAULT_ALIGN_MODELS_HF[hf_lang],
    )  # fmt: skip
    assert align_paths.resolve_align_model("xx", "org/model") == (
        align_paths.HUGGINGFACE,
        "org/model",
    )
    with pytest.raises(ValueError, match="no default aligner for 'xx'"):
        align_paths.resolve_align_model("xx")
    assert align_paths.align_model_dir(torch_lang) is None
    assert align_paths.align_model_dir(hf_lang) == str(tmp_path)
    langs = align_paths.supported_languages()
    assert langs == sorted(langs)
    assert {torch_lang, hf_lang} <= set(langs)


class FakeBundle:
    _path = "wav2vec.pth"

    def get_model(self) -> None:
        pass


class FakeWav2Vec:
    loads: list[tuple[str, bool]] = []  # noqa: RUF012

    @classmethod
    def from_pretrained(cls, name: str, cache_dir: Path, local_files_only: bool = False) -> None:
        if name == "missing/model":
            raise OSError("not cached")
        cls.loads.append((name, local_files_only))


@pytest.fixture
def staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "_models", tmp_path)
    monkeypatch.setattr(stage_language, "Wav2Vec2Processor", FakeWav2Vec)
    monkeypatch.setattr(stage_language, "Wav2Vec2ForCTC", FakeWav2Vec)
    pipelines = types.SimpleNamespace(WAV2VEC_FAKE=FakeBundle())
    monkeypatch.setattr(stage_language, "torchaudio", types.SimpleNamespace(pipelines=pipelines))
    monkeypatch.setitem(align_paths.DEFAULT_ALIGN_MODELS_TORCH, "zz", "WAV2VEC_FAKE")
    FakeWav2Vec.loads = []
    return tmp_path


def punkt(root: Path, lang: str) -> None:
    (root / "tokenizers" / "punkt_tab" / lang).mkdir(parents=True)


def test_stage_one(
    staging: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloads: list[str] = []

    def download(name: str, download_dir: str, quiet: bool) -> bool:
        downloads.append(name)
        punkt(Path(download_dir), "english")
        (Path(download_dir) / "tokenizers" / "punkt_tab" / "README").write_text("x")
        return True

    monkeypatch.setattr(stage_language.nltk, "download", download)
    failures = stage_language.stage_one("zz", None, verify_only=False)
    assert failures == ["zz: torchaudio checkpoint wav2vec.pth missing from "
                        f"{staging / 'torch' / 'hub' / 'checkpoints'}"]  # fmt: skip
    assert downloads == ["punkt_tab"]
    assert (staging / "nltk_data" / "tokenizers" / "punkt_tab" / "README").exists()
    (staging / "torch" / "hub" / "checkpoints").mkdir(parents=True)
    (staging / "torch" / "hub" / "checkpoints" / "wav2vec.pth").touch()
    assert stage_language.stage_one("zz", None, verify_only=False) == []
    assert downloads == ["punkt_tab"]
    out = capsys.readouterr().out
    assert "already present" in out
    assert "ok: wav2vec.pth present" in out

    assert stage_language.stage_one("xx", "org/model", verify_only=True) == []
    assert FakeWav2Vec.loads == [("org/model", True), ("org/model", True)]
    bad = stage_language.stage_one("xx", "missing/model", verify_only=True)
    assert bad == ["xx: HF aligner missing/model not loadable offline: not cached"]
    assert stage_language.stage_one("xx", None, verify_only=True)[0].startswith(
        "xx: staging failed: ValueError"
    )


def test_stage_punkt_fails(staging: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*args: object, **kwargs: object) -> bool:
        return False

    def empty(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(stage_language.nltk, "download", refused)
    with pytest.raises(RuntimeError, match="could not download punkt_tab"):
        stage_language.stage_punkt("en")
    monkeypatch.setattr(stage_language.nltk, "download", empty)
    with pytest.raises(RuntimeError, match="downloaded but"):
        stage_language.stage_punkt("en")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "not both or neither"),
        (["en", "--all"], "not both or neither"),
        (["--all", "--align-model", "x/y"], "cannot be used with --all"),
    ],
)
def test_stage_language_usage(
    argv: list[str], message: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], staging: Path,
) -> None:  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["stage_language", *argv])
    with pytest.raises(SystemExit) as exc:
        stage_language.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_stage_language_main(
    staging: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["stage_language", "zz"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert stage_language.main() == 2
    monkeypatch.delenv("HF_HUB_OFFLINE")
    punkt(staging / "nltk_data", "english")
    monkeypatch.setattr(sys, "argv", ["stage_language", "zz", "--verify-only"])
    assert stage_language.main() == 1
    assert "FAILED" in capsys.readouterr().err
    (staging / "torch" / "hub" / "checkpoints").mkdir(parents=True)
    (staging / "torch" / "hub" / "checkpoints" / "wav2vec.pth").touch()
    monkeypatch.setattr(sys, "argv", ["stage_language", "zz", "--models", str(staging)])
    assert stage_language.main() == 0
    assert "1 language(s) staged" in capsys.readouterr().out
