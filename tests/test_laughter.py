"""The laughter stage on made-up sound: the masks, the candidates, the
classifier's trees, the cascade's labels and the stage end to end. The
network is drawn from a seeded generator and the classifier is written by
hand, so what surrounds them is what is tested; 2026_laugh holds the check
of the trained weights against the paper's own code."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf
import torch
from conftest import needs_ffmpeg
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from dvt import config, laughter
from dvt.audio_clock import AUDIO_START
from dvt.laughter_net import LaughterNet

STEP = laughter.STEP


def at(seconds: float) -> int:
    return int(round(seconds / STEP))


def mask(n: int, *spans: tuple[float, float]) -> NDArray[np.bool_]:
    on = np.zeros(n, bool)
    for a, b in spans:
        on[at(a) : at(b)] = True
    return on


def words_frame(*spans: tuple[float, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"word": ["w"] * len(spans), "start": [a for a, _ in spans], "end": [b for _, b in spans]},
        schema={"word": pl.String, "start": pl.Float64, "end": pl.Float64},
    )


def praat_frame(n: int, voiced: NDArray[np.bool_] | None = None, hnr: float = 10.0) -> pl.DataFrame:
    on = np.zeros(n, bool) if voiced is None else voiced
    return pl.DataFrame(
        {
            "time": np.arange(n) * STEP,
            "f0_hz": pl.Series(np.where(on, 150.0, np.nan)).fill_nan(None),
            "hnr_db": np.full(n, hnr),
        }
    )


def timeline_frame(seconds: float, music: tuple[float, float] | None = None) -> pl.DataFrame:
    t = np.arange(0, seconds, 0.05)
    loud = np.zeros(len(t)) if music is None else ((t >= music[0]) & (t < music[1])) * 0.8
    return pl.DataFrame(
        {
            "time": t,
            "bass": loud / 2,
            "drums": loud / 2,
            "other": np.full(len(t), 0.1),
            "speech": 0.9 - loud,
        }
    )


# ---- masks ---------------------------------------------------------------------


@given(st.lists(st.booleans(), max_size=200), st.sampled_from([0.01, 0.05, 0.3, 0.5]))
def test_close_only_adds(bits: list[bool], sec: float) -> None:
    on = np.array(bits, bool)
    closed = laughter.close(on, sec)
    assert len(closed) == len(on)
    assert not (on & ~closed).any()
    assert laughter.close(closed, sec).tolist() == closed.tolist()


def test_close_fills_short_gaps_only() -> None:
    on = mask(300, (0.0, 1.0), (1.2, 2.0), (2.6, 3.0))
    closed = laughter.close(on, 0.3)
    assert closed[at(1.0) : at(1.2)].all()
    assert not closed[at(2.0) : at(2.6)].any()


@given(st.lists(st.tuples(st.floats(-5, 50), st.floats(0, 10)), max_size=10), st.integers(0, 3000))
def test_frames(spans: list[tuple[float, float]], n: int) -> None:
    got = laughter.frames(((a, a + d) for a, d in spans), n)
    want = np.zeros(n, bool)
    for a, d in spans:
        want[max(0, int(a / STEP)) : max(0, int((a + d) / STEP))] = True
    assert got.tolist() == want.tolist()


def test_speech_mask() -> None:
    """A word is padded by PAD and short gaps between words closed; a word
    longer than LONG_WORD is not speech; a run of voicing that touches a
    word is speech whole; voicing that touches none is speech only where
    half of VOICED_WIN is voiced at a good HNR."""
    n = 2000
    voiced = mask(n, (4.9, 6.5), (12.0, 12.2))
    words = words_frame((2.0, 2.4), (2.6, 3.0), (8.0, 9.5), (5.0, 5.3))
    talk = laughter.speech_mask(words, praat_frame(n, voiced), n)
    assert talk[at(1.9) : at(3.1)].all()
    assert not talk[at(1.8)]
    assert not talk[at(8.0) : at(9.5)].any()
    assert talk[at(4.9) : at(6.5)].all()
    assert not talk[at(12.0) : at(12.2)].any()
    held = laughter.speech_mask(words_frame(), praat_frame(n, mask(n, (15.0, 17.0))), n)
    assert held[at(15.3) : at(16.7)].all()
    low = laughter.speech_mask(words_frame(), praat_frame(n, mask(n, (15.0, 17.0)), hnr=2.0), n)
    assert not low.any()


def test_shares_and_local_floor() -> None:
    share = laughter.shares(timeline_frame(10.0, music=(2.0, 4.0)), 1000)
    assert set(share) == {"music", "other", "speech"}
    assert share["music"][at(3.0)] == pytest.approx(0.8)
    assert share["music"][at(6.0)] == 0.0
    lv = np.full(12000, -40.0, np.float32)
    lv[:6000] = -60.0
    floor = laughter.local_floor(lv, np.ones(12000, bool))
    assert floor[at(10.0)] == pytest.approx(-60.0)
    assert floor[at(110.0)] == pytest.approx(-40.0)
    few = laughter.local_floor(lv, mask(12000, (0.0, 1.0), (100.0, 101.0)))
    assert few[at(50.0)] == pytest.approx(
        float(np.percentile(lv[mask(12000, (0.0, 1.0), (100.0, 101.0))], 20))
    )


# ---- candidates ----------------------------------------------------------------


def scene(n: int = 3000) -> tuple[NDArray[np.float32], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """30 s at -60 dB with lines at 2-6 s and 14-18 s at -30 dB, a burst at
    -40 dB from 8 to 10 s, one under the second line, one inside music from
    22 to 24 s, and one too short at 26 s."""
    lv = np.full(n, -60.0, np.float32)
    for a, b, db in (
        (2, 6, -30),
        (14, 18, -30),
        (8, 10, -40),
        (16, 17, -25),
        (22, 24, -40),
        (26, 26.3, -40),
    ):
        lv[at(a) : at(b)] = db
    words = words_frame(
        (2.0, 2.5),
        (3.0, 3.5),
        (4.0, 4.5),
        (5.0, 5.5),
        (14.0, 15.0),
        (15.5, 16.0),
        (16.5, 17.5),
        (17.6, 17.9),
    )
    return lv, words, praat_frame(n), timeline_frame(n * STEP, music=(21.0, 25.0))


def test_candidates() -> None:
    lv, words, praat, timeline = scene()
    c = laughter.candidates(lv, lv - 3.0, words, praat, timeline)
    assert c.columns == list(laughter.CANDIDATE_COLUMNS)
    assert c.height == 1
    row = c.row(0, named=True)
    assert (row["start"], row["end"]) == pytest.approx((8.0, 10.0), abs=0.15)
    assert row["peak_db"] == pytest.approx(-40.0, abs=0.1)
    assert row["over_speech_db"] == pytest.approx(-10.0, abs=0.1)
    assert row["residual_db"] == pytest.approx(0.0, abs=0.1)
    assert row["after_speech"] == pytest.approx(row["start"] - 5.6, abs=0.15)
    assert row["music"] == 0.0


def test_candidates_of_nothing() -> None:
    n = 1000
    silent = np.full(n, -120.0, np.float32)
    c = laughter.candidates(silent, silent, words_frame(), praat_frame(n), timeline_frame(10.0))
    assert c.height == 0
    assert c.schema == laughter.CANDIDATE_SCHEMA


def test_level_of_a_known_signal() -> None:
    y = np.full(16000, 0.1, np.float32)
    lv = laughter.level(y)
    assert lv.dtype == np.float32
    assert len(lv) == 100
    assert lv[50] == pytest.approx(-20.0, abs=0.01)


# ---- the classifier ------------------------------------------------------------


def tree(
    feature: int, threshold: float, below: float, above: float, missing_left: bool
) -> dict[str, list[float] | list[int] | list[bool]]:
    return {
        "feature": [feature, 0, 0],
        "threshold": [threshold, 0.0, 0.0],
        "missing_left": [missing_left, False, False],
        "left": [1, 0, 0],
        "right": [2, 0, 0],
        "leaf": [False, True, True],
        "value": [0.0, below, above],
    }


def classifier_json(cut: float = 0.5, baseline: float = 0.0) -> str:
    return json.dumps(
        {
            "features": ["a", "b"],
            "cut": cut,
            "baseline": baseline,
            "trees": [tree(0, 1.0, -2.0, 2.0, True), tree(1, 0.0, 0.5, -0.5, False)],
        }
    )


@given(
    st.lists(
        st.tuples(
            st.one_of(st.floats(-5, 5), st.just(np.nan)),
            st.one_of(st.floats(-5, 5), st.just(np.nan)),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_classifier_scores_its_trees(rows: list[tuple[float, float]]) -> None:
    model = laughter.parse_classifier(classifier_json(baseline=0.25))
    x = np.array(rows, dtype=np.float64)
    want: list[float] = []
    for a, b in rows:
        raw = 0.25
        raw += -2.0 if (np.isnan(a) or a <= 1.0) else 2.0
        raw += -0.5 if (np.isnan(b) or b > 0.0) else 0.5
        want.append(float(1 / (1 + np.exp(-raw))))
    assert model.score(x) == pytest.approx(want)


def test_classifier_refuses_what_it_cannot_read() -> None:
    good: dict[str, object] = json.loads(classifier_json())
    first = {**json.loads(classifier_json())["trees"][0]}
    changes: list[tuple[dict[str, object], str]] = [
        ({"features": []}, "features"),
        ({"trees": []}, "no trees"),
        ({"cut": "high"}, "cut"),
        ({"trees": [{**first, "leaf": [0, 1, 1]}]}, "leaf"),
        ({"trees": [{**first, "value": [0.0, "x", 1.0]}]}, "value"),
    ]
    assert laughter.parse_classifier(json.dumps(good)).features == ("a", "b")
    for change, msg in changes:
        with pytest.raises(ValueError, match=msg):
            laughter.parse_classifier(json.dumps({**good, **change}))


def test_kept() -> None:
    model = laughter.parse_classifier(classifier_json(cut=0.5))
    c = pl.DataFrame({"a": [0.0, 3.0, np.nan], "b": [1.0, -1.0, 1.0]})
    got = laughter.kept(c, model)
    assert got["a"].to_list() == [3.0]
    empty = laughter.kept(c.clear(), model)
    assert empty.height == 0
    assert empty.schema["p"] == pl.Float64


# ---- the network ---------------------------------------------------------------


def seeded_net() -> LaughterNet:
    """Weights from a seeded generator, as conftest's TransNetV2 has."""
    net = LaughterNet()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.as_tensor(rng.normal(0.0, 0.05, tuple(p.shape))))
    net.eval()
    return net


def test_mel_and_network_shapes() -> None:
    y = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.05).astype(np.float32)
    spec = laughter.mel(y)
    assert spec.dtype == np.float16
    # one row a HOP, and one more from the padding, as the network was trained on
    assert spec.shape == (301, laughter.N_MELS)
    fb = laughter.mel_filters()
    assert fb.shape == (laughter.N_MELS, laughter.N_FFT // 2 + 1)
    assert (fb >= 0).all()
    assert (fb.max(1) > 0).all()
    p = laughter.laughter_p(spec, seeded_net())
    assert p.shape == (150,)
    assert ((p >= 0) & (p <= 1)).all()
    quiet = laughter.laughter_p(np.full((300, laughter.N_MELS), -100.0, np.float16), seeded_net())
    assert quiet.tolist() == [0.0] * 150


@settings(max_examples=12, deadline=None)
@given(st.integers(1, 400), st.sampled_from([2, 64, 128, 1000]), st.integers(0, 10_000))
def test_network_in_pieces_is_the_network_whole(half: int, chunk: int, seed: int) -> None:
    """The network run over pieces, each layer's GroupNorm statistics
    gathered first, gives what one pass over the whole spectrogram gives."""
    rng = np.random.default_rng(seed)
    spec = rng.normal(-60.0, 15.0, (2 * half, laughter.N_MELS)).astype(np.float16)
    net = seeded_net()
    whole = torch.as_tensor(np.asarray(spec, np.float32) + 5.0).unsqueeze(0)
    with torch.no_grad():
        want = torch.sigmoid(net(whole)[0, :, 0]).numpy()
    got = net.laughter(spec, -5.0, chunk=chunk)
    assert got.shape == want.shape
    assert got == pytest.approx(want, abs=1e-5)


def test_network_wants_even_lengths() -> None:
    net = seeded_net()
    with pytest.raises(ValueError, match="even number"):
        net.laughter(np.zeros((3, laughter.N_MELS), np.float16), 0.0)
    with pytest.raises(ValueError, match="even number"):
        net.laughter(np.zeros((4, laughter.N_MELS), np.float16), 0.0, chunk=3)


@settings(max_examples=12, deadline=None)
@given(st.integers(0, 40_000), st.sampled_from([1, 7, 100, 65536]), st.integers(0, 10_000))
def test_mel_in_pieces_is_the_mel_whole(samples: int, chunk: int, seed: int) -> None:
    """The spectrogram made a few frames at a time is the same, number for
    number, as the one made in one go."""
    y = (np.random.default_rng(seed).standard_normal(samples) * 0.1).astype(np.float32)
    whole = laughter.mel(y, chunk=10**9)
    pieces = laughter.mel(y, chunk=chunk)
    assert pieces.shape == whole.shape == (samples // laughter.HOP + 1, laughter.N_MELS)
    assert np.array_equal(pieces, whole)


def test_level_sets_the_network_input() -> None:
    """The spectrogram is moved to its median level, so a louder copy of
    the same sound gives the same p."""
    y = (np.random.default_rng(1).standard_normal(16000 * 2) * 0.05).astype(np.float32)
    net = seeded_net()
    a = laughter.laughter_p(laughter.mel(y), net)
    b = laughter.laughter_p(laughter.mel(y * 4), net)
    assert a == pytest.approx(b, abs=2e-3)


def test_with_network() -> None:
    c = pl.DataFrame({"start": [0.0, 1.0, 50.0], "end": [0.5, 1.01, 51.0]})
    p = np.linspace(0, 1, 100).astype(np.float32)
    got = laughter.with_network(c, p)
    assert got["len"].to_list() == pytest.approx([0.5, 0.01, 1.0])
    assert got["net_mean"][0] == pytest.approx(float(p[:25].mean()))
    assert got["net_max"][1] == pytest.approx(float(p[50]))
    assert got["net_mean"].is_nan()[2]


# ---- the cascade ---------------------------------------------------------------


def test_assemble_and_runs() -> None:
    """Music over speech over laughter; pieces of a laugh joined across a
    gap under MERGE with no speech or music in it, not otherwise: the three
    from 1 s join up to the line at 5 s, and the one after the line stays
    apart, though the line is shorter than MERGE."""
    n = 2000
    talk = mask(n, (5.0, 6.0))
    music = mask(n, (15.0, 17.0))
    laughs = pl.DataFrame(
        {
            "start": [1.0, 2.5, 4.0, 6.2, 14.0],
            "end": [2.0, 3.5, 5.5, 7.0, 16.0],
            "p": [0.8, 0.9, 0.7, 0.75, 0.95],
        }
    )
    code, p = laughter.assemble(laughs, talk, music)
    assert (code[at(1.0) : at(3.5)] == 3).all()
    assert p[at(2.2)] == 0.0
    assert (code[at(5.0) : at(6.0)] == 2).all()
    assert (code[at(6.0) : at(6.2)] == 0).all()
    assert (code[at(15.0) : at(17.0)] == 1).all()
    assert (code[at(14.0) : at(15.0)] == 3).all()
    runs = laughter.runs(code)
    assert runs["label"].to_list()[:4] == ["other", "laughter", "speech", "other"]
    assert runs["start"][1] == 1.0
    assert runs["end"][1] == 5.0
    assert (runs["end"].shift(1).drop_nulls() == runs["start"].slice(1)).all()
    found = laughter.laughs_of(code, p)
    first = found.row(0, named=True)
    assert (first["start"], first["end"], first["pieces"], first["p"]) == (1.0, 5.0, 3, 0.9)
    assert laughter.runs(np.zeros(0, np.int8)).height == 0


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, 3), min_size=1, max_size=300))
def test_runs_cover_every_frame(codes: list[int]) -> None:
    code = np.array(codes, np.int8)
    runs = laughter.runs(code)
    back = np.zeros(len(code), np.int8)
    for a, b, lab in runs.rows():
        back[at(a) : at(b)] = laughter.LABELS.index(lab)
    assert back.tolist() == code.tolist()


# ---- the stage -----------------------------------------------------------------


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    where = tmp_path / "models"
    (where / "laughter").mkdir(parents=True)
    torch.save(seeded_net().state_dict(), where / "laughter" / laughter.NET_FILE)
    keep_all = {
        "features": [*laughter.CANDIDATE_COLUMNS[2:], "len", "net_mean", "net_max"],
        "cut": 0.5,
        "baseline": 5.0,
        "trees": [
            {
                "feature": [0],
                "threshold": [0.0],
                "missing_left": [True],
                "left": [0],
                "right": [0],
                "leaf": [True],
                "value": [0.0],
            }
        ],
    }
    (where / "laughter" / laughter.CLASSIFIER_FILE).write_text(json.dumps(keep_all))
    monkeypatch.setattr(config, "_models", where)
    laughter.classifier.cache_clear()
    laughter.network.cache_clear()
    return where


def episode(tmp_path: Path, start: float) -> tuple[Path, Path, Path]:
    """A 30 s stem: noise at -60 dB, two lines of louder noise and a
    louder burst between them; a video whose sound is the mix; and the tables, on the
    mp4's clock, `start` after the stem's."""
    rate = 16000
    rng = np.random.default_rng(2)
    y = rng.standard_normal(30 * rate).astype(np.float32) * 10 ** (-60 / 20)
    for a, b, db in ((2, 6, -25), (14, 18, -25), (8, 10, -35)):
        y[a * rate : b * rate] *= 10 ** ((db + 60) / 20)
    stem = tmp_path / "speech.flac"
    sf.write(str(stem), y, rate)
    mix = tmp_path / "v.mp4"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(stem), "-c:a", "aac", str(mix)], check=True
    )
    od = tmp_path / "Film"
    od.mkdir()
    words = words_frame(
        (2.0, 2.8),
        (3.0, 3.8),
        (4.0, 4.8),
        (5.0, 5.8),
        (14.0, 14.8),
        (15.0, 15.8),
        (16.0, 16.8),
        (17.0, 17.8),
    )
    words.with_columns(pl.col("start", "end") + start).write_parquet(od / "words.parquet")
    praat_frame(3000).with_columns(pl.col("time") + start).write_parquet(od / "praat.parquet")
    timeline_frame(30.0).with_columns(pl.col("time") + start).write_parquet(
        od / "audioseg_timeline.parquet"
    )
    return stem, mix, od


@needs_ffmpeg
def test_process_laughter(tmp_path: Path, models: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stem, mix, od = episode(tmp_path, 0.5)
    laughter.process_laughter(stem, mix, od, start=0.5)
    laughs = pl.read_parquet(od / "laughter.parquet")
    labels = pl.read_parquet(od / "laughter_labels.parquet")
    assert laughs.drop("version").schema == pl.Schema(laughter.LAUGHS_SCHEMA)
    assert laughs.height == 1
    assert laughs["start"][0] == pytest.approx(8.5, abs=0.2)
    assert laughs["end"][0] == pytest.approx(10.5, abs=0.2)
    assert labels["start"][0] == pytest.approx(0.5)
    assert set(labels["label"]) == {"other", "speech", "laughter"}
    assert pl.read_parquet_metadata(od / "laughter.parquet")[AUDIO_START] == "0.5"
    assert "Saved 1 laughs" in capsys.readouterr().out
    laughter.process_laughter(stem, mix, od, start=0.5)
    assert capsys.readouterr().out == "[Film:laughter] Already exists\n"


@needs_ffmpeg
def test_process_laughter_skips(
    tmp_path: Path, models: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stem, mix, od = episode(tmp_path, 0.0)
    with pytest.raises(ValueError, match="needs the audio start"):
        laughter.process_laughter(stem, mix, od)
    (od / "praat.parquet").unlink()
    laughter.process_laughter(stem, mix, od, start=0.0)
    assert capsys.readouterr().out == "[Film:laughter] No praat.parquet — skipped\n"
    praat_frame(3000).write_parquet(od / "praat.parquet")
    pl.DataFrame({"time": [0.0], "speech": [1.0]}).write_parquet(od / "audioseg_timeline.parquet")
    laughter.process_laughter(stem, mix, od, start=0.0)
    assert "has no stem shares — skipped" in capsys.readouterr().out
    assert not (od / "laughter.parquet").exists()


def test_models_missing(models: Path) -> None:
    assert laughter.models_missing() == []
    (models / "laughter" / laughter.NET_FILE).unlink()
    assert laughter.models_missing() == [laughter.NET_FILE]
