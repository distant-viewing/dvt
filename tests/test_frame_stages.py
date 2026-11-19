"""The stages that decode every frame or a sample of them: frames, shots,
the saved frames, and the one decode the sampled stages share."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest
from conftest import make_video, needs_ffmpeg
from hypothesis import given
from hypothesis import strategies as st

from dvt import frame_ops, sampling, shots, video
from dvt.frames import create_frames
from dvt.saved_frames import FramesStage
from dvt.source import Source

# ---- frame_ops ------------------------------------------------------------------


def solid(bgr: tuple[int, int, int], h: int = 120, w: int = 160) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def test_chroma_stats() -> None:
    grey = frame_ops.chroma_stats(solid((128, 128, 128)))
    red = frame_ops.chroma_stats(solid((0, 0, 255)))
    assert grey == (0.0, 0.0)
    assert red[0] > 50
    assert red[1] == pytest.approx(1.0)


def test_motion_thumb() -> None:
    thumb = frame_ops.motion_thumb(solid((10, 20, 30), 108, 192))
    assert thumb.shape == (frame_ops.MOTION_H, frame_ops.MOTION_W)
    assert thumb.dtype == np.float32


def meta(h: int, w: int, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0,
         ) -> pl.DataFrame:  # fmt: skip
    return pl.DataFrame({"img_height": [h], "img_width": [w], "crop_top": [top],
                         "crop_bottom": [bottom], "crop_left": [left],
                         "crop_right": [right]})  # fmt: skip


def test_mid_crop_box() -> None:
    assert frame_ops.mid_crop_box(meta(1080, 1920)) == frame_ops.FULL_BOX
    assert frame_ops.mid_crop_box(meta(1080, 1920, 140, 140)) == (14, 94, 0, 192)
    assert frame_ops.mid_crop_box(meta(1080, 1920, 540, 530)) == frame_ops.FULL_BOX


def test_source_crop_box() -> None:
    assert frame_ops.source_crop_box(meta(1080, 1920, 140, 140), 1920, 1080) == (140, 940, 0, 1920)
    assert frame_ops.source_crop_box(meta(1080, 1920, 2, 2), 1920, 1080) is None
    assert frame_ops.source_crop_box(meta(1080, 1920, 500, 500), 1920, 1080) is None


def test_box_scale() -> None:
    assert frame_ops.box_scale(frame_ops.FULL_BOX, 1920, 1080) == (30.0, 30.0)
    assert frame_ops.box_scale((14, 94, 0, 192), 1920, 1080, 96, 40) == pytest.approx((20.0, 20.0))


def test_frame_metrics() -> None:
    img = solid((50, 100, 150), 108, 192)
    img[:, 96:] = (200, 200, 200)
    m = frame_ops.frame_metrics(img)
    lmean, lstd, p5, p50, p95, edge, sat, lab_a, lab_b, hist = m
    assert 0 < lmean < 1
    assert lstd > 0
    assert p5 <= p50 <= p95
    assert edge > 0
    assert 0 < sat < 1
    assert lab_a > 0  # warm: red over green
    assert lab_b > 0  # and yellow over blue
    assert hist.shape == (frame_ops.HIST_BINS,)
    assert float(hist.sum()) == pytest.approx(1.0)


# ---- sampling -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fps", "hz", "expected"),
    [
        (25.0, None, (1, frozenset({0}))),
        (25.0, 1.0, (25, frozenset({0}))),
        (23.976, 2.0, (24, frozenset({0, 12}))),
        (30.0, 0.5, (60, frozenset({0}))),
        (0.0, 1.0, (sampling.FALLBACK_STRIDE, frozenset({0}))),
        (float("nan"), 3.0, (24, frozenset({0, 8, 16}))),
    ],
)
def test_sample_grid(fps: float, hz: float | None, expected: tuple[int, frozenset[int]]) -> None:
    assert sampling.sample_grid(fps, hz) == expected


@given(st.floats(1, 120), st.floats(0.05, 30))
def test_sample_grid_offsets_fit_the_period(fps: float, hz: float) -> None:
    base, offsets = sampling.sample_grid(fps, hz)
    assert base >= 1
    assert all(0 <= o < base for o in offsets)


def write_shots(od: Path, bounds: list[tuple[int, int]]) -> None:
    od.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame({"f0": [a for a, _ in bounds], "f1": [b for _, b in bounds],
                          "fmid": [(a + b) // 2 for a, b in bounds]})  # fmt: skip
    table.write_parquet(od / "shots.parquet")


def test_shot_backfill(tmp_path: Path) -> None:
    write_shots(tmp_path, [(0, 30), (31, 40), (41, 49), (50, 99)])
    assert sampling.shot_backfill(tmp_path, 25, frozenset({0})) == frozenset({35, 45})
    assert sampling.shot_backfill(tmp_path, 25, frozenset({0}), every=True) == frozenset(
        {15, 35, 45, 74})  # fmt: skip
    write_shots(tmp_path, [])
    assert sampling.shot_backfill(tmp_path, 25, frozenset({0})) == frozenset()


@given(st.lists(st.integers(1, 80), min_size=1, max_size=30), st.integers(1, 50),
       st.sets(st.integers(0, 49), min_size=1, max_size=5))  # fmt: skip
def test_shot_backfill_finds_every_missed_shot(lengths: list[int], period: int,
                                               offsets: set[int]) -> None:  # fmt: skip
    import tempfile

    offs = frozenset(o % period for o in offsets)
    bounds: list[tuple[int, int]] = []
    f = 0
    for n in lengths:
        bounds.append((f, f + n - 1))
        f += n
    with tempfile.TemporaryDirectory() as d:
        write_shots(Path(d), bounds)
        got = sampling.shot_backfill(Path(d), period, offs)
    missed = {(a + b) // 2 for a, b in bounds
              if not any(x % period in offs for x in range(a, b + 1))}  # fmt: skip
    assert got == missed


class Stage(sampling.FrameStage):
    label = "test"

    def __init__(self, od: Path, hz: float | None = 1.0, backfill: bool = False) -> None:
        super().__init__(od)
        self.sample_hz = hz
        self.backfill_shots = backfill
        self.seen: list[int] = []

    def done(self) -> bool:
        return False

    def handle(self, fnum: int, bgr: object, rgb: object) -> None:
        self.seen.append(fnum)

    def finish(self) -> None:
        pass


def test_stage_gates(tmp_path: Path) -> None:
    write_shots(tmp_path, [(0, 9), (10, 59)])
    plain, filled = Stage(tmp_path, 1.0), Stage(tmp_path, 1.0, backfill=True)
    gates = sampling.stage_gates(tmp_path, [plain, filled], 25.0)
    assert [f for f in range(60) if gates[0](f)] == [0, 25, 50]
    assert [f for f in range(60) if gates[1](f)] == [0, 25, 50]
    filled.every_shot = True
    gates = sampling.stage_gates(tmp_path, [filled], 25.0)
    assert [f for f in range(60) if gates[0](f)] == [0, 4, 25, 34, 50]


def test_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampling, "OVERLAP", True)
    s = Stage(tmp_path)
    s.overlap = True
    threads: list[str] = []

    def work(n: int) -> None:
        threads.append(threading.current_thread().name)
        s.seen.append(n)

    for n in range(3):
        s.submit(work, n)
    s.wait()
    assert s.seen == [0, 1, 2]
    assert all(t.startswith("test") for t in threads)

    def fail() -> None:
        raise RuntimeError("batch failed")

    s.submit(fail)
    with pytest.raises(RuntimeError, match="batch failed"):
        s.wait()
    s.close()
    s.overlap = False
    s.submit(work, 9)
    assert threads[-1] == threading.current_thread().name


def test_resolve_crop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = Stage(tmp_path)
    assert s.resolve_crop(1920, 1080) is None
    meta(1080, 1920, 140, 140).write_parquet(tmp_path / "metadata.parquet")
    assert s.resolve_crop(1920, 1080) == (140, 940, 0, 1920)
    meta(1080, 1920).write_parquet(tmp_path / "metadata.parquet")
    assert s.resolve_crop(1920, 1080, "dino2") is None
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("NO metadata.parquet — embedding the stored frame, bars included")
    assert (
        out[1] == f"[{tmp_path.name}:test] cropped to [140:940, 0:1920] of 1920x1080 "
        "— 25.9% of the frame was bar"
    )
    assert out[2] == f"[{tmp_path.name}:dino2] letterbox crop is empty — frame is all content"


@needs_ffmpeg
@pytest.mark.parametrize("prefetch", [False, True])
def test_decoded_frames(tmp_path: Path, prefetch: bool) -> None:
    v = make_video(tmp_path / "v.mp4", seconds=2)
    cap = cv2.VideoCapture(str(v))
    frames = list(sampling.decoded_frames(cap, 160, 120, prefetch=prefetch))
    assert len(frames) == 20
    assert frames[0].shape == (120, 160, 3)
    assert sampling.sample_stride(v) == 10


class Broken:
    """A capture that gives two frames and then fails."""

    def __init__(self) -> None:
        self.n = 0

    def read(self) -> tuple[bool, np.ndarray]:
        self.n += 1
        if self.n > 2:
            raise OSError("decoder died")
        return True, np.zeros((2, 2, 3), np.uint8)


def test_prefetch_passes_on_a_failure() -> None:
    frames: Iterator[np.ndarray] = sampling.decoded_frames(Broken(), 2, 2, prefetch=True)
    assert next(frames).shape == (2, 2, 3)
    assert next(frames).shape == (2, 2, 3)
    with pytest.raises(OSError, match="decoder died"):
        next(frames)


def test_sample_stride_of_junk(tmp_path: Path) -> None:
    junk = tmp_path / "v.mp4"
    junk.write_text("no")
    assert sampling.sample_stride(junk) == sampling.FALLBACK_STRIDE


# ---- the stages on a video --------------------------------------------------------


def source(tmp_path: Path) -> Source:
    v = make_video(tmp_path / "src" / "Film.mp4", seconds=6, bars=(16, 16, 0, 0))
    return Source.under(v, tmp_path / "out")


@needs_ffmpeg
def test_process_mp4(tmp_path: Path, models: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dvt.tools.audio import extract_audio

    s = source(tmp_path)
    extract_audio(s.video, s.audio)
    names = ["metadata", "letterbox", "frames", "shots", "frames_r720"]
    video.process_mp4(s, names)
    out = capsys.readouterr().out
    assert out.startswith("Working on Film.")
    assert out.rstrip().endswith("Finished Film.")

    frames = pl.read_parquet(s.out / "frames.parquet")
    assert frames.height == 60
    assert frames["frame"].to_list() == list(range(60))
    assert frames["t"].to_list() == pytest.approx([i / 10 for i in range(60)], abs=1e-6)
    assert frames.columns[:6] == ["frame", "t", "mean", "std", "sfp", "afp"]
    assert frames["sfp"][0] == 0.0
    assert frames["dx"][0] is None
    assert len(frames["hist16"][5]) == frame_ops.HIST_BINS
    info = json.loads(pl.read_parquet_metadata(s.out / "frames.parquet")["frames"])
    assert info["n_frames"] == 60
    assert info["crop_source"] == [16, 16, 0, 0]
    assert info["boxed"]

    sh = pl.read_parquet(s.out / "shots.parquet")
    assert sh["f0"][0] == 0
    assert sh["f1"][-1] == 59
    assert {"vai", "cam_dx", "hist_dist_in", "edge"} <= set(sh.columns)
    assert sh["hist_dist_in"][0] is None or np.isnan(sh["hist_dist_in"][0])

    saved = sorted(p.name for p in s.frames.iterdir())
    gate = {0, 10, 20, 30, 40, 50} | set(sh["fmid"].to_list())
    assert saved == [f"Film_{f:07d}.jpg" for f in sorted(gate)]
    first = cv2.imread(str(s.frames / saved[0]))
    assert first is not None
    assert first.shape == (152, 160, 3)
    assert not (s.frames.parent / "Film.part").exists()

    video.process_mp4(s, names)
    again = capsys.readouterr().out
    assert again.count("Already exists") == 5


@needs_ffmpeg
def test_frames_needs_metadata(tmp_path: Path, models: Path) -> None:
    s = source(tmp_path)
    s.out.mkdir(parents=True)
    with pytest.raises(ValueError, match=r"needs metadata\.parquet"):
        create_frames(s.video, s.out)
    pl.DataFrame({"n_frames": [1]}).write_parquet(s.out / "metadata.parquet")
    with pytest.raises(ValueError, match="predates letterbox"):
        create_frames(s.video, s.out)


def test_stages_wait_for_their_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert video.frame_stages_ready(tmp_path, ["frames_r720", "face", "dino2"]) == ["face"]
    out = capsys.readouterr().out
    assert f"[{tmp_path.name}:frames_r720] No shots.parquet — skipped" in out
    assert f"[{tmp_path.name}:dino2] No shots.parquet — skipped" in out


def test_check_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    class Needy(Stage):
        imports = ("no_such_module_here",)

    monkeypatch.setitem(video.FRAME_STAGES, "needy", Needy)
    video.check_imports(["frames_r720"])
    with pytest.raises(SystemExit, match="needy: No module named 'no_such_module_here'"):
        video.check_imports(["needy"])


def test_add_shot_metrics_wants_the_inputs() -> None:
    with pytest.raises(ValueError, match="predates the shot metrics"):
        shots.add_shot_metrics(pl.DataFrame({"sid": [0], "f0": [0]}), pl.DataFrame({"frame": [0]}))


def test_saved_frames_need_a_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="there is none"):
        FramesStage(tmp_path)


@needs_ffmpeg
def test_saved_frames_shrink(tmp_path: Path) -> None:
    s = Source(video=tmp_path / "v", out=tmp_path / "Film", audio=tmp_path / "a",
               frames=tmp_path / "frames" / "Film", stems="x")  # fmt: skip
    st_ = FramesStage(s.out, s)
    st_.start(1920, 1080)
    assert st_.size == (1280, 720)
    st_.handle(3, solid((1, 2, 3), 1080, 1920), None)
    st_.finish()
    img = cv2.imread(str(s.frames / "Film_0000003.jpg"))
    assert img is not None
    assert img.shape == (720, 1280, 3)
