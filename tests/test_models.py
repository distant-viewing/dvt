"""The stages that run a model, with the model stood in for: tracks and
ByteTrack on made-up detections, TransNetV2 with seeded weights, and the
embedding and YOLO stages with a fake model, so what surrounds the model is
what is tested."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scene import write_scene

from dvt import clock, embeddings, face, tracks, yolo
from dvt.bytetrack import BaseTrack, BYTETracker, STrack, TrackerArgs, matching
from dvt.bytetrack.kalman_filter import KalmanFilter
from dvt.transnet import TransNetV2

# ---- tracks ---------------------------------------------------------------------


def test_create_tracks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    od = tmp_path / "Film"
    write_scene(od)
    tracks.create_tracks(None, od)
    t = pl.read_parquet(od / "tracks.parquet")
    f = pl.read_parquet(od / "tracks_frames.parquet")
    assert t.drop("version").schema == pl.Schema(tracks.TRACKS_SCHEMA)
    assert f.drop("version").schema == pl.Schema(tracks.FRAMES_SCHEMA)
    assert t["track_id"].to_list() == list(range(1, t.height + 1))
    assert t["sid"].to_list() == sorted(t["sid"].to_list())
    assert (t["first_frame"] <= t["last_frame"]).all()
    assert t["t0"].to_list() == pytest.approx([x / 24 for x in t["first_frame"]])
    assert t["embedding"].is_not_null().all()
    norms = np.linalg.norm(np.stack(list(t["embedding"].to_numpy())), axis=1)
    assert norms == pytest.approx(np.ones(t.height), abs=1e-5)
    assert f.filter(pl.col("track_id").is_null())["face_id"].is_not_null().all()
    meta = json.loads(pl.read_parquet_metadata(od / "tracks.parquet")["tracks"])
    assert (meta["fps"], meta["img_width"], meta["img_height"]) == (24.0, 640, 360)
    out = capsys.readouterr().out
    assert "[Film:tracks] 2 shots," in out
    tracks.create_tracks(None, od)
    assert capsys.readouterr().out == "[Film:tracks] Already exists\n"


def test_tracks_wait_for_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tracks.create_tracks(None, tmp_path)
    assert "No shots.parquet — skipped" in capsys.readouterr().out


def test_no_detections(tmp_path: Path) -> None:
    od = tmp_path / "Film"
    write_scene(od, people=0)
    tracks.create_tracks(None, od)
    assert pl.read_parquet(od / "tracks.parquet").height == 0


def test_video_props(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no video and no metadata"):
        tracks.video_props(None, tmp_path)
    pl.DataFrame({"fps": [0.0], "img_width": [640], "img_height": [360]}).write_parquet(
        tmp_path / "metadata.parquet")  # fmt: skip
    assert tracks.video_props(None, tmp_path) == (tracks.FALLBACK_FPS, 640, 360)


def test_face_yaw() -> None:
    frontal = np.array([10, 0, 30, 0, 20, 10, 12, 20, 28, 20], dtype=float)
    turned = np.array([10, 0, 30, 0, 28, 10, 12, 20, 28, 20], dtype=float)
    assert tracks.face_yaw(np.stack([frontal, turned])).tolist() == pytest.approx([0.0, 0.4])


def test_as_matrix() -> None:
    s = pl.Series([[1.0, 2.0], [3.0, 4.0]], dtype=pl.Array(pl.Float32, 2))
    assert tracks.as_matrix(s, 2).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert tracks.as_matrix(pl.Series([], dtype=pl.Array(pl.Float32, 2)), 2).shape == (0, 2)


def test_farthest_point_exemplars() -> None:
    x = np.eye(4)
    # Only faces of at least the median quality are candidates.
    assert tracks.farthest_point_exemplars(x, np.array([1.0, 0.9, 0.8, 0.1]), 3) == [0, 1]
    assert tracks.farthest_point_exemplars(np.zeros((0, 4)), np.zeros(0), 3) == []


def test_describe() -> None:
    stats = {"n_faces": 4, "n_faces_attached": 3, "n_tracks": 2, "n_shots_tracked": 1,
             "n_shots": 2, "n_dup": 0, "n_split": 1, "n_stitch": 0,
             "n_with_embedding": 2}  # fmt: skip
    assert tracks.describe(stats) == (
        "2 tracks in 1/2 shots; 0 duplicate merges, 1 splits, 0 stitches; "
        "3/4 faces on a track (75.0%); 2 tracks with an embedding")  # fmt: skip


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 4))
def test_every_face_lands_once(seed: int, people: int) -> None:
    """Every face is in tracks_frames once, on a track or as an orphan, and
    every track lies inside its shot."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        od = Path(d) / "Film"
        write_scene(od, seed=seed, people=people)
        tracks.create_tracks(None, od)
        faces = pl.read_parquet(od / "face.parquet")
        f = pl.read_parquet(od / "tracks_frames.parquet").filter(pl.col("face_id").is_not_null())
        t = pl.read_parquet(od / "tracks.parquet")
    assert f.height == faces.height
    assert f.select("frame", "face_id").is_duplicated().sum() == 0
    within = t.join(pl.DataFrame({"sid": [0, 1], "f0": [0, 60], "f1": [59, 119]}), on="sid")
    assert ((within["first_frame"] >= within["f0"]) & (within["last_frame"] <= within["f1"])).all()


# ---- ByteTrack ------------------------------------------------------------------


def args() -> TrackerArgs:
    return TrackerArgs(track_thresh=0.4, track_buffer=12, match_thresh=0.8)


def test_bytetrack_follows_two_boxes() -> None:
    BaseTrack.reset_ids()
    tracker = BYTETracker(args())
    seen: dict[int, list[int]] = {}
    for f in range(20):
        boxes = np.array([[10 + 2 * f, 10, 60 + 2 * f, 80, 0.9],
                          [300 - 3 * f, 50, 360 - 3 * f, 130, 0.8]])  # fmt: skip
        for t in tracker.update(boxes, (360, 640), (360, 640)):
            seen.setdefault(int(t.track_id), []).append(f)
    assert sorted(seen) == [1, 2]
    # A track is reported from its first frame, which is the tracker's first.
    assert all(v == list(range(20)) for v in seen.values())


def test_bytetrack_low_scores_keep_a_track() -> None:
    BaseTrack.reset_ids()
    tracker = BYTETracker(args())
    ids: list[list[int]] = []
    for f in range(12):
        score = 0.9 if f < 4 else 0.3
        out = tracker.update(
            np.array([[100.0 + f, 100, 150 + f, 160, score]]), (360, 640), (360, 640)
        )
        ids.append([int(t.track_id) for t in out])
    assert ids[0] == [1]
    assert all(i == [1] for i in ids[1:])


def test_strack_boxes() -> None:
    s = STrack(np.array([10.0, 20.0, 30.0, 40.0]), 0.9)
    assert s.tlbr.tolist() == [10.0, 20.0, 40.0, 60.0]
    assert STrack.tlwh_to_xyah(np.array([10.0, 20.0, 30.0, 40.0])).tolist() == [
        25.0,
        40.0,
        0.75,
        40.0,
    ]
    with pytest.raises(AttributeError, match="until it is activated"):
        s.predict()
    s.activate(KalmanFilter(), 1)
    assert s.is_activated
    assert s.tlwh.tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])
    s.update_features(np.array([3.0, 4.0]))
    assert s.smooth_feat is not None
    assert s.smooth_feat.tolist() == pytest.approx([0.6, 0.8])


def test_linear_assignment() -> None:
    cost = np.array([[0.1, 0.9], [0.8, 0.2], [0.95, 0.99]])
    m, ua, ub = matching.linear_assignment(cost, 0.5)
    assert m.tolist() == [[0, 0], [1, 1]]
    assert list(ua) == [2]
    assert list(ub) == []
    m, ua, ub = matching.linear_assignment(np.zeros((0, 3)), 0.5)
    assert (m.shape, tuple(ua), tuple(ub)) == ((0, 2), (), (0, 1, 2))


def test_iou_distance() -> None:
    a = [np.array([0.0, 0.0, 9.0, 9.0])]
    b = [np.array([0.0, 0.0, 9.0, 9.0]), np.array([20.0, 20.0, 29.0, 29.0])]
    assert matching.iou_distance(a, b).tolist() == [[0.0, 1.0]]


def test_kalman_filter() -> None:
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([50.0, 50.0, 0.5, 100.0]))
    mean2, cov2 = kf.predict(mean, cov)
    assert mean2[:4].tolist() == pytest.approx([50.0, 50.0, 0.5, 100.0])
    assert np.trace(cov2) > np.trace(cov)
    mean3, _ = kf.update(mean2, cov2, np.array([52.0, 50.0, 0.5, 100.0]))
    assert 50.0 < mean3[0] < 52.0
    means, covs = kf.multi_predict(np.stack([mean, mean]), np.stack([cov, cov]))
    assert means.shape == (2, 8)
    assert covs.shape == (2, 8, 8)
    d = kf.gating_distance(
        mean, cov, np.array([[50.0, 50.0, 0.5, 100.0], [90.0, 50.0, 0.5, 100.0]])
    )
    assert d[0] < d[1]


# ---- TransNetV2 -----------------------------------------------------------------


def test_transnet(model_dir: Path) -> None:
    net = TransNetV2()
    net.load_state_dict(torch.load(model_dir / "detect_shots.pt"))
    net.eval()
    frames = torch.as_tensor(np.random.default_rng(0).integers(0, 255, (130, 27, 48, 3)),
                             dtype=torch.uint8)  # fmt: skip
    single, every = net.predict_frames(frames)
    assert single.shape == every.shape == (130,)
    assert ((single >= 0) & (single <= 1)).all()
    again, _ = net.predict_frames(frames.numpy())
    assert np.array_equal(single, again)
    with pytest.raises(AssertionError, match="Input shape"):
        net.predict_frames(torch.zeros(4, 27, 48))


# ---- embedding and YOLO stages with a fake model -------------------------------------


class FakeProcessor:
    def __call__(
        self, images: np.ndarray, return_tensors: str, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        mean = torch.as_tensor(images.astype(np.float32).mean(axis=(0, 1)))
        return {"pixel_values": mean[None, :]}


class FakeModel(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.cat([pixel_values, pixel_values * 2], dim=1)


class Stage(embeddings.EmbeddingStage):
    label = "fake"
    parquet = "fake.parquet"
    name = "chroma"
    BATCH = 2

    def load_model(self, width: int, height: int) -> None:
        self.processor = FakeProcessor()
        self.model = FakeModel()

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


def test_embedding_stage(tmp_path: Path) -> None:
    pl.DataFrame({"f0": [0, 2], "f1": [1, 4], "fmid": [0, 3]}).write_parquet(
        tmp_path / "shots.parquet"
    )
    s = Stage(tmp_path)
    s.backfill = frozenset({3})
    s.start(4, 4)
    for f in range(5):
        img = np.full((4, 4, 3), f * 10, dtype=np.uint8)
        s.handle(f, img, img)
    s.finish()
    df = pl.read_parquet(tmp_path / "fake.parquet")
    assert df["frame"].to_list() == list(range(5))
    assert df["src"].to_list() == ["grid", "grid", "grid", "shot", "grid"]
    e = np.stack(list(df["embedding"].to_numpy()))
    assert e.shape == (5, 6)
    assert np.linalg.norm(e[1:], axis=1) == pytest.approx(np.ones(4))


def test_write_no_embeddings(tmp_path: Path) -> None:
    embeddings.write_embeddings(tmp_path / "e.parquet", [], [], [], "dino2")
    assert pl.read_parquet(tmp_path / "e.parquet").columns == ["frame", "src", "version"]


class FakeBoxes:
    def __init__(self, n: int) -> None:
        self.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]] * n)
        self.conf = torch.tensor([0.5] * n)

    def __len__(self) -> int:
        return len(self.conf)


class FakeYOLO:
    def __init__(self, path: str) -> None:
        self.path = path

    def predict(
        self, imgs: list[np.ndarray] | np.ndarray, **kwargs: object
    ) -> list[types.SimpleNamespace]:
        n = len(imgs) if isinstance(imgs, list) else 1
        return [types.SimpleNamespace(boxes=FakeBoxes(i % 3) if i % 4 else None) for i in range(n)]


def test_yolo_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: Path) -> None:
    fake = types.ModuleType("ultralytics")
    setattr(fake, "YOLO", FakeYOLO)  # noqa: B010
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(yolo.YoloStage, "BATCH", 3)
    s = yolo.YoloStage(tmp_path)
    s.start(1920, 1080)
    assert s.scale == pytest.approx(0.5)
    for f in range(8):
        s.handle(f, np.zeros((1080, 1920, 3), np.uint8), None)
    s.finish()
    s.close()
    df = pl.read_parquet(tmp_path / "yolo.parquet")
    assert df.drop("version").schema == pl.Schema(yolo.YoloStage.SCHEMA)
    # Batches of three; in each, the first result has no boxes and the
    # second and third one and two.
    assert df["frame"].to_list() == [1, 2, 2, 4, 5, 5, 7]
    assert df.row(0, named=True)["top_left_x"] == pytest.approx(20.0)


def test_clock_for_tracks() -> None:
    assert float(clock.Clock(None, 24.0, clock.FROM_FPS).at(48)) == 2.0


# ---- the face stage with a fake SCRFD and ArcFace --------------------------------------

STRIDES = (8, 16, 32)
ANCHORS = 2
# Five landmarks around a centre, in strides: the eyes, the nose, the mouth.
LANDMARKS = [-1.0, -1.0, 1.0, -1.0, 0.0, 0.0, -0.8, 1.0, 0.8, 1.0]


class FakeDetector(torch.nn.Module):
    """SCRFD's nine outputs, laid out as det_10g lays them out for a batch:
    the scores, box distances and landmarks of each stride, with a row for
    each cell, image and anchor. Every image has a face at stride 8 and a
    weaker one beside it that overlaps it; even images have a stronger one
    at stride 32."""

    def forward(self, blob: torch.Tensor) -> list[torch.Tensor]:
        n = blob.shape[0]
        scores: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []
        kps: list[torch.Tensor] = []
        for stride in STRIDES:
            w = 640 // stride
            rows = (640 // stride) * w * n * ANCHORS
            sc, bx, kp = torch.zeros(rows, 1), torch.zeros(rows, 4), torch.zeros(rows, 10)
            for i in range(n):
                hits: list[tuple[int, int, int, float]] = []
                if stride == 8:
                    hits = [(20, 30, 0, 0.6 + 0.01 * i), (20, 31, 1, 0.55)]
                elif stride == 32 and i % 2 == 0:
                    hits = [(5, 5, 0, 0.9)]
                for row, col, anchor, score in hits:
                    at = ((row * w + col) * n + i) * ANCHORS + anchor
                    sc[at, 0] = score
                    bx[at] = 2.5
                    kp[at] = torch.tensor(LANDMARKS)
            scores.append(sc)
            boxes.append(bx)
            kps.append(kp)
        return scores + boxes + kps


class FakeRecognizer(torch.nn.Module):
    def forward(self, blob: torch.Tensor) -> torch.Tensor:
        return blob.mean(dim=(2, 3)).repeat(1, 171)[:, :512]


def fake_convert(path: str) -> torch.nn.Module:
    return FakeDetector() if path.endswith("det_10g.onnx") else FakeRecognizer()


def test_face_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: Path) -> None:
    fake = types.ModuleType("onnx2torch")
    setattr(fake, "convert", fake_convert)  # noqa: B010
    monkeypatch.setitem(sys.modules, "onnx2torch", fake)
    monkeypatch.setattr(face.FaceStage, "BATCH", 3)
    monkeypatch.setattr(face.FaceStage, "REC_BATCH", 2)
    s = face.FaceStage(tmp_path)
    s.start(320, 240)
    assert (s.new_w, s.new_h, s.det_scale) == (640, 480, 2.0)
    rng = np.random.default_rng(0)
    for f in range(5):
        s.handle(f, rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), None)
    s.finish()
    s.close()
    assert s.done()
    df = pl.read_parquet(tmp_path / "face.parquet")
    assert df.drop("version").schema == pl.Schema(face.FaceStage.SCHEMA)
    # Batches of three and two; the image's place in its batch decides
    # what the fake finds.
    assert df["frame"].to_list() == [0, 0, 1, 2, 2, 3, 3, 4]
    assert df["face_id"].to_list() == [0, 1, 0, 0, 1, 0, 1, 0]
    assert df["det_score"].to_list() == pytest.approx(
        [0.9, 0.6, 0.61, 0.9, 0.62, 0.9, 0.6, 0.61]
    )  # fmt: skip
    big = df.filter(pl.col("det_score") > 0.8).select("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    assert big.rows() == [(40.0, 40.0, 120.0, 120.0)] * 3
    small = df.filter(pl.col("det_score") < 0.8).select("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    assert set(small.rows()) == {(110.0, 70.0, 130.0, 90.0)}
    assert df["kps"][1].to_list() == pytest.approx(
        [116, 76, 124, 76, 120, 80, 116.8, 84, 123.2, 84]
    )
    assert df.schema["embedding"] == pl.Array(pl.Float16, 512)
    e = np.stack(list(df["embedding"].to_numpy()))
    assert e.shape == (8, 512)
    assert np.isfinite(e).all()
    assert not np.allclose(e[0], e[1])


def test_face_stage_no_faces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: Path) -> None:
    fake = types.ModuleType("onnx2torch")
    setattr(fake, "convert", fake_convert)  # noqa: B010
    monkeypatch.setitem(sys.modules, "onnx2torch", fake)
    s = face.FaceStage(tmp_path)
    s.start(1920, 800)
    assert (s.new_w, s.new_h) == (640, 266)
    s.finish()
    s.close()
    df = pl.read_parquet(tmp_path / "face.parquet")
    assert df.height == 0
    assert df.drop("version").schema == pl.Schema(face.FaceStage.SCHEMA)
