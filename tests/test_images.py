"""Separate images: every image to every stage, the tables by path, and the
stages told when the size changes. The stages here load no model."""

from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest
import torch
from cv2.typing import MatLike

import dvt
from dvt import cli, embeddings, fetch, images, video
from dvt.sampling import FrameStage


class Detector(FrameStage):
    """Finds as many things in an image as its first pixel says, in pairs,
    the way a detector batches: nothing is written down until a flush."""

    label = "detector"
    said: list[str]

    def __init__(self, od: Path, source: object = None) -> None:
        super().__init__(od)
        self.said = []
        self.held: list[tuple[int, int]] = []
        self.rows: list[tuple[int, int, int]] = []
        self.width = 0

    def start(self, width: int, height: int) -> None:
        self.said.append(f"start {width}x{height}")
        self.width = width

    def resize(self, width: int, height: int) -> None:
        assert not self.held, "resized with frames still held at the old size"
        self.said.append(f"resize {width}x{height}")
        self.width = width

    def handle(self, fnum: int, bgr: MatLike, rgb: MatLike | None) -> None:
        assert rgb is None
        assert bgr.shape[1] == self.width
        self.held.append((fnum, int(bgr[0, 0, 0])))
        if len(self.held) >= 2:
            self.flush()

    def flush(self) -> None:
        self.rows += [(f, i, self.width) for f, n in self.held for i in range(n)]
        self.held = []

    def finish(self) -> None:
        self.flush()
        schema = {"frame": pl.Int64, "face_id": pl.Int64, "seen_at": pl.Int64}
        pl.DataFrame(self.rows, schema=schema, orient="row").write_parquet(self.od / "face.parquet")


class Processor:
    def __call__(self, images: MatLike, return_tensors: str) -> dict[str, torch.Tensor]:
        return {"pixel_values": torch.as_tensor(images.astype(np.float32).mean(axis=(0, 1)))[None]}


class Embedder(embeddings.EmbeddingStage):
    label = "embedder"
    parquet = "dino2.parquet"
    name = "dino2"
    needs_rgb = True
    wants_crop = True
    BATCH = 2

    def load_model(self, width: int, height: int) -> None:
        self.processor = Processor()

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(video.FRAME_STAGES, "face", Detector)
    monkeypatch.setitem(video.FRAME_STAGES, "dino2", Embedder)

    def none(names: object = None) -> list[str]:
        return []

    monkeypatch.setattr(fetch, "missing", none)


def picture(path: Path, width: int, height: int, blue: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0] = blue
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def album(tmp_path: Path) -> Path:
    root = tmp_path / "album"
    picture(root / "a.png", 8, 6, 2)
    picture(root / "b.png", 8, 6, 0)
    picture(root / "sub" / "c.png", 4, 10, 1)
    picture(root / "sub" / "d.PNG", 8, 6, 3)
    (root / "broken.jpg").write_bytes(b"not a picture")
    (root / "notes.txt").write_text("not an image")
    return root


def test_listed(album: Path) -> None:
    root, paths = images.listed(album)
    assert root == album
    assert [p.relative_to(album).as_posix() for p in paths] == [
        "a.png",
        "b.png",
        "broken.jpg",
        "sub/c.png",
        "sub/d.PNG",
    ]
    assert images.listed(album / "a.png") == (album, [album / "a.png"])


def test_a_directory(album: Path, tmp_path: Path, stages: None) -> None:
    od = images.annotate(album, tmp_path / "out", ["face", "dino2", "shots"])
    assert od == tmp_path / "out" / "annotations" / "album"
    assert sorted(p.name for p in od.iterdir()) == [
        "dino2.parquet",
        "face.parquet",
        "images.parquet",
    ]

    listing = pl.read_parquet(od / "images.parquet")
    assert listing.rows() == [
        ("a.png", 8, 6),
        ("b.png", 8, 6),
        ("broken.jpg", None, None),
        ("sub/c.png", 4, 10),
        ("sub/d.PNG", 8, 6),
    ]

    # One row a thing found, by path, and each image seen at its own width.
    face = pl.read_parquet(od / "face.parquet")
    assert face.columns == ["path", "face_id", "seen_at"]
    assert face.rows() == [
        ("a.png", 0, 8),
        ("a.png", 1, 8),
        ("sub/c.png", 0, 4),
        ("sub/d.PNG", 0, 8),
        ("sub/d.PNG", 1, 8),
        ("sub/d.PNG", 2, 8),
    ]

    # One row an image that could be read; the video's own columns are gone.
    dino = pl.read_parquet(od / "dino2.parquet")
    assert dino.columns == ["path", "embedding", "version"]
    assert dino["path"].to_list() == ["a.png", "b.png", "sub/c.png", "sub/d.PNG"]
    blue = [row[2] for row in dino["embedding"].to_list()]
    assert blue == pytest.approx([1.0, 0.0, 1.0, 1.0])


def test_the_stages_are_told_of_a_new_size(album: Path, tmp_path: Path) -> None:
    stage = Detector(tmp_path)
    sizes = images.run_stages([stage], images.listed(album)[1])
    assert sizes == [(8, 6), (8, 6), None, (4, 10), (8, 6)]
    assert stage.said == ["start 8x6", "resize 4x10", "resize 8x6"]


def test_one_image_and_the_python_call(album: Path, tmp_path: Path, stages: None) -> None:
    tables = dvt.annotate(album / "sub" / "c.png", tmp_path / "out", ["face"])
    assert sorted(tables) == ["face", "images"]
    assert tables["face"].rows() == [("c.png", 0, 4)]
    assert (tmp_path / "out" / "annotations" / "c" / "face.parquet").exists()

    # With nowhere to keep them, only the tables come back.
    tables = dvt.annotate(album, annotators=["dino2"])
    assert sorted(tables) == ["dino2", "images"]
    assert tables["dino2"].height == 4

    with pytest.raises(ValueError, match="not annotators of an image: shots"):
        dvt.annotate(album, annotators=["face", "shots"])
    with pytest.raises(FileNotFoundError):
        dvt.annotate(tmp_path / "nothing.png")


def test_the_command_line(album: Path, tmp_path: Path, stages: None) -> None:
    argv = [str(album), "--out", str(tmp_path / "o"), "--models", str(tmp_path / "m")]
    assert cli.main([*argv, "--annotators", "face"]) == 0
    assert pl.read_parquet(tmp_path / "o" / "annotations" / "album" / "face.parquet").height == 6


def test_no_images(tmp_path: Path, stages: None) -> None:
    (tmp_path / "empty").mkdir()
    tables = dvt.annotate(tmp_path / "empty", tmp_path / "out", ["face"])
    assert tables["images"].height == 0
    assert tables["face"].columns == ["path", "face_id", "seen_at"]
    assert tables["face"].height == 0
