"""Runs the annotators that read a picture over separate images.

A video's tables name a frame by its number and time. An image has neither,
so each table here names the image by its path, in a `path` column: one row
an image for an embedding, one row a face or a head for the detectors, and
every image of a directory in the one table. images.parquet lists the
images with their sizes, and says which could not be read.
"""

import shutil
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path

# The settings every annotator runs under, before anything that loads a model.
# isort: off
from . import runtime as runtime

# isort: on
import cv2
import polars as pl

from .annotators import ANNOTATORS, IMAGE_ANNOTATORS
from .sampling import FrameStage
from .tools.values import StrPath

NAMES = IMAGE_ANNOTATORS

SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

# In a video's table and not an image's: the frame number, and whether an
# embedded frame came from the sampling grid or from a shot.
FRAME, FROM = "frame", "src"
PATH = "path"

IMAGES = "images.parquet"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUFFIXES


def listed(path: Path) -> tuple[Path, list[Path]]:
    """The directory paths are stated from and the images under it, in
    order: a directory's images, however deep, or the one image."""
    if path.is_dir():
        return path, sorted(p for p in path.rglob("*") if p.is_file() and is_image(p))
    return path.parent, [path]


def stages_for(names: Collection[str], od: Path) -> list[tuple[str, FrameStage]]:
    from .video import FRAME_STAGES, check_imports

    check_imports(names)
    made: list[tuple[str, FrameStage]] = []
    for name in NAMES:
        if name in names:
            stage = FRAME_STAGES[name](od, None)
            stage.still = True
            made.append((name, stage))
    return made


def run_stages(stages: Sequence[FrameStage], paths: Sequence[Path]) -> list[tuple[int, int] | None]:
    """Hands every image to every stage, numbered as it is listed, and
    returns the width and height of each, None for one that cannot be read.
    The stages start at the first image's size and are told when it changes."""
    sizes: list[tuple[int, int] | None] = []
    current: tuple[int, int] | None = None
    wants_rgb = any(s.needs_rgb for s in stages)
    try:
        for number, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                sizes.append(None)
                continue
            height, width = image.shape[:2]
            if current is None:
                for s in stages:
                    s.start(width, height)
            elif current != (width, height):
                for s in stages:
                    s.flush()
                    s.wait()
                    s.resize(width, height)
            current = (width, height)
            sizes.append(current)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if wants_rgb else None
            for s in stages:
                s.handle(number, image, rgb if s.needs_rgb else None)
        for s in stages:
            s.finish()
    finally:
        for s in stages:
            s.close()
    return sizes


def by_path(table: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame:
    """A stage's table with the image's path where its number was."""
    paths = pl.DataFrame(
        {FRAME: range(len(names)), PATH: names},
        schema={FRAME: table.schema[FRAME], PATH: pl.String},
    )
    rest = [c for c in table.columns if c not in (FRAME, FROM)]
    return table.join(paths, on=FRAME, how="left", maintain_order="left").select(PATH, *rest)


def annotate(path: StrPath, out: StrPath, names: Collection[str] | None = None) -> Path:
    """Runs the named annotators, all that read an image when none are
    named, over an image or a directory of images. The tables go under
    `out` in annotations/<name>/, where the name is the directory's or the
    image's without its suffix; that directory is returned."""
    path, out = Path(path), Path(out)
    names = list(NAMES) if names is None else [n for n in NAMES if n in names]
    root, paths = listed(path)
    od = out / "annotations" / (path.name if path.is_dir() else path.stem)
    od.mkdir(parents=True, exist_ok=True)
    stated = [p.relative_to(root).as_posix() for p in paths]

    # The stages write the tables a video has, by frame number, into a
    # directory of their own; the tables by path are made from those. It
    # has the name of the one the tables go to, which is what a stage
    # calls its work in what it prints.
    scratch = Path(tempfile.mkdtemp(prefix="dvt-", dir=od))
    work = scratch / od.name
    work.mkdir()
    try:
        stages = stages_for(names, work)
        sizes = run_stages([s for _, s in stages], paths)
        for name, _ in stages:
            for output in ANNOTATORS[name].outputs:
                by_path(pl.read_parquet(work / output), stated).write_parquet(od / output)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    pl.DataFrame(
        {
            PATH: stated,
            "width": [None if s is None else s[0] for s in sizes],
            "height": [None if s is None else s[1] for s in sizes],
        },
        schema={PATH: pl.String, "width": pl.Int32, "height": pl.Int32},
    ).write_parquet(od / IMAGES)
    return od
