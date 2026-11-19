"""The Distant Viewing Toolkit: annotators that each write a parquet table about one video or image.

    import dvt
    tables = dvt.annotate("film.mp4", out="out")

Nothing here knows where a corpus is kept. A Source says where one file is
and where its outputs go, config names the model directory, and run.annotate
does the work; annotate, below, is those three for one file laid out under
one directory. Importing the package loads no model and none of the
libraries that run one: dvt.tools is used on machines that have neither.
"""

from collections.abc import Collection
from typing import TYPE_CHECKING

from .tools.values import StrPath

if TYPE_CHECKING:
    import polars as pl


def annotate(
    path: StrPath,
    out: StrPath | None = None,
    annotators: Collection[str] | None = None,
    models: StrPath | None = None,
) -> "dict[str, pl.DataFrame]":
    """Annotates a video, an image or a directory of images and returns its
    tables by name.

    Everything made is kept under `out`: the tables in annotations/<name>/,
    and for a video its sound, saved frames and separated stems beside
    them. With no `out` it is made in a temporary directory and only the
    tables come back. `annotators` names the ones to run; all run when none
    are named, which for an image is the ones that read a single picture.
    `models` is the model directory, when config.configure has not named it.
    """
    from .api import annotate as annotate_one

    return annotate_one(path, out, annotators, models)
