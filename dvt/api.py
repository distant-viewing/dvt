"""annotate: one video, image or directory of images, laid out under one directory."""

import tempfile
from collections.abc import Collection
from pathlib import Path

import polars as pl

from . import config, fetch
from .annotators import ANNOTATORS
from .tools.values import StrPath


def annotate(
    path: StrPath,
    out: StrPath | None = None,
    annotators: Collection[str] | None = None,
    models: StrPath | None = None,
) -> dict[str, pl.DataFrame]:
    """What dvt.annotate does; its docstring says what."""
    from . import images

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    still = path.is_dir() or images.is_image(path)
    known = images.NAMES if still else tuple(ANNOTATORS)
    names = list(known) if annotators is None else list(annotators)
    unknown = [n for n in names if n not in known]
    if unknown:
        kind = "an image" if still else "a video"
        raise ValueError(f"not annotators of {kind}: {', '.join(unknown)}")
    if models is not None:
        config.configure(models)
    absent = fetch.missing(names)
    if absent:
        raise RuntimeError(
            f"no model for {', '.join(absent)} in {config.models()}; "
            "`dvt models fetch` downloads them"
        )

    def made(root: Path) -> dict[str, pl.DataFrame]:
        if still:
            od = images.annotate(path, root, names)
        else:
            from . import run
            from .source import Source

            source = Source.under(path, root)
            run.load(names)
            said = run.extract(source)
            if said is not None:
                print(f"[{source.name}:audio] {said}")
            run.annotate(source, names)
            od = source.out
        return {p.stem: pl.read_parquet(p) for p in sorted(od.glob("*.parquet"))}

    if out is not None:
        return made(Path(out))
    with tempfile.TemporaryDirectory(prefix="dvt-") as scratch:
        return made(Path(scratch))
