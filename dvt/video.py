"""Runs the video annotators over one file and writes their parquets."""

import importlib
import sys
import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING

# The settings every annotator runs under, before anything that loads a model.
# isort: off
from . import runtime as runtime

# isort: on
import cv2
from tqdm import tqdm

from . import chroma, container, letterbox, tracks
from .annotators import ANNOTATORS
from .embeddings import DinoStage, SiglipBaseStage, SiglipStage
from .face import FaceStage
from .frames import create_frames
from .log import log_finished, log_status
from .metadata import create_meta
from .sampling import FrameStage, decoded_frames, sample_grid, stage_gates
from .saved_frames import FramesStage
from .shots import create_shots
from .tools.values import StrPath
from .yolo import YoloStage

if TYPE_CHECKING:
    from .source import Source


FRAME_STAGES: dict[str, type[FrameStage]] = {
    "frames_r720": FramesStage,
    "dino2": DinoStage,
    "siglip2": SiglipStage,
    "siglip2_base": SiglipBaseStage,
    "face": FaceStage,
    "yolo": YoloStage,
}


def check_imports(annotators: Collection[str]) -> None:
    """Import every module an enabled frame stage loads lazily, and exit if
    any is missing. A run over many films catches each one's exception and moves on, so
    without this a package absent from the environment fails every film that
    needs it, one at a time, for as long as the array runs."""
    missing: list[str] = []
    for name in annotators:
        stage = FRAME_STAGES.get(name)
        for module in stage.imports if stage is not None else ():
            try:
                importlib.import_module(module)
            except ImportError as exc:
                missing.append(f"    {name}: {exc}")
    if missing:
        raise SystemExit(
            f"enabled stages cannot import what they need in {sys.prefix}:\n" + "\n".join(missing)
        )


def run_frame_sampled(
    ip: StrPath, od: Path, names: Collection[str], source: "Source | None" = None
) -> None:
    stages = [FRAME_STAGES[n](od, source) for n in FRAME_STAGES if n in names]

    pending: list[FrameStage] = []
    for s in stages:
        if s.done():
            log_status(od, s.label, "Already exists")
        else:
            pending.append(s)
    if not pending:
        return

    t0 = time.monotonic()

    vinput = cv2.VideoCapture(str(ip))
    total_frames = int(vinput.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vinput.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vinput.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vinput.get(cv2.CAP_PROP_FPS)

    for s in pending:
        s.ip = ip
    gates = stage_gates(od, pending, fps)

    for s in pending:
        s.start(width, height)

    def described(s: FrameStage) -> str:
        base, offsets = sample_grid(fps, s.sample_hz)
        extra = f" +{len(s.backfill)} shot" if s.backfill else ""
        return f"{s.label} ({len(offsets)}/{base}f{extra})"

    log_status(od, "frame-sampled", "one decode for " + ", ".join(described(s) for s in pending))

    fnum = 0
    try:
        with tqdm(total=total_frames, desc="Processing frames", disable=None) as pbar:
            for img in decoded_frames(vinput, width, height):
                due = [s for s, g in zip(pending, gates, strict=True) if g(fnum)]
                if due:
                    rgb = (
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if any(s.needs_rgb for s in due)
                        else None
                    )
                    for s in due:
                        s.handle(fnum, img, rgb)

                fnum += 1
                pbar.update(1)

        vinput.release()

        for s in pending:
            s.finish()
    finally:
        for s in pending:
            s.close()
    log_finished(od, "frame-sampled", t0)


# metadata is not here: it alone reads the source's sound and container.
STAGES: dict[str, Callable[[Path, Path], None]] = {
    "letterbox": letterbox.create_letterbox,
    "chroma": chroma.create_chroma,
    "chapters": container.create_chapters,
    "subtitles": container.create_subtitles,
    "frames": create_frames,
    "shots": create_shots,
    "tracks": tracks.create_tracks,
}


def frame_stages_ready(od: Path, annotators: Collection[str] | None) -> list[str]:
    """The enabled frame stages whose inputs are on disk now. Asked at the
    point in the annotator order where the frame stages run, after the
    sequential stages before them, so a stage that wants shots.parquet
    gets it from the same pass that made it."""
    ready: list[str] = []
    for name in FRAME_STAGES:
        if annotators is not None and name not in annotators:
            continue
        if FRAME_STAGES[name].wants_crop and not (od / "metadata.parquet").exists():
            log_status(od, name, "No metadata.parquet — skipped")
            continue
        if FRAME_STAGES[name].backfill_shots and not (od / "shots.parquet").exists():
            log_status(od, name, "No shots.parquet — skipped")
            continue
        ready.append(name)
    return ready


def process_mp4(source: "Source", annotators: Collection[str] | None = None) -> None:
    ip, od = source.video, source.out
    od.mkdir(parents=True, exist_ok=True)
    print(f"Working on {od.name}.")

    frames_run = False
    for name in ANNOTATORS:
        if ANNOTATORS[name].source != "mp4":
            continue

        if name in FRAME_STAGES:
            if not frames_run:
                frames_run = True
                ready = frame_stages_ready(od, annotators)
                if ready:
                    run_frame_sampled(ip, od, ready, source)
            continue

        if annotators is not None and name not in annotators:
            continue
        if name == "metadata":
            create_meta(source)
        else:
            STAGES[name](ip, od)

    print(f"Finished {od.name}.")
