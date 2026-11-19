"""The frames_r720 stage: the frames kept on disk for the stages that read pictures."""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from .annotators import FRAMES_SIZE
from .log import log_status
from .sampling import FrameStage

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from .source import Source


class FramesStage(FrameStage):
    """Saves the frames the later stages read from disk instead of decoding
    the mp4 again: one a second and the middle frame of every shot, which
    covers every frame the embedding stages sample. Each is the whole
    frame, bars included, resized so the short edge is at most SHORT_EDGE
    and never enlarged, as <fname>_<frame>.jpg under the source's frames. They
    are written into a .part directory that is renamed when the last one
    is in, so a directory that exists is complete and a killed task leaves
    nothing that looks finished."""

    label = f"Frames {FRAMES_SIZE}"
    sample_hz = 1.0
    backfill_shots = True
    every_shot = True

    SHORT_EDGE = 720
    QUALITY = 90

    def __init__(self, od: Path, source: "Source | None" = None) -> None:
        super().__init__(od, source)
        if source is None:
            raise ValueError("the saved frames go where the source says, and there is none")
        self.dir = source.frames
        self.part = self.dir.with_name(od.name + ".part")
        self.size: tuple[int, int] | None = None
        self.saved = 0

    def done(self) -> bool:
        return self.dir.is_dir()

    def start(self, width: int, height: int) -> None:
        scale = min(1.0, self.SHORT_EDGE / max(1, min(width, height)))
        if scale < 1.0:
            self.size = (round(width * scale), round(height * scale))
        shutil.rmtree(self.part, ignore_errors=True)
        self.part.mkdir(parents=True)

    def handle(self, fnum: int, bgr: "MatLike", rgb: "MatLike | None") -> None:
        img = bgr if self.size is None else cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        out = self.part / f"{self.od.name}_{fnum:07d}.jpg"
        if not cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, self.QUALITY]):
            raise RuntimeError(f"{self.od.name}: could not write {out}")
        self.saved += 1

    def finish(self) -> None:
        self.part.rename(self.dir)
        log_status(self.od, self.label, f"Saved {self.saved} frames under {self.dir}")
