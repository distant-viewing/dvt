"""What the frame-sampled stages share: the frames each takes and the one
decode they read."""

import os
import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
import polars as pl

from . import frame_ops
from .frame_ops import Box
from .log import log_status
from .runtime import OVERLAP
from .tools.values import StrPath

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from .source import Source

source_crop_box = frame_ops.source_crop_box


FALLBACK_STRIDE = 24


def sample_stride(ip: StrPath) -> int:
    vinput = cv2.VideoCapture(str(ip))
    fps = vinput.get(cv2.CAP_PROP_FPS)
    vinput.release()

    if not fps or not np.isfinite(fps) or fps <= 0:
        return FALLBACK_STRIDE

    return max(1, round(fps))


def sample_grid(fps: float, hz: float | None) -> tuple[int, frozenset[int]]:
    if hz is None:
        return 1, frozenset({0})
    base = max(1, round(fps)) if fps and np.isfinite(fps) and fps > 0 else FALLBACK_STRIDE
    if hz < 1:
        return base * max(1, int(round(1 / hz))), frozenset({0})
    n = max(1, int(round(hz)))
    return base, frozenset(int(round(i * base / n)) % base for i in range(n))


def shot_backfill(
    od: Path, period: int, offsets: Iterable[int], every: bool = False
) -> frozenset[int]:
    """The middle frame of each shot the grid misses, or with `every` of
    each shot whether the grid lands in it or not."""
    shots = pl.read_parquet(od / "shots.parquet", columns=["f0", "f1", "fmid"])
    if not shots.height:
        return frozenset()
    if every:
        return frozenset(shots["fmid"].to_list())
    f0 = shots["f0"].to_numpy()
    f1 = shots["f1"].to_numpy()

    covered = np.zeros(len(f0), dtype=bool)
    for o in offsets:
        n = np.maximum(0, -((o - f0) // period))
        covered |= (o + period * n) <= f1

    return frozenset(shots["fmid"].to_numpy()[~covered].tolist())


def _gate(
    base: int, offsets: frozenset[int], extra: frozenset[int] = frozenset()
) -> Callable[[int], bool]:
    """Whether a frame is due: on the grid, or one of the extra frames."""

    def due(fnum: int) -> bool:
        return (fnum % base) in offsets or fnum in extra

    return due


def stage_gates(
    od: Path, stages: Sequence["FrameStage"], fps: float
) -> list[Callable[[int], bool]]:
    gates: list[Callable[[int], bool]] = []
    for s in stages:
        base, offsets = sample_grid(fps, s.sample_hz)
        if s.backfill_shots:
            s.backfill = shot_backfill(od, base, offsets, every=s.every_shot)
            gates.append(_gate(base, offsets, s.backfill))
        else:
            gates.append(_gate(base, offsets))
    return gates


class FrameStage:
    label = ""
    needs_rgb = False
    sample_hz: float | None = 1.0
    backfill_shots = False
    # With backfill_shots, take every shot's middle frame rather than only
    # those the grid misses.
    every_shot = False
    wants_crop = False
    # Modules start() imports lazily. check_imports() loads them once before
    # the work list, so a missing package fails the task rather than each film.
    imports: tuple[str, ...] = ()

    backfill: frozenset[int] = frozenset()
    ip: StrPath | None = None

    # A stage that sets overlap runs each full batch on a worker thread of
    # its own, so the card works on one batch while the main thread decodes
    # and resizes the frames of the next. One batch is in flight at a time
    # and they run in the order they filled, so the rows come out as they
    # would on the main thread.
    overlap = False

    def __init__(self, od: Path, source: "Source | None" = None) -> None:
        self.od = od
        self.source = source
        self._worker: ThreadPoolExecutor | None = None
        self._inflight: Future[object] | None = None

    def submit[*Ts](self, fn: Callable[[*Ts], object], *args: *Ts) -> None:
        if not (self.overlap and OVERLAP):
            fn(*args)
            return
        if self._worker is None:
            self._worker = ThreadPoolExecutor(
                1, thread_name_prefix=self.label or type(self).__name__
            )
        self.wait()
        self._inflight = self._worker.submit(fn, *args)

    def wait(self) -> None:
        """Block until the batch in flight is done, and raise what it raised."""
        if self._inflight is not None:
            future, self._inflight = self._inflight, None
            future.result()

    def close(self) -> None:
        """Stop the worker. A batch still in flight is finished and its rows
        dropped with the stage."""
        if self._worker is not None:
            self._worker.shutdown(wait=True, cancel_futures=True)
            self._worker = None
        self._inflight = None

    def done(self) -> bool:
        raise NotImplementedError

    def resolve_crop(self, width: int, height: int, name: str | None = None) -> Box | None:
        name = name or self.label
        meta_path = self.od / "metadata.parquet"
        if not meta_path.exists() or "crop_top" not in (meta := pl.read_parquet(meta_path)).columns:
            log_status(
                self.od, name, "NO metadata.parquet — embedding the stored frame, bars included"
            )
            return None
        crop = source_crop_box(meta, width, height)
        if crop is None:
            log_status(self.od, name, "letterbox crop is empty — frame is all content")
        else:
            t, b, left, r = crop
            bar = 100 * (1 - (b - t) * (r - left) / (width * height))
            log_status(
                self.od,
                name,
                f"cropped to [{t}:{b}, {left}:{r}] of {width}x{height} "
                f"— {bar:.1f}% of the frame was bar",
            )
        return crop

    def start(self, width: int, height: int) -> None:
        pass

    def handle(self, fnum: int, bgr: "MatLike", rgb: "MatLike | None") -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


PREFETCH_BYTES = int(os.environ.get("TBM_PREFETCH_BYTES", 256 * 1024 * 1024))


class Reader(Protocol):
    """What decoded_frames reads from: a cv2.VideoCapture."""

    def read(self) -> tuple[bool, "MatLike"]: ...


class _Done:
    """Put on the queue when the decode thread stops."""


def decoded_frames(
    vinput: Reader, width: int, height: int, prefetch: bool | None = None
) -> Iterator["MatLike"]:
    if prefetch is None:
        prefetch = os.environ.get("TBM_PREFETCH", "0") != "0"

    if not prefetch:
        while True:
            status, img = vinput.read()
            if not status:
                return
            yield img
        return

    frame_bytes = max(1, width * height * 3)
    q: queue.Queue[MatLike | BaseException | _Done] = queue.Queue(
        maxsize=max(4, min(64, PREFETCH_BYTES // frame_bytes))
    )
    stop = threading.Event()
    done = _Done()

    def produce() -> None:
        try:
            while not stop.is_set():
                status, img = vinput.read()
                if not status:
                    break
                q.put(img)
        except BaseException as exc:
            q.put(exc)
        finally:
            q.put(done)

    thread = threading.Thread(target=produce, name="decode", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if isinstance(item, _Done):
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while thread.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                thread.join(timeout=0.1)
