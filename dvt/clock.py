"""Frame numbers to seconds on the mp4's own timeline.

Frame numbers count decoded frames and are exact. Seconds are not frame / fps.
A file may hold frames of more than one duration, and its video may begin a
few frames into the container, so frame / fps is right at the first and last
frame of a constant-rate file and wrong by seconds in between for the rest.
frames.parquet carries the time each frame is shown in `t`, read from the
container by dvt.tools.mp4.frame_times, and every table that states a time in
seconds for a frame number looks it up here.

The audio tables are on the same timeline: each is written with the m4a's
start on it added, by audio_clock, and metadata.parquet records that start as
`audio_start`, which is zero for nearly every file.
"""

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from .tools import ffprobe, mp4
from .tools.values import StrPath

FROM_FRAMES = "frames.parquet"
TIME_TOL = 0.002
FROM_VIDEO = "the container, read now"
FROM_FPS = "frame / fps"


class Clock:
    def __init__(
        self, t: ArrayLike | None = None, fps: float | None = None, source: str = FROM_FRAMES
    ) -> None:
        self.source = source
        self.fps = fps
        self.edges: NDArray[np.float64] | None = None
        times = None if t is None else np.asarray(t, dtype=np.float64)
        if times is not None and len(times):
            # The last frame is taken to last as long as the one before it.
            last = times[-1] - times[-2] if len(times) > 1 else (1.0 / fps if fps else 0.0)
            self.edges = np.append(times, times[-1] + last)

    @property
    def exact(self) -> bool:
        return self.edges is not None

    def at(self, frame: ArrayLike) -> NDArray[np.floating]:
        """When a frame begins. One past the last frame is when the video ends,
        so `at(f1 + 1)` is the end of a span that closes on frame f1."""
        frames = np.asarray(frame)
        if self.edges is None:
            if self.fps is None:
                raise TypeError("a clock with no times has no fps to count frames by")
            return frames / self.fps
        return self.edges[np.clip(frames, 0, len(self.edges) - 1)]


def places(shown: ArrayLike, t: ArrayLike) -> NDArray[np.intp]:
    """Which place in the index each decoded frame holds.

    A decoder can hand over fewer frames than the container lists. It skips a
    stretch of damaged video, after which every frame number is a few short
    of its place in the index, and it may keep back the last few frames it
    was holding when the file ended. It reports the time of each frame it
    does hand over, counted from the first frame of the video, and `shown`
    is those times. Raises when a frame matches no time in the index, since
    then nothing says which frame it is."""
    index = np.asarray(t, dtype=np.float64)
    rel = index - index[0]
    seen = np.asarray(shown, dtype=np.float64)
    j = np.clip(np.searchsorted(rel, seen), 1, len(rel) - 1)
    place = np.where(seen - rel[j - 1] < rel[j] - seen, j - 1, j)
    unmatched = int((np.abs(rel[place] - seen) > TIME_TOL).sum())
    if unmatched or np.any(np.diff(place) <= 0):
        raise ValueError(
            f"{unmatched} decoded frames match no time in the index"
            if unmatched
            else "decoded frames are not in index order"
        )
    return place


def of(od: StrPath, ip: StrPath | None = None, fps: float | None = None) -> Clock:
    """The clock of one film. frames.parquet written before `t` existed is
    made good from the video when the video is at hand and holds as many
    frames; failing that the old frame / fps stands in, and `source` says so."""
    path = Path(od) / "frames.parquet"
    decoded: int | None = None
    if path.exists():
        has_t = "t" in pl.read_parquet_schema(path)
        frames = pl.read_parquet(path, columns=["frame", "t"] if has_t else ["frame"])
        if has_t:
            return Clock(frames.sort("frame")["t"].to_numpy(), fps)
        decoded = frames.height
    if ip is not None and Path(ip).exists():
        t: list[float]
        try:
            t = mp4.frame_times(ip)
        except ffprobe.FFProbeError:
            t = []
        if t and decoded in (None, len(t)):
            return Clock(t, fps, FROM_VIDEO)
    if not fps:
        raise ValueError(
            f"{Path(od).name}: frames.parquet has no t, the video "
            f"does not supply one, and no fps was given"
        )
    return Clock(None, fps, FROM_FPS)
