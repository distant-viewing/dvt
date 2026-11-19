"""The time each video frame of an mp4 is shown, read from the file's index.

An mp4 keeps its timing in the `moov` box, a few megabytes however long the
film: how long each sample lasts (stts), how far its display is moved from its
decoding (ctts), and where the track sits on the file's timeline (elst). This
reads that box and nothing else. ffprobe gives the same times but reads every
packet of the file to list them, which for the corpus is eighteen terabytes.

Only the plain layout is handled: any number of empty edits, then one edit at
normal speed that covers every sample. Anything else raises Unhandled, and
`frame_times` then asks ffprobe, so an unusual file costs a full read and is
never timed wrongly.
"""

import struct
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path

from . import ffprobe
from .values import StrPath

Box = tuple[int, int]

MAX_MOOV = 1 << 30


class Unhandled(ValueError):
    pass


def _moov(path: StrPath) -> bytes:
    with open(path, "rb") as f:
        end = f.seek(0, 2)
        pos = 0
        while pos + 8 <= end:
            f.seek(pos)
            size, kind = struct.unpack(">I4s", f.read(8))
            head = 8
            if size == 1:
                size, head = struct.unpack(">Q", f.read(8))[0], 16
            elif size == 0:
                size = end - pos
            if size < head:
                raise Unhandled(f"a {kind!r} box of size {size}")
            if kind == b"moov":
                if size > MAX_MOOV:
                    raise Unhandled(f"a moov of {size} bytes")
                return f.read(size - head)
            pos += size
    raise Unhandled("no moov box")


def _boxes(buf: bytes, start: int, stop: int) -> Iterator[tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= stop:
        size, kind = struct.unpack_from(">I4s", buf, pos)
        head = 8
        if size == 1:
            size, head = struct.unpack_from(">Q", buf, pos + 8)[0], 16
        elif size == 0:
            size = stop - pos
        if size < head or pos + size > stop:
            raise Unhandled(f"a {kind!r} box that overruns its parent")
        yield kind, pos + head, pos + size
        pos += size


def _find(buf: bytes, start: int, stop: int, *names: bytes) -> Box | None:
    """The first box at the end of a path of names, as (start, stop), or None."""
    for kind, a, b in _boxes(buf, start, stop):
        if kind == names[0]:
            return (a, b) if len(names) == 1 else _find(buf, a, b, *names[1:])
    return None


def _timescale(buf: bytes, box: Box) -> int:
    version = buf[box[0]]
    scale: int = struct.unpack_from(">I", buf, box[0] + (20 if version else 12))[0]
    return scale


def _runs(buf: bytes, box: Box, signed: bool = False) -> list[int]:
    """The (count, value) runs of an stts or ctts box, written out per sample."""
    count: int = struct.unpack_from(">I", buf, box[0] + 4)[0]
    flat: tuple[int, ...] = struct.unpack_from(f">{2 * count}I", buf, box[0] + 8)
    out: list[int] = []
    for n, value in zip(flat[0::2], flat[1::2], strict=True):
        if signed and value >= 1 << 31:
            value -= 1 << 32
        out.extend([value] * n)
    return out


def _edits(buf: bytes, box: Box) -> list[tuple[int, int, int]]:
    """(duration in movie ticks, media time, rate) for each edit."""
    version = buf[box[0]]
    count: int = struct.unpack_from(">I", buf, box[0] + 4)[0]
    out: list[tuple[int, int, int]] = []
    pos = box[0] + 8
    for _ in range(count):
        edit: tuple[int, int, int]
        if version:
            edit = struct.unpack_from(">Qqi", buf, pos)
            pos += 20
        else:
            edit = struct.unpack_from(">Iii", buf, pos)
            pos += 12
        out.append(edit)
    return out


def _rescale(value: int, to: int, of: int) -> int:
    """value * to / of rounded to nearest, halves away from zero, as ffmpeg does."""
    return (value * to + of // 2) // of


def index_times(path: StrPath) -> list[float]:
    """Frame times from the index alone. Raises Unhandled for any layout this
    does not cover."""
    buf = _moov(path)
    mvhd = _find(buf, 0, len(buf), b"mvhd")
    if mvhd is None:
        raise Unhandled("no mvhd")
    movie_scale = _timescale(buf, mvhd)

    video = None
    for kind, a, b in _boxes(buf, 0, len(buf)):
        if kind != b"trak":
            continue
        hdlr = _find(buf, a, b, b"mdia", b"hdlr")
        if hdlr and buf[hdlr[0] + 8 : hdlr[0] + 12] == b"vide":
            video = (a, b)
            break
    if video is None:
        raise Unhandled("no video track")

    mdhd = _find(buf, *video, b"mdia", b"mdhd")
    stts = _find(buf, *video, b"mdia", b"minf", b"stbl", b"stts")
    if mdhd is None or stts is None:
        raise Unhandled("no mdhd or stts")
    scale = _timescale(buf, mdhd)
    if not scale or not movie_scale:
        raise Unhandled("a timescale of zero")

    deltas = _runs(buf, stts)
    if not deltas:
        raise Unhandled("no samples in the index; the file may be fragmented")
    dts = [0, *accumulate(deltas[:-1])]
    ctts = _find(buf, *video, b"mdia", b"minf", b"stbl", b"ctts")
    if ctts is None:
        pts = dts
    else:
        shift = _runs(buf, ctts, signed=True)
        if len(shift) != len(dts):
            raise Unhandled(f"{len(dts)} samples but {len(shift)} display offsets")
        pts = [d + s for d, s in zip(dts, shift, strict=True)]

    empty, media_time = 0, 0
    span: int | None = None
    elst = _find(buf, *video, b"edts", b"elst")
    if elst is not None:
        edits = _edits(buf, elst)
        while edits and edits[0][1] == -1:
            empty += edits.pop(0)[0]
        if len(edits) > 1:
            raise Unhandled(f"{len(edits)} edits")
        if edits:
            span, media_time, rate = edits[0]
            if rate != 0x10000:
                raise Unhandled(f"an edit at rate {rate / 0x10000:g}")
    if min(pts) < media_time:
        raise Unhandled("samples shown before the edit begins")
    if span and max(pts) >= media_time + _rescale(span, scale, movie_scale) + max(deltas):
        raise Unhandled("samples shown after the edit ends")

    offset = _rescale(empty, scale, movie_scale) - media_time
    return sorted((p + offset) / scale for p in pts)


def frame_times(path: StrPath) -> list[float]:
    """The time each video frame is shown, in seconds on the file's own
    timeline and in the order the frames are shown, so the nth value belongs
    to the nth frame a decoder hands over. From the index when the file is an
    mp4 of the plain layout, from ffprobe otherwise."""
    if Path(path).suffix.lower() in (".mp4", ".m4v", ".mov"):
        try:
            return index_times(path)
        except (Unhandled, struct.error, OSError):
            pass
    return ffprobe.frame_times(path)
