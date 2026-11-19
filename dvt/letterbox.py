"""Letterbox/pillarbox detection for the metadata stage."""

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from .tools.values import StrPath


class Unread(TypedDict):
    """A sampled frame the video would not give up."""

    frame: int
    ok: Literal[False]


class Read(TypedDict):
    """One sampled frame as sample_evidence measures it."""

    frame: int
    ok: Literal[True]
    f_min: int
    f_max: int
    f_mean: float
    f_std: float
    f_chroma: float
    f_sat: float
    dead: bool
    row_max: list[int]
    col_max: list[int]
    row_mean: list[int]
    col_mean: list[int]


Sample = Read | Unread


# The crop: the bar on each side in pixels, how far it can be trusted, and
# the aspect of what is left, keyed as metadata.parquet's columns are.
Crop = dict[str, int | float | bool]
# The frame count, rate and size the video reports.
SampleMeta = dict[str, int | float]
Profiles = NDArray[np.uint8]
Index = NDArray[np.intp]

SIDES = ("top", "bottom", "left", "right")

SAMPLES = 240

DEAD_MAX = 32

MIN_USABLE = 10

THR = 8

PCTL = 10

UNION_PCTL = 5

MIN_RUN_FRAC = 0.03

GAP_LIT_MAX = 0.25

BLOCKS = 8
BLOCK_MIN = 5
DRIFT_MAX = 2

ASYM_MAX = 8

STD_THR = 8
STD_MIN = 4


def edge_runs(
    prof: ArrayLike, thr: int, min_run_frac: float = MIN_RUN_FRAC, gap_lit_max: float = GAP_LIT_MAX
) -> tuple[Index, Index, NDArray[np.bool_]]:
    prof = np.atleast_2d(prof)
    n, length = prof.shape
    lit = prof > thr
    zero = np.zeros((n, 1), np.int32)
    cl = np.concatenate([zero, np.cumsum(lit, axis=1, dtype=np.int32)], axis=1)

    any_lit = np.asarray(lit.any(axis=1))
    naive_near = lit.argmax(axis=1)
    naive_far = length - 1 - lit[:, ::-1].argmax(axis=1)

    run = max(1, min(int(min_run_frac * length), length))
    ok = lit if run <= 1 else (cl[:, run:] - cl[:, :-run]) == run
    has_run = ok.any(axis=1)
    run_near = ok.argmax(axis=1)
    run_far = ok.shape[1] - 1 - ok[:, ::-1].argmax(axis=1) + run - 1

    def guard(naive: Index, runwise: Index, lo: Index, hi: Index) -> Index:
        gap_len = np.maximum(hi - lo, 0)
        gap_lit = (
            np.take_along_axis(cl, hi[:, None], 1)[:, 0]
            - np.take_along_axis(cl, lo[:, None], 1)[:, 0]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(gap_len > 0, gap_lit / np.maximum(gap_len, 1), 0.0)
        return np.where(has_run & (frac <= gap_lit_max), runwise, naive)

    near = guard(naive_near, run_near, naive_near, run_near)
    far_idx = guard(naive_far, run_far, run_far + 1, naive_far + 1)
    return near, length - 1 - far_idx, any_lit


def block_spread(v: NDArray[np.integer], blocks: int = BLOCKS, pctl: float = PCTL) -> float:
    parts = [p for p in np.array_split(v, blocks) if len(p)]
    if len(parts) < 2:
        return 0.0
    est = [np.percentile(p, pctl) for p in parts]
    return float(max(est) - min(est))


def _qualify(rec: Crop, width: int, height: int, n_sampled: int, n_usable: int) -> Crop:
    rec["crop_n_sampled"] = int(n_sampled)
    rec["crop_n_usable"] = int(n_usable)
    rec["crop_dead_frac"] = 1.0 - n_usable / n_sampled if n_sampled else 1.0

    rec["crop_asym_px"] = int(
        max(
            abs(rec["crop_top"] - rec["crop_bottom"]),
            abs(rec["crop_left"] - rec["crop_right"]),
        )
    )
    rec["crop_odd"] = any(rec[f"crop_{s}"] % 2 for s in SIDES)

    kept_w = max(1, width - rec["crop_left"] - rec["crop_right"])
    kept_h = max(1, height - rec["crop_top"] - rec["crop_bottom"])
    rec["content_ar"] = kept_w / kept_h
    return rec


def crop_from_profiles(
    rows: ArrayLike, cols: ArrayLike, width: int, height: int, n_sampled: int | None = None
) -> Crop:
    """The crop from the row and column profiles of the usable frames, one
    row each. Fewer than MIN_USABLE frames leave it unmeasured."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    n_usable = len(rows)
    if n_sampled is None:
        n_sampled = n_usable

    rec: Crop = {
        "crop_top": 0,
        "crop_bottom": 0,
        "crop_left": 0,
        "crop_right": 0,
        "crop_stable": False,
        "crop_usable": False,
        "crop_drift_px": 0.0,
        "crop_union": False,
        "crop_grey": False,
        "crop_override": False,
    }
    if n_usable < MIN_USABLE:
        return _qualify(rec, width, height, n_sampled, n_usable)

    top, bottom, vr = edge_runs(rows, THR)
    left, right, vc = edge_runs(cols, THR)
    v = vr & vc
    if not v.sum():
        return _qualify(rec, width, height, n_sampled, n_usable)

    bars = {s: arr[v] for s, arr in zip(SIDES, (top, bottom, left, right), strict=True)}
    for s in SIDES:
        rec[f"crop_{s}"] = int(np.percentile(bars[s], PCTL))

    moved = [block_spread(bars[s]) for s in SIDES if len(bars[s]) >= BLOCKS * BLOCK_MIN]
    rec["crop_usable"] = True
    rec["crop_drift_px"] = float(max(moved)) if moved else 0.0
    rec["crop_stable"] = bool(moved and max(moved) <= DRIFT_MAX)

    rec["crop_union"] = bool(moved) and not rec["crop_stable"]
    if rec["crop_union"]:
        for s in SIDES:
            rec[f"crop_{s}"] = int(np.percentile(bars[s], UNION_PCTL))

    for prof, n, pair in ((rows, height, ("top", "bottom")), (cols, width, ("left", "right"))):
        live = np.flatnonzero(prof.astype(np.float32).std(axis=0) > STD_THR)
        if not len(live):
            continue
        near, far = int(live[0]), int(n - 1 - live[-1])
        if min(near, far) < STD_MIN or abs(near - far) > ASYM_MAX:
            continue
        if any(rec[f"crop_{s}"] > 2 for s in pair):
            continue
        for s, val in zip(pair, (near, far), strict=True):
            rec[f"crop_{s}"] = val
            rec["crop_grey"] = True

    return _qualify(rec, width, height, n_sampled, n_usable)


def usable(rows: Iterable[Sample]) -> list[Read]:
    """The sampled frames that were read and are not dead black."""
    return [r for r in rows if r["ok"] is True and not r["dead"]]


def profiles_from_rows(rows: Sequence[Sample]) -> tuple[Profiles, Profiles] | tuple[None, None]:
    use = usable(rows)
    if not use:
        return None, None
    return (
        np.array([r["row_mean"] for r in use], dtype=np.uint8),
        np.array([r["col_mean"] for r in use], dtype=np.uint8),
    )


EVIDENCE_SCHEMA = {
    "frame": pl.Int64,
    "ok": pl.Boolean,
    "f_min": pl.UInt8,
    "f_max": pl.UInt8,
    "f_mean": pl.Float32,
    "f_std": pl.Float32,
    "dead": pl.Boolean,
    "f_chroma": pl.Float32,
    "f_sat": pl.Float32,
    "row_max": pl.List(pl.UInt8),
    "col_max": pl.List(pl.UInt8),
    "row_mean": pl.List(pl.UInt8),
    "col_mean": pl.List(pl.UInt8),
}


def sample_evidence(ip: StrPath, samples: int = SAMPLES) -> tuple[list[Sample] | None, SampleMeta]:
    import cv2

    from .frame_ops import chroma_stats

    vinput = cv2.VideoCapture(str(ip))
    n_frames = int(vinput.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vinput.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vinput.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vinput.get(cv2.CAP_PROP_FPS)

    if n_frames <= 0 or height <= 0 or width <= 0:
        vinput.release()
        return None, {}

    idx = np.linspace(n_frames * 0.02, n_frames * 0.98, samples).astype(int)

    rows: list[Sample] = []
    for f in idx:
        vinput.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        status, img = vinput.read()
        if not status:
            rows.append({"frame": int(f), "ok": False})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        chroma, sat = chroma_stats(img)
        rows.append(
            {
                "frame": int(f),
                "ok": True,
                "f_min": int(gray.min()),
                "f_max": int(gray.max()),
                "f_mean": float(gray.mean()) / 255.0,
                "f_std": float(gray.std()) / 255.0,
                "f_chroma": chroma,
                "f_sat": sat,
                "dead": bool(gray.max() < DEAD_MAX),
                "row_max": gray.max(axis=1).tolist(),
                "col_max": gray.max(axis=0).tolist(),
                "row_mean": gray.mean(axis=1).round().astype(np.uint8).tolist(),
                "col_mean": gray.mean(axis=0).round().astype(np.uint8).tolist(),
            }
        )

    vinput.release()
    return rows, {
        "n_frames": n_frames,
        "fps": fps,
        "img_width": width,
        "img_height": height,
    }


# The last video sampled, by its path: letterbox, chroma and metadata read
# the same frames of one film.
_SAMPLED: dict[str, tuple[list[Sample] | None, SampleMeta]] = {}


def sampled(ip: StrPath, samples: int = SAMPLES) -> tuple[list[Sample] | None, SampleMeta]:
    key = str(ip)
    if key not in _SAMPLED:
        _SAMPLED.clear()
        _SAMPLED[key] = sample_evidence(ip, samples)
    return _SAMPLED[key]


def create_letterbox(ip: StrPath, od: Path, samples: int = SAMPLES) -> None:
    out = od / "letterbox.parquet"
    if out.exists():
        print(f"[{od.name}:letterbox] Already exists")
        return

    t0 = time.monotonic()
    if not Path(ip).exists():
        print(f"[{od.name}:letterbox] Missing film at {ip} — skipped")
        return

    rows, meta = sampled(ip, samples)
    if not rows:
        print(f"[{od.name}:letterbox] Unreadable — skipped")
        return

    df = pl.DataFrame(rows, schema=EVIDENCE_SCHEMA)
    df = df.with_columns(
        fname=pl.lit(od.name),
        **{k: pl.lit(v) for k, v in meta.items()},
    )
    od.mkdir(parents=True, exist_ok=True)
    from .annotators import stamp

    stamp(df, "letterbox").write_parquet(out)

    usable = df.filter(pl.col("ok") & ~pl.col("dead")).height
    size_kb = out.stat().st_size / 1024
    print(
        f"[{od.name}:letterbox] {df.height} sampled, {usable} usable, "
        f"{size_kb:.0f} KB, {time.monotonic() - t0:.0f}s"
    )
