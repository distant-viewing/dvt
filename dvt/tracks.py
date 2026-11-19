"""Head/face tracks within shots, and one face embedding per track."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import col as c
from scipy.cluster.vq import kmeans2

from . import clock
from .annotators import stamp
from .bytetrack import BaseTrack, BYTETracker, TrackerArgs
from .tools.boxes import TrackParams, containment, nms, pairwise_iou
from .tools.values import Raw, StrPath

NAME = "tracks"
OUTPUTS = ("tracks.parquet", "tracks_frames.parquet")
INPUTS = ("shots.parquet", "face.parquet", "yolo.parquet")

EMB_DIM = 512
KPS_DIM = 10

FALLBACK_FPS = 24.0


PARAMS = TrackParams()

TRACKS_SCHEMA = {
    "track_id": pl.Int64,
    "sid": pl.Int64,
    "first_frame": pl.Int64,
    "last_frame": pl.Int64,
    "n_frames": pl.Int64,
    "t0": pl.Float64,
    "t1": pl.Float64,
    "n_faces": pl.Int64,
    "n_faces_good": pl.Int64,
    "n_used": pl.Int64,
    "embedding": pl.Array(pl.Float32, EMB_DIM),
    "emb_quality": pl.Float32,
    "emb_variance": pl.Float32,
    "frontal_frac": pl.Float32,
    "yaw_abs_med": pl.Float32,
    "cx": pl.Float32,
    "cy": pl.Float32,
    "h": pl.Float32,
    "exemplars": pl.List(pl.Struct({"frame": pl.Int64, "face_id": pl.Int64})),
    "n_stitched": pl.Int32,
    "was_split": pl.Boolean,
}

FRAMES_SCHEMA = {
    "track_id": pl.Int64,
    "sid": pl.Int64,
    "frame": pl.Int64,
    "x1": pl.Float32,
    "y1": pl.Float32,
    "x2": pl.Float32,
    "y2": pl.Float32,
    "head_score": pl.Float32,
    "face_id": pl.Int64,
    "fx1": pl.Float32,
    "fy1": pl.Float32,
    "fx2": pl.Float32,
    "fy2": pl.Float32,
    "face_score": pl.Float32,
    "emb_norm": pl.Float32,
    "kps": pl.Array(pl.Float32, KPS_DIM),
}


class Det(TypedDict):
    """One detection to track: a head, or a face standing in for a missed
    head, with the face it carries when it has one."""

    frame: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    fidx: int | None


class Piece(TypedDict):
    """A tracklet as stitch_tracklets weighs it: its span, its embedding
    centroid, where its box starts and ends, and its width."""

    first: int
    last: int
    cen: NDArray[np.float64] | None
    c0: NDArray[np.float64]
    c1: NDArray[np.float64]
    w: float


# A row of tracks.parquet, before the columns are made from it.
TrackRow = dict[str, object]
# The counts a run reports.
Stats = dict[str, int]

FACE_OUT = ["face_id", "fx1", "fy1", "fx2", "fy2", "face_score", "emb_norm", "kps"]


def log_status(od: Path, status: str) -> None:
    print(f"[{od.name}:{NAME}] {status}")


def _whole(value: object) -> int:
    """int() of a polars min or max, which is None for an empty column; that
    raises as int(None) does."""
    if isinstance(value, int | float):
        return int(value)
    raise TypeError(f"int() argument must be a number, not {type(value).__name__}")


def as_matrix(series: pl.Series, dim: int) -> np.ndarray:
    arr = series.to_numpy()
    if arr.dtype == object:
        arr = np.stack(list(arr)) if len(arr) else np.zeros((0, dim))
    return np.ascontiguousarray(arr, dtype=np.float64).reshape(-1, dim)


def face_yaw(kps: np.ndarray) -> np.ndarray:
    k = kps.reshape(-1, 5, 2)
    le, re, nose = k[:, 0], k[:, 1], k[:, 2]
    iod = np.abs(re[:, 0] - le[:, 0])
    return (nose[:, 0] - (le[:, 0] + re[:, 0]) / 2) / np.maximum(iod, 1e-3)


def load_inputs(od: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    shots = pl.read_parquet(od / "shots.parquet", columns=["sid", "f0", "f1"]).sort("f0")
    heads = (
        pl.scan_parquet(od / "yolo.parquet")
        .filter(c.label == "head")
        .select(
            "frame",
            "score",
            c.top_left_x.alias("x1"),
            c.top_left_y.alias("y1"),
            c.bottom_right_x.alias("x2"),
            c.bottom_right_y.alias("y2"),
        )
        .sort("frame", "score", descending=[False, True])
        .collect()
    )
    faces = (
        pl.scan_parquet(od / "face.parquet")
        .select(
            "frame",
            "face_id",
            c.det_score.alias("score"),
            c.bbox_x1.alias("x1"),
            c.bbox_y1.alias("y1"),
            c.bbox_x2.alias("x2"),
            c.bbox_y2.alias("y2"),
            "kps",
            "embedding",
        )
        .sort("frame", "face_id")
        .collect()
    )
    emb = as_matrix(faces["embedding"], EMB_DIM)
    norms = np.linalg.norm(emb, axis=1)
    unit_emb = emb / np.maximum(norms, 1e-12)[:, None]
    faces = faces.with_columns(
        emb_norm=pl.Series(norms, dtype=pl.Float64),
        embedding=pl.Series(unit_emb.astype(np.float32), dtype=pl.Array(pl.Float32, EMB_DIM)),
        yaw=pl.Series(face_yaw(as_matrix(faces["kps"], KPS_DIM)), dtype=pl.Float64),
    )
    return shots, faces, heads


def assign_shots(df: pl.DataFrame, shots: pl.DataFrame) -> pl.DataFrame:
    f0 = shots["f0"].to_numpy()
    sids = shots["sid"].to_numpy()
    pos = np.clip(np.searchsorted(f0, df["frame"].to_numpy(), side="right") - 1, 0, len(f0) - 1)
    return df.with_columns(sid=pl.Series(sids[pos], dtype=pl.Int64))


DETS_SCHEMA = {
    "frame": pl.Int64,
    "score": pl.Float64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
    "fidx": pl.Int64,
}


def fuse_detections(heads: pl.DataFrame, faces: pl.DataFrame, p: TrackParams) -> pl.DataFrame:
    faces_by = faces.partition_by("frame", as_dict=True)
    heads_by = heads.partition_by("frame", as_dict=True)
    out: list[Det] = []
    for key in sorted(set(faces_by) | set(heads_by)):
        frame = key[0]
        rows: list[Det] = []
        hdf, fdf = heads_by.get(key), faces_by.get(key)
        h = (
            hdf.select("x1", "y1", "x2", "y2").to_numpy().astype(float)
            if hdf is not None
            else np.zeros((0, 4))
        )
        hs = hdf["score"].to_numpy().astype(float).copy() if hdf is not None else np.zeros(0)
        hf: list[int | None] = [None] * len(h)
        hf_score = np.full(len(h), -1.0)
        if fdf is not None:
            f = fdf.select("x1", "y1", "x2", "y2").to_numpy().astype(float)
            fs = fdf["score"].to_numpy().astype(float)
            fidx = fdf["fidx"].to_list()
            cont = containment(f, h) if len(h) else np.zeros((len(f), 0))
            for fi in range(len(f)):
                hi = int(cont[fi].argmax()) if cont.shape[1] else -1
                if hi >= 0 and cont[fi, hi] >= p.face_in_head_min:
                    if fs[fi] > hf_score[hi]:
                        hf[hi], hf_score[hi] = fidx[fi], fs[fi]
                    hs[hi] = max(hs[hi], fs[fi])
                else:
                    fw, fh = f[fi, 2] - f[fi, 0], f[fi, 3] - f[fi, 1]
                    cx = (f[fi, 0] + f[fi, 2]) / 2
                    cy = (f[fi, 1] + f[fi, 3]) / 2 + p.head_dy * fh
                    w, hh = fw * p.head_wr, fh * p.head_hr
                    rows.append(
                        {
                            "frame": frame,
                            "score": float(fs[fi]),
                            "x1": cx - w / 2,
                            "y1": cy - hh / 2,
                            "x2": cx + w / 2,
                            "y2": cy + hh / 2,
                            "fidx": fidx[fi],
                        }
                    )
        for i in range(len(h)):
            rows.append(
                {
                    "frame": frame,
                    "score": float(hs[i]),
                    "x1": float(h[i, 0]),
                    "y1": float(h[i, 1]),
                    "x2": float(h[i, 2]),
                    "y2": float(h[i, 3]),
                    "fidx": hf[i],
                }
            )
        if len(rows) > 1:
            b = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows])
            sc = np.array([r["score"] for r in rows])
            rows = [rows[i] for i in nms(b, sc, p.nms_iou)]
        out += rows
    return pl.DataFrame(out, schema=DETS_SCHEMA).sort("frame", "score", descending=[False, True])


TRACKS_RAW_SCHEMA = {
    "frame": pl.Int64,
    "track_id": pl.Int64,
    "score": pl.Float64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
}


def run_bytetrack(
    shot: Raw,
    dets: pl.DataFrame,
    fps: float,
    img_hw: tuple[int, int],
    E: np.ndarray,
    norms: np.ndarray,
    p: TrackParams,
) -> pl.DataFrame:
    BaseTrack.reset_ids()
    buffer_frames = max(1, int(round(p.buffer_s * fps)))
    args = TrackerArgs(
        track_thresh=p.track_thresh,
        track_buffer=buffer_frames,
        match_thresh=p.match_thresh,
        mot20=False,
        embed_gate=p.embed_gate,
    )
    tracker = BYTETracker(args, frame_rate=30)

    by_frame = dets.partition_by("frame", as_dict=True)
    empty = np.zeros((0, 5), dtype=np.float64)
    rows: list[tuple[int, int, float, float, float, float, float]] = []
    for frame in range(shot["f0"], shot["f1"] + 1):
        df = by_frame.get((frame,))
        if df is None:
            boxes, feats = empty, None
        else:
            boxes = df.select("x1", "y1", "x2", "y2", "score").to_numpy().astype(np.float64)
            feats = [
                None if i is None or norms[i] < p.emb_min_norm else E[i]
                for i in df["fidx"].to_list()
            ]
        for t in tracker.update(boxes.copy(), img_hw, img_hw, feats=feats):
            x1, y1, x2, y2 = t.tlbr
            rows.append(
                (frame, int(t.track_id), float(t.score), float(x1), float(y1), float(x2), float(y2))
            )

    tracks = pl.DataFrame(rows, schema=TRACKS_RAW_SCHEMA, orient="row")
    if tracks.is_empty():
        return tracks
    keep = (
        tracks.group_by("track_id")
        .agg(n=pl.len(), first=c.frame.min())
        .filter(c.n >= p.min_track_frames)
        .sort("first", "track_id")
        .with_row_index("new_id", offset=1)
        .select("track_id", c.new_id.cast(pl.Int64))
    )
    return (
        tracks.join(keep, on="track_id", how="inner")
        .drop("track_id")
        .rename({"new_id": "track_id"})
        .sort("frame", "track_id")
    )


def attach_faces(tracks: pl.DataFrame, faces: pl.DataFrame, p: TrackParams) -> pl.DataFrame:
    tracks = tracks.with_row_index("ri")
    fidx_of = np.full(tracks.height, -1, dtype=np.int64)
    faces_by = faces.partition_by("frame", as_dict=True)
    for key, tdf in tracks.partition_by("frame", as_dict=True).items():
        fdf = faces_by.get(key)
        if fdf is None:
            continue
        contain = containment(
            fdf.select("x1", "y1", "x2", "y2").to_numpy().astype(float),
            tdf.select("x1", "y1", "x2", "y2").to_numpy().astype(float),
        )
        order = np.dstack(np.unravel_index(np.argsort(-contain, axis=None), contain.shape))[0]
        ri, fidx = tdf["ri"].to_numpy(), fdf["fidx"].to_numpy()
        used_f: set[int] = set()
        used_h: set[int] = set()
        for fi, hi in order:
            if contain[fi, hi] < p.face_in_head_min:
                break
            if fi in used_f or hi in used_h:
                continue
            used_f.add(fi)
            used_h.add(hi)
            fidx_of[ri[hi]] = fidx[fi]
    face_cols = faces.select(
        "fidx",
        "face_id",
        c.x1.alias("fx1"),
        c.y1.alias("fy1"),
        c.x2.alias("fx2"),
        c.y2.alias("fy2"),
        c.score.alias("face_score"),
        "emb_norm",
        "kps",
        "yaw",
    )
    return (
        tracks.drop("ri")
        .with_columns(fidx=pl.Series(fidx_of, dtype=pl.Int64))
        .with_columns(fidx=pl.when(c.fidx >= 0).then(c.fidx).otherwise(None))
        .join(face_cols, on="fidx", how="left")
        .sort("frame", "track_id")
    )


def renumber(tracked: pl.DataFrame) -> tuple[pl.DataFrame, dict[int, int]]:
    order = (
        tracked.group_by("track_id")
        .agg(first=c.frame.min())
        .sort("first", "track_id")
        .with_row_index("new_id", offset=1)
        .select("track_id", c.new_id.cast(pl.Int64))
    )
    mapping = dict(zip(order["track_id"].to_list(), order["new_id"].to_list(), strict=True))
    return (
        tracked.join(order, on="track_id")
        .drop("track_id")
        .rename({"new_id": "track_id"})
        .sort("frame", "track_id")
    ), mapping


def merge_duplicate_tracks(tracked: pl.DataFrame, p: TrackParams) -> tuple[pl.DataFrame, int]:
    parts = {k[0]: df for k, df in tracked.partition_by("track_id", as_dict=True).items()}
    nfaces = {tid: int(df["fidx"].is_not_null().sum()) for tid, df in parts.items()}
    span = {
        tid: (_whole(df["frame"].min()), _whole(df["frame"].max())) for tid, df in parts.items()
    }
    keys = sorted(parts, key=lambda t: (-nfaces[t], t))
    owner = {tid: tid for tid in parts}
    drop_rows: list[tuple[int, int]] = []
    n_merged = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if owner[b] != b or owner[a] != a:
                continue
            if span[a][1] < span[b][0] or span[b][1] < span[a][0]:
                continue
            j = parts[a].join(
                parts[b].select("frame", "x1", "y1", "x2", "y2"), on="frame", suffix="_b"
            )
            if j.height < 3:
                continue
            iou = pairwise_iou(
                j.select("x1", "y1", "x2", "y2").to_numpy().astype(float),
                j.select("x1_b", "y1_b", "x2_b", "y2_b").to_numpy().astype(float),
            )
            if iou.mean() > p.dup_track_iou:
                owner[b] = a
                drop_rows += [(b, int(fr)) for fr in j["frame"].to_list()]
                n_merged += 1
    if not n_merged:
        return tracked, 0
    drop = pl.DataFrame(drop_rows, schema={"track_id": pl.Int64, "frame": pl.Int64}, orient="row")
    return (
        tracked.join(drop, on=["track_id", "frame"], how="anti")
        .with_columns(track_id=c.track_id.replace_strict(owner, return_dtype=pl.Int64))
        .sort("frame", "track_id")
    ), n_merged


def split_id_switches(
    tracked: pl.DataFrame, E: np.ndarray, p: TrackParams
) -> tuple[pl.DataFrame, set[int], int]:
    next_id = _whole(tracked["track_id"].max()) + 1
    new_ids = tracked["track_id"].to_numpy().copy()
    fidx = tracked["fidx"].fill_null(-1).to_numpy()
    good = (fidx >= 0) & (tracked["emb_norm"].fill_null(0.0).to_numpy() >= p.emb_min_norm)
    touched: set[int] = set()
    n_splits = 0
    for df in tracked.with_row_index("ri").partition_by("track_id", as_dict=True).values():
        ri = df["ri"].to_numpy()
        seg_sum: NDArray[np.float64] | None = None
        seg_n = 0
        spike: list[int] = []
        for r in ri:
            if not good[r]:
                continue
            e = E[fidx[r]]
            if seg_n >= p.split_min_faces:
                if seg_sum is None:
                    raise ValueError("split_min_faces must be at least 1")
                cen = seg_sum / np.linalg.norm(seg_sum)
                if 1.0 - float(e @ cen) > p.split_dist:
                    spike.append(r)
                    if len(spike) >= p.split_persist:
                        cut = spike[0]
                        new_ids[ri[ri >= cut]] = next_id
                        touched |= {int(new_ids[ri[0]]), next_id}
                        next_id += 1
                        n_splits += 1
                        seg_sum = E[fidx[spike]].sum(axis=0)
                        seg_n, spike = len(spike), []
                    continue
            spike = []
            seg_sum = e.copy() if seg_sum is None else seg_sum + e
            seg_n += 1
    return tracked.with_columns(track_id=pl.Series(new_ids, dtype=pl.Int64)), touched, n_splits


def stitch_tracklets(
    tracked: pl.DataFrame, E: np.ndarray, fps: float, p: TrackParams
) -> tuple[pl.DataFrame, dict[int, int]]:
    max_gap = int(round(p.stitch_gap_s * fps))
    info: dict[int, Piece] = {}
    fidx_all = tracked["fidx"].fill_null(-1).to_numpy()
    good_all = (fidx_all >= 0) & (tracked["emb_norm"].fill_null(0.0).to_numpy() >= p.emb_min_norm)
    for key, df in tracked.with_row_index("ri").partition_by("track_id", as_dict=True).items():
        df = df.sort("frame")
        ri = df["ri"].to_numpy()
        sel = ri[good_all[ri]]
        cen: NDArray[np.float64] | None = None
        if len(sel):
            centroid: NDArray[np.float64] = E[fidx_all[sel]].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            cen = centroid
        b = df.select("x1", "y1", "x2", "y2").to_numpy().astype(float)
        info[key[0]] = {
            "first": int(df["frame"][0]),
            "last": int(df["frame"][-1]),
            "cen": cen,
            "c0": np.array([(b[0, 0] + b[0, 2]) / 2, (b[0, 1] + b[0, 3]) / 2]),
            "c1": np.array([(b[-1, 0] + b[-1, 2]) / 2, (b[-1, 1] + b[-1, 3]) / 2]),
            "w": float(np.mean(b[:, 2] - b[:, 0])),
        }
    cands: list[tuple[float, int, int]] = []
    for a, A in info.items():
        if A["cen"] is None:
            continue
        for b, B in info.items():
            gap = B["first"] - A["last"] - 1
            if a == b or gap < 0 or gap > max_gap or B["cen"] is None:
                continue
            dist = 1.0 - float(A["cen"] @ B["cen"])
            space = float(np.linalg.norm(A["c1"] - B["c0"])) / max((A["w"] + B["w"]) / 2, 1.0)
            if dist <= p.stitch_dist and space <= p.stitch_space + p.stitch_space_per_frame * gap:
                cands.append((dist, a, b))
    parent = {tid: tid for tid in info}

    def root(t: int) -> int:
        while parent[t] != t:
            t = parent[t]
        return t

    has_next: set[int] = set()
    has_prev: set[int] = set()
    for _dist, a, b in sorted(cands):
        if a in has_next or b in has_prev or root(a) == root(b):
            continue
        has_next.add(a)
        has_prev.add(b)
        parent[root(b)] = root(a)
    mapping = {tid: root(tid) for tid in info}
    return tracked.with_columns(
        track_id=c.track_id.replace_strict(mapping, return_dtype=pl.Int64)
    ), mapping


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def medoid_index(X: np.ndarray) -> int:
    return int((X @ X.sum(axis=0)).argmax())


def farthest_point_exemplars(X: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    if len(X) == 0:
        return []
    cand = np.where(q >= np.median(q))[0]
    chosen = [int(cand[q[cand].argmax()])]
    while len(chosen) < min(k, len(cand)):
        mind = (1.0 - X[cand] @ X[chosen].T).min(axis=1)
        mind[np.isin(cand, chosen)] = -1
        if mind.max() <= 0:
            break
        chosen.append(int(cand[mind.argmax()]))
    return chosen


def pool_track(df: pl.DataFrame, E: np.ndarray, p: TrackParams) -> dict[str, object]:
    fdf = df.filter(c.fidx.is_not_null()).sort("frame")
    n = fdf.height
    out: dict[str, object] = {
        "n_faces": n,
        "n_faces_good": 0,
        "n_used": 0,
        "embedding": None,
        "emb_quality": None,
        "emb_variance": None,
        "frontal_frac": None,
        "yaw_abs_med": None,
        "exemplars": [],
    }
    if n == 0:
        return out
    fidx = fdf["fidx"].to_numpy()
    X = E[fidx]
    norms = fdf["emb_norm"].to_numpy().astype(np.float64)
    q = fdf["face_score"].to_numpy().astype(np.float64) * np.clip(norms / p.norm_ref, 0.0, 1.5)
    yaw_abs = np.abs(fdf["yaw"].to_numpy().astype(np.float64))
    out["n_faces_good"] = int((norms >= p.emb_min_norm).sum())
    out["frontal_frac"] = float((yaw_abs <= p.frontal_yaw_max).mean())
    out["yaw_abs_med"] = float(np.median(yaw_abs))

    keep = np.arange(n)
    if n >= p.min_faces_pool:
        good = np.where(norms >= p.emb_min_norm)[0]
        base = good if len(good) >= p.min_faces_pool else keep
        m = base[medoid_index(X[base])]
        d_med = 1.0 - X @ X[m]
        keep = np.where(d_med <= p.trim_dist)[0]
        if len(keep) < p.min_faces_pool:
            keep = np.argsort(d_med)[: p.min_faces_pool]
        out["emb_variance"] = float(d_med[keep].mean())
    else:
        out["emb_variance"] = float((1.0 - X @ X[medoid_index(X)]).mean())
    Xk, qk = X[keep], q[keep]
    out["n_used"] = int(len(keep))
    out["emb_quality"] = float(qk.mean())

    K = int(np.clip(len(keep) // p.pose_min_per_k, 1, p.pose_k_max))
    if K >= 2:
        _, lab = kmeans2(Xk, K, minit="++", rng=0)
        means = [unit((Xk[lab == k] * qk[lab == k, None]).sum(axis=0)) for k in np.unique(lab)]
        centroid = unit(np.mean(means, axis=0))
    else:
        centroid = unit((Xk * qk[:, None]).sum(axis=0))
    out["embedding"] = centroid.astype(np.float32)

    frames = fdf["frame"].to_numpy()[keep]
    face_ids = fdf["face_id"].to_numpy()[keep]
    out["exemplars"] = [
        {"frame": int(frames[i]), "face_id": int(face_ids[i])}
        for i in farthest_point_exemplars(Xk, qk, p.k_exemplars)
    ]
    return out


def track_shot(
    shot: Raw,
    heads: pl.DataFrame,
    faces: pl.DataFrame,
    fps: float,
    times: clock.Clock,
    width: int,
    height: int,
    p: TrackParams,
) -> tuple[pl.DataFrame | None, list[TrackRow], Stats]:
    st: Stats = {"n_dup": 0, "n_split": 0, "n_stitch": 0, "n_tracks": 0}
    if heads.is_empty() and faces.is_empty():
        return None, [], st
    faces = faces.with_row_index("fidx").with_columns(c.fidx.cast(pl.Int64))
    E = as_matrix(faces["embedding"], EMB_DIM)
    norms = faces["emb_norm"].to_numpy().astype(np.float64)

    dets = fuse_detections(heads, faces, p)
    tracks = run_bytetrack(shot, dets, fps, (height, width), E, norms, p)
    if tracks.is_empty():
        return None, [], st
    tracked = attach_faces(tracks, faces, p)
    tracked, st["n_dup"] = merge_duplicate_tracks(tracked, p)
    tracked, split_touched, st["n_split"] = split_id_switches(tracked, E, p)
    tracked, stitch_map = stitch_tracklets(tracked, E, fps, p)
    tracked, renum = renumber(tracked)
    st["n_stitch"] = len(stitch_map) - len(set(stitch_map.values()))
    st["n_tracks"] = len(renum)

    members: dict[int, list[int]] = {}
    for old, root in stitch_map.items():
        members.setdefault(renum[root], []).append(old)

    rows: list[TrackRow] = []
    for key, df in sorted(tracked.partition_by("track_id", as_dict=True).items()):
        tid = key[0]
        first, last = _whole(df["frame"].min()), _whole(df["frame"].max())
        b = df.select("x1", "y1", "x2", "y2").to_numpy().astype(float)
        rows.append(
            {
                "track_id": tid,
                "sid": shot["sid"],
                "first_frame": first,
                "last_frame": last,
                "n_frames": df.height,
                "t0": float(times.at(first)),
                "t1": float(times.at(last + 1)),
                **pool_track(df, E, p),
                "cx": float(((b[:, 0] + b[:, 2]) / 2).mean() / width),
                "cy": float(((b[:, 1] + b[:, 3]) / 2).mean() / height),
                "h": float((b[:, 3] - b[:, 1]).mean() / height),
                "n_stitched": len(members.get(tid, [tid])),
                "was_split": any(m in split_touched for m in members.get(tid, [tid])),
            }
        )
    frames = tracked.select(
        "track_id",
        pl.lit(shot["sid"], dtype=pl.Int64).alias("sid"),
        "frame",
        "x1",
        "y1",
        "x2",
        "y2",
        c.score.alias("head_score"),
        *FACE_OUT,
    )
    return frames, rows, st


def build_tracks(
    shots: pl.DataFrame,
    faces: pl.DataFrame,
    heads: pl.DataFrame,
    fps: float,
    times: clock.Clock,
    width: int,
    height: int,
    p: TrackParams = PARAMS,
) -> tuple[pl.DataFrame, pl.DataFrame, Stats]:
    stats: Stats = {
        "n_shots": shots.height,
        "n_shots_tracked": 0,
        "n_tracks": 0,
        "n_dup": 0,
        "n_split": 0,
        "n_stitch": 0,
        "n_faces": faces.height,
        "n_faces_attached": 0,
        "n_with_embedding": 0,
    }
    frame_parts: list[pl.DataFrame] = []
    track_rows: list[TrackRow] = []
    if shots.height:
        faces = assign_shots(faces, shots)
        heads = assign_shots(heads, shots)
    else:
        faces = faces.with_columns(sid=pl.lit(None, pl.Int64))
    if shots.height and (faces.height or heads.height):
        faces_by = faces.partition_by("sid", as_dict=True)
        heads_by = heads.partition_by("sid", as_dict=True)
        offset = 0
        for shot in shots.iter_rows(named=True):
            key = (shot["sid"],)
            frames, rows, st = track_shot(
                shot,
                heads_by.get(key, heads.clear()),
                faces_by.get(key, faces.clear()),
                fps,
                times,
                width,
                height,
                p,
            )
            for k in ("n_dup", "n_split", "n_stitch", "n_tracks"):
                stats[k] += st[k]
            if frames is None:
                continue
            stats["n_shots_tracked"] += 1
            frame_parts.append(frames.with_columns(track_id=c.track_id + offset))
            for r in rows:
                r["track_id"] = _whole(r["track_id"]) + offset
            track_rows += rows
            offset += st["n_tracks"]

    def column(name: str, dtype: pl.DataType | type[pl.DataType]) -> pl.Series:
        vals = [r[name] for r in track_rows]
        if isinstance(dtype, pl.Array):
            lists = [v.tolist() if isinstance(v, np.ndarray) else v for v in vals]
            return pl.Series(name, lists, dtype=pl.List(dtype.inner)).cast(dtype)
        return pl.Series(name, vals, dtype=dtype)

    tracks = (
        pl.DataFrame([column(k, t) for k, t in TRACKS_SCHEMA.items()])
        if track_rows
        else pl.DataFrame(schema=TRACKS_SCHEMA)
    )

    attached = pl.concat(frame_parts) if frame_parts else pl.DataFrame(schema=FRAMES_SCHEMA)
    claimed = attached.filter(c.face_id.is_not_null()).select("frame", "face_id")
    orphans = faces.join(claimed, on=["frame", "face_id"], how="anti").select(
        pl.lit(None, pl.Int64).alias("track_id"),
        "sid",
        "frame",
        *[pl.lit(None, pl.Float32).alias(n) for n in ("x1", "y1", "x2", "y2", "head_score")],
        "face_id",
        c.x1.alias("fx1"),
        c.y1.alias("fy1"),
        c.x2.alias("fx2"),
        c.y2.alias("fy2"),
        c.score.alias("face_score"),
        "emb_norm",
        "kps",
    )
    frames = pl.concat(
        [
            attached.select(list(FRAMES_SCHEMA)).cast(pl.Schema(FRAMES_SCHEMA)),
            orphans.cast(pl.Schema(FRAMES_SCHEMA)),
        ]
    ).sort("frame", "track_id", nulls_last=True)
    stats["n_faces_attached"] = int(claimed.height)
    stats["n_with_embedding"] = int(tracks["embedding"].is_not_null().sum())
    return tracks, frames, stats


def video_props(ip: Path | None, od: Path) -> tuple[float, int, int]:
    fps, width, height = 0.0, 0, 0
    if ip is not None and Path(ip).exists():
        import cv2

        cap = cv2.VideoCapture(str(ip))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    elif (od / "metadata.parquet").exists():
        meta = pl.read_parquet(
            od / "metadata.parquet", columns=["fps", "img_width", "img_height"]
        ).row(0)
        fps, width, height = float(meta[0]), int(meta[1]), int(meta[2])
    else:
        raise FileNotFoundError(
            f"{od.name}: no video and no metadata.parquet to read fps and frame size from"
        )
    if not fps or not np.isfinite(fps) or fps <= 0:
        log_status(od, f"frame rate unusable ({fps!r}); using {FALLBACK_FPS}")
        fps = FALLBACK_FPS
    if width <= 0 or height <= 0:
        raise ValueError(f"{od.name}: frame size {width}x{height} unusable")
    return float(fps), width, height


def describe(stats: Stats) -> str:
    n_f = stats["n_faces"]
    pct = 100.0 * stats["n_faces_attached"] / n_f if n_f else 0.0
    return (
        f"{stats['n_tracks']} tracks in {stats['n_shots_tracked']}/{stats['n_shots']} shots; "
        f"{stats['n_dup']} duplicate merges, {stats['n_split']} splits, "
        f"{stats['n_stitch']} stitches; "
        f"{stats['n_faces_attached']}/{n_f} faces on a track ({pct:.1f}%); "
        f"{stats['n_with_embedding']} tracks with an embedding"
    )


def write_outputs(
    od: Path,
    tracks: pl.DataFrame,
    frames: pl.DataFrame,
    fps: float,
    width: int,
    height: int,
    p: TrackParams,
) -> None:
    meta = {
        NAME: json.dumps(
            {"params": asdict(p), "fps": fps, "img_width": width, "img_height": height}
        )
    }
    stamp(tracks, NAME).write_parquet(od / OUTPUTS[0], metadata=meta)
    stamp(frames, NAME).write_parquet(od / OUTPUTS[1], metadata=meta)


def create_tracks(ip: Path | None, od: StrPath) -> None:
    od = Path(od)
    if all((od / fn).exists() for fn in OUTPUTS):
        log_status(od, "Already exists")
        return
    absent = [fn for fn in INPUTS if not (od / fn).exists()]
    if absent:
        log_status(od, f"No {absent[0]} — skipped")
        return

    t0 = time.monotonic()
    fps, width, height = video_props(ip, od)
    times = clock.of(od, ip, fps)
    if times.source != clock.FROM_FRAMES:
        log_status(od, f"frames.parquet has no t; times from {times.source}")
    shots, faces, heads = load_inputs(od)
    log_status(
        od, f"{shots.height} shots, {faces.height} faces, {heads.height} heads at {fps:.3f} fps"
    )
    tracks, frames, stats = build_tracks(shots, faces, heads, fps, times, width, height, PARAMS)
    write_outputs(od, tracks, frames, fps, width, height, PARAMS)
    log_status(od, describe(stats))
    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    log_status(od, f"Finished ({h:02d}:{m:02d}:{s:02d})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build head/face tracks for one annotation directory.")
    ap.add_argument("od", type=Path, help="annotation directory holding shots/face/yolo parquets")
    ap.add_argument(
        "--video",
        type=Path,
        default=None,
        help="the film, for fps and frame size; defaults to metadata.parquet in the directory",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()
    if args.force:
        for fn in OUTPUTS:
            (args.od / fn).unlink(missing_ok=True)
    create_tracks(args.video, args.od)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
