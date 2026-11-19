"""The shots stage: the frames cut into shots, and what each shot looks like."""

import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from polars import col as c

from . import clock, frame_ops
from .annotators import stamp
from .log import log_finished, log_status
from .segment import segment
from .tools.values import StrPath

MIN_DRESP = frame_ops.MIN_DRESP


def create_shots(ip: StrPath, od: Path) -> None:

    shots_path = od / "shots.parquet"
    if shots_path.exists():
        log_status(od, "shots", "Already exists")
        return

    t0 = time.monotonic()
    frames = pl.read_parquet(od / "frames.parquet")

    vinput = cv2.VideoCapture(str(ip))
    fps = vinput.get(cv2.CAP_PROP_FPS)
    vinput.release()

    times = clock.of(od, ip, fps)
    if times.source != clock.FROM_FRAMES:
        log_status(od, "shots", f"frames.parquet has no t; times from {times.source}")
    shots = segment(frames, times)

    shots = add_shot_metrics(shots, frames)

    stamp(shots, "shots").write_parquet(shots_path)
    log_finished(od, "shots", t0)


SHOT_METRIC_INPUTS = ("dabs", "dx", "dy", "dresp", "edge", "sat", "lab_a", "lab_b", "hist16")


def add_shot_metrics(shots: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    missing = [k for k in SHOT_METRIC_INPUTS if k not in frames.columns]
    if missing:
        raise ValueError(
            f"frames.parquet predates the shot metrics (missing {missing}). "
            f"Delete it and rerun the frames stage."
        )

    fr = frames.sort("frame").join_asof(
        shots.select(c.sid, c.f0).sort("f0"), left_on="frame", right_on="f0", strategy="backward"
    )

    look = fr.group_by("sid").agg(
        edge=c.edge.mean(),
        sat=c.sat.mean(),
        lab_a=c.lab_a.mean(),
        lab_b=c.lab_b.mean(),
    )

    move = (
        fr.filter(c.frame > c.f0)
        .group_by("sid")
        .agg(
            vai=c.dabs.mean(),
            vai_p95=c.dabs.quantile(0.95),
            cam_dx=c.dx.filter(c.dresp > MIN_DRESP).mean(),
            cam_dy=c.dy.filter(c.dresp > MIN_DRESP).mean(),
            cam_frac=(c.dresp > MIN_DRESP).mean(),
        )
    )

    hist = np.stack(list(frames.sort("frame")["hist16"].to_numpy()))
    lum = frames.sort("frame")["mean"].to_numpy()
    f0 = shots["f0"].to_numpy()
    prev = np.maximum(f0 - 1, 0)
    hist_dist = np.abs(hist[f0] - hist[prev]).sum(1)
    lum_jump = np.abs(lum[f0] - lum[prev])
    hist_dist[0] = lum_jump[0] = np.nan

    return (
        shots.with_columns(
            hist_dist_in=pl.Series(hist_dist, dtype=pl.Float32),
            lum_jump_in=pl.Series(lum_jump, dtype=pl.Float32),
        )
        .join(look, on="sid", how="left")
        .join(move, on="sid", how="left")
        .sort("sid")
    )
