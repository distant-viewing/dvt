"""Frames to shots, as a pure function of a dataframe and a parameter set."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import col as c

from .clock import Clock


@dataclass(frozen=True)
class Params:
    sfp_t: float = 0.5
    peak_window: int = 13

    spike_lo: float | None = 0.30
    spike_r: int = 12
    spike_q: float = 0.25
    spike_w: int = 9

    flat_std: float = 0.0001

    dark_mean: float = 0.04

    dark_max_len: int | None = 48
    dark_std: float | None = None

    fade_min_ramp: int = 3
    ramp_level: float = 0.5
    ramp_report: float = 0.9

    afp_t: float | None = 0.45
    afp_min_len: int = 4

    fade_lstd: float = 0.05

    fade_win: int = 24

    tag_win: int = 12

    confident_sfp: float = 0.60

    sfp_col: str = "sfp"
    afp_col: str = "afp"
    mean_col: str = "mean"
    std_col: str = "std"
    lstd_col: str = "lum_std"


Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]


def spike_ok(sfp: Floats, i: int, p: "Params") -> bool:
    if p.spike_lo is None or sfp[i] < p.spike_lo:
        return False
    a, b = max(0, i - p.spike_r), min(len(sfp), i + p.spike_r + 1)
    return int((sfp[a:b] >= p.spike_q * sfp[i]).sum()) <= p.spike_w


def dissolve_midpoints(
    afp: Floats, is_boundary: Bools, afp_t: float, min_len: int
) -> tuple[Ints, Ints]:
    hot = np.asarray(afp > afp_t, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], hot, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    keep = (ends - starts) >= min_len
    starts, ends = starts[keep], ends[keep]

    cum = np.concatenate(([0], np.cumsum(np.asarray(is_boundary, dtype=np.int64))))
    clean = (cum[ends] - cum[starts]) == 0
    starts, ends = starts[clean], ends[clean]

    return (starts + ends) // 2, ends - starts


def rule_tiers(
    sfp: Floats, is_max: Bools, afp: Floats | None, p: "Params"
) -> tuple[Ints, Ints, Ints, Ints]:
    t1 = np.flatnonzero(is_max & (sfp >= p.sfp_t))

    t2 = np.zeros(0, dtype=np.int64)
    if p.spike_lo is not None:
        band = np.flatnonzero(is_max & (sfp >= p.spike_lo) & (sfp < p.sfp_t))
        t2 = np.array([i for i in band if spike_ok(sfp, int(i), p)], dtype=np.int64)

    t3 = lens = np.zeros(0, dtype=np.int64)
    if p.afp_t is not None:
        if afp is None:
            raise ValueError("the gradual rule needs the afp column")
        seen = np.zeros(len(sfp), dtype=bool)
        seen[t1] = True
        t3, lens = dissolve_midpoints(afp, seen, p.afp_t, p.afp_min_len)

    return t1, t2, t3, lens


def segment(frames: pl.DataFrame, times: Clock, p: Params = Params()) -> pl.DataFrame:
    """`times` is the film's clock.Clock, which turns frame numbers into seconds."""
    missing = {p.sfp_col, p.mean_col, p.std_col} - set(frames.columns)
    if missing:
        raise ValueError(f"frames is missing {sorted(missing)}")
    if p.afp_t is not None:
        need = [k for k in (p.afp_col, p.lstd_col) if k not in frames.columns]
        if need:
            raise ValueError(
                f"frames is missing {need}, needed for the gradual rule. A "
                f"frames.parquet written before 2026-08-10 has neither these "
                f"columns nor the cropped TransNetV2 input every threshold "
                f"here is calibrated on; delete it and rerun the frames stage."
            )

    df = (
        frames.select(
            c.frame,
            pl.col(p.mean_col).alias("_mean"),
            pl.col(p.std_col).alias("_std"),
            pl.col(p.sfp_col).alias("_sfp"),
            pl.col(p.sfp_col).alias("_raw"),
            *(
                [pl.col(p.afp_col).alias("_afp"), pl.col(p.lstd_col).alias("_lstd")]
                if p.afp_t is not None
                else []
            ),
        )
        .with_columns(
            _sfp=pl.when(
                ((c._std < p.flat_std) & (c._std.shift(1).fill_null(1.0) >= p.flat_std))
                | ((c._std < p.flat_std) & (c._std.shift(-1).fill_null(1.0) >= p.flat_std))
            )
            .then(2)
            .otherwise(c._sfp)
        )
        .with_columns(
            _local_max=c._sfp.rolling_max(window_size=p.peak_window, center=True, min_samples=1)
        )
        .with_columns(
            _is_max=c._sfp == c._local_max,
        )
    )

    t1, t2, t3, t3_len = rule_tiers(
        df["_sfp"].to_numpy(),
        df["_is_max"].to_numpy(),
        df["_afp"].to_numpy() if p.afp_t is not None else None,
        p,
    )
    tier = np.zeros(df.height, dtype=np.int8)
    tier[t1], tier[t2], tier[t3] = 1, 2, 3
    dis = np.zeros(df.height, dtype=np.int64)
    dis[t3] = t3_len
    df = df.with_columns(
        _tier=pl.Series(tier),
        _dis=pl.Series(dis),
        _bound=pl.Series(tier > 0),
    )

    win = 2 * p.tag_win + 1
    scored = (c._tier == 2) | ((c._tier == 1) & (c._raw >= p.sfp_t))
    kind = pl.when(scored).then(pl.lit("cut"))
    if p.afp_t is not None:
        half = (np.asarray(dis) + 2 * p.fade_win) // 2
        lstd = df[p.lstd_col if p.lstd_col in df.columns else "_lstd"].to_numpy()
        nfr = len(lstd)
        lo = np.maximum(0, np.arange(nfr) - half)
        hi = np.minimum(nfr, np.arange(nfr) + half + 1)
        lmin = np.array([lstd[a:b].min() for a, b in zip(lo, hi, strict=True)])
        df = df.with_columns(_lstd_min=pl.Series(lmin))
        kind = kind.when(c._tier == 3).then(
            pl.when(c._lstd_min <= p.fade_lstd).then(pl.lit("fade")).otherwise(pl.lit("dissolve"))
        )

    df = df.with_columns(
        _peak=c._raw.rolling_max(window_size=win, center=True, min_samples=1)
    ).with_columns(
        _kind=kind.otherwise(pl.lit(None, dtype=pl.String)),
        _conf=pl.when((c._tier == 1) & (c._raw >= p.sfp_t))
        .then(
            pl.when(c._peak >= p.confident_sfp)
            .then(pl.lit("confident"))
            .otherwise(pl.lit("probable"))
        )
        .when(c._tier > 1)
        .then(pl.lit("borderline"))
        .otherwise(pl.lit(None, dtype=pl.String)),
    )

    shots = (
        df.with_columns(segment=c._bound.cum_sum())
        .group_by(c.segment, maintain_order=True)
        .agg(
            start_frame=c.frame.min(),
            end_frame=c.frame.max(),
            start_ramp=(c._std < p.ramp_level * c._std.median()).cast(pl.Int32).cum_prod().sum(),
            end_ramp=(c._std.reverse() < p.ramp_level * c._std.median())
            .cast(pl.Int32)
            .cum_prod()
            .sum(),
            start_ramp_t=(c._std < p.ramp_report * c._std.median()).cast(pl.Int32).cum_prod().sum(),
            end_ramp_t=(c._std.reverse() < p.ramp_report * c._std.median())
            .cast(pl.Int32)
            .cum_prod()
            .sum(),
            mean_max=c._mean.max(),
            std_max=c._std.max(),
            trigger_sfp=c._sfp.first(),
            trigger_dis=c._dis.first(),
            trigger_kind=c._kind.first(),
            trigger_conf=c._conf.first(),
        )
        .sort("start_frame")
        .drop("segment")
    )

    shots = shots.with_columns(
        shot_type=pl.when(c.std_max > p.flat_std)
        .then(pl.lit("normal"))
        .otherwise(pl.lit("blackout"))
    )

    absorb = (
        (c.shot_type != "blackout")
        & (c.mean_max < p.dark_mean)
        & ((c.shot_type.shift(1) == "blackout") | (c.shot_type.shift(-1) == "blackout"))
    )
    if p.dark_max_len is not None:
        absorb = absorb & ((c.end_frame - c.start_frame) <= p.dark_max_len)
    if p.dark_std is not None:
        absorb = absorb & (c.std_max < p.dark_std)

    shots = shots.with_columns(
        shot_type=pl.when(absorb).then(pl.lit("blackout")).otherwise(c.shot_type)
    )

    shots = (
        shots.with_columns(
            merge_group=(
                (c.shot_type != c.shot_type.shift(1)).fill_null(True) | (c.shot_type != "blackout")
            ).cum_sum()
        )
        .group_by(["merge_group", "shot_type"], maintain_order=True)
        .agg(
            start_frame=c.start_frame.min(),
            end_frame=c.end_frame.max(),
            start_ramp=c.start_ramp.first(),
            end_ramp=c.end_ramp.last(),
            start_ramp_t=c.start_ramp_t.first(),
            end_ramp_t=c.end_ramp_t.last(),
            mean_max=c.mean_max.max(),
            std_max=c.std_max.max(),
            trigger_sfp=c.trigger_sfp.first(),
            trigger_dis=c.trigger_dis.first(),
            trigger_kind=c.trigger_kind.first(),
            trigger_conf=c.trigger_conf.first(),
        )
        .sort("start_frame")
        .drop("merge_group")
    )
    shots = shots.with_columns(
        start_time=pl.Series(times.at(shots["start_frame"].to_numpy()), dtype=pl.Float64),
        end_time=pl.Series(times.at(shots["end_frame"].to_numpy() + 1), dtype=pl.Float64),
    )

    shots = (
        shots.with_columns(
            trans_in=pl.when(
                (c.trigger_sfp == 2)
                & (c.shot_type != "blackout")
                & (c.start_ramp >= p.fade_min_ramp)
            )
            .then(pl.lit("fade-in"))
            .otherwise(pl.lit("hard")),
            trans_out=pl.when(
                (c.trigger_sfp.shift(-1) == 2)
                & (c.shot_type != "blackout")
                & (c.end_ramp >= p.fade_min_ramp)
            )
            .then(pl.lit("fade-out"))
            .otherwise(pl.lit("hard")),
        )
        .with_columns(
            trans_in=pl.when((c.trigger_dis > 0) & (c.shot_type != "blackout"))
            .then(pl.lit("dissolve"))
            .otherwise(c.trans_in),
            trans_out=pl.when((c.trigger_dis.shift(-1) > 0) & (c.shot_type != "blackout"))
            .then(pl.lit("dissolve"))
            .otherwise(c.trans_out),
        )
        .with_columns(
            trans_in=pl.when(c.shot_type == "blackout")
            .then(pl.lit("blackout"))
            .otherwise(c.trans_in),
            trans_out=pl.when(c.shot_type == "blackout")
            .then(pl.lit("blackout"))
            .otherwise(c.trans_out),
            t_in=pl.when(c.trans_in == "fade-in")
            .then(c.start_ramp_t)
            .when(c.trans_in == "dissolve")
            .then(c.trigger_dis)
            .otherwise(0),
            t_out=pl.when(c.trans_out == "fade-out")
            .then(c.end_ramp_t)
            .when(c.trans_out == "dissolve")
            .then(c.trigger_dis.shift(-1))
            .otherwise(0),
            sid=pl.int_range(pl.len()),
        )
        .select(
            c.sid,
            c.start_frame,
            c.end_frame,
            c.start_time,
            c.end_time,
            c.shot_type,
            c.trans_in,
            c.trans_out,
            c.t_in,
            c.t_out,
            trans_kind=c.trigger_kind,
            confidence=c.trigger_conf,
        )
    )

    return shots.with_columns(
        fmid=(c.start_frame + c.end_frame) // 2,
        kframes=pl.concat_list(
            [
                c.start_frame,
                (c.start_frame + c.end_frame) // 2,
                c.end_frame,
            ]
        ),
    ).rename(
        {
            "start_frame": "f0",
            "end_frame": "f1",
            "start_time": "t0",
            "end_time": "t1",
        }
    )
