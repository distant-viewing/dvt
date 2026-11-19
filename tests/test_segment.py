"""Frames to shots: the rules on a made-up frames table."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dvt import clock
from dvt.segment import Params, dissolve_midpoints, rule_tiers, segment, spike_ok


def frames(n: int, **cols: np.ndarray) -> pl.DataFrame:
    base = {"frame": np.arange(n), "sfp": np.zeros(n), "afp": np.zeros(n), "mean": np.full(n, 0.5),
            "std": np.full(n, 0.2), "lum_std": np.full(n, 0.2)}  # fmt: skip
    base.update(cols)
    return pl.DataFrame(base)


def at(n: int, **values: float) -> np.ndarray:
    out = np.zeros(n)
    for k, v in values.items():
        out[int(k[1:])] = v
    return out


FPS = clock.Clock(None, 25.0, clock.FROM_FPS)


def test_hard_cuts() -> None:
    shots = segment(frames(100, sfp=at(100, f30=0.9, f60=0.55)), FPS)
    assert shots.select("sid", "f0", "f1", "trans_kind", "confidence").rows() == [
        (0, 0, 29, None, None),
        (1, 30, 59, "cut", "confident"),
        (2, 60, 99, "cut", "probable"),
    ]
    assert shots["t0"].to_list() == pytest.approx([0.0, 1.2, 2.4])
    assert shots["t1"].to_list() == pytest.approx([1.2, 2.4, 4.0])
    assert shots["kframes"].to_list()[1] == [30, 44, 59]
    assert shots["fmid"].to_list() == [14, 44, 79]


def test_spike_tier() -> None:
    sfp = at(100, f40=0.35)
    shots = segment(frames(100, sfp=sfp), FPS)
    assert shots.select("f0", "confidence").rows() == [(0, None), (40, "borderline")]
    p = Params()
    assert spike_ok(sfp, 40, p)
    busy = sfp + 0.2
    assert not spike_ok(busy, 40, p)


def test_dissolve_and_fade() -> None:
    n = 200
    afp = np.zeros(n)
    afp[50:60] = 0.8
    afp[140:150] = 0.8
    lum = np.full(n, 0.2)
    lum[140:150] = 0.01
    shots = segment(frames(n, afp=afp, lum_std=lum), FPS)
    assert shots.select("f0", "trans_kind", "trans_in", "t_in").rows() == [
        (0, None, "hard", 0),
        (55, "dissolve", "dissolve", 10),
        (145, "fade", "dissolve", 10),
    ]


def test_blackout() -> None:
    """A flat stretch is a blackout; its last flat frame opens the next shot,
    and a cut that sharp has no ramp to call a fade."""
    n = 120
    std = np.full(n, 0.2)
    std[40:60] = 0.0
    mean = np.full(n, 0.5)
    mean[40:60] = 0.0
    shots = segment(frames(n, std=std, mean=mean), FPS)
    assert shots.select("f0", "f1", "shot_type", "trans_in", "trans_out").rows() == [
        (0, 39, "normal", "hard", "hard"),
        (40, 58, "blackout", "blackout", "blackout"),
        (59, 119, "normal", "hard", "hard"),
    ]


def test_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        segment(frames(10).drop("sfp"), FPS)
    with pytest.raises(ValueError, match="gradual rule"):
        segment(frames(10).drop("afp"), FPS)
    shots = segment(frames(10).drop("afp", "lum_std"), FPS, Params(afp_t=None))
    assert shots.height == 1


def test_dissolve_midpoints() -> None:
    afp = np.array([0, 0.9, 0.9, 0.9, 0.9, 0, 0.9, 0.9, 0, 0.9, 0.9, 0.9, 0.9, 0.9])
    bound = np.zeros(len(afp), dtype=bool)
    bound[11] = True
    mids, lens = dissolve_midpoints(afp, bound, 0.45, 4)
    assert (mids.tolist(), lens.tolist()) == ([3], [4])


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=2, max_size=300),
       st.lists(st.floats(0, 1), min_size=2, max_size=300))  # fmt: skip
def test_shots_cover_every_frame_once(sfp: list[float], afp: list[float]) -> None:
    n = min(len(sfp), len(afp))
    shots = segment(frames(n, sfp=np.array(sfp[:n]), afp=np.array(afp[:n])), FPS)
    assert shots["sid"].to_list() == list(range(shots.height))
    assert shots["f0"][0] == 0
    assert shots["f1"][-1] == n - 1
    assert (shots["f0"][1:].to_numpy() == shots["f1"][:-1].to_numpy() + 1).all()
    assert (shots["t1"].to_numpy() > shots["t0"].to_numpy()).all()
    t1, t2, _, _ = rule_tiers(np.array(sfp[:n]), np.ones(n, dtype=bool),
                               np.array(afp[:n]), Params())  # fmt: skip
    assert set(t1.tolist()).isdisjoint(t2.tolist())
