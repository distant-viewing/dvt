"""The cpu annotators that sample frames: letterbox, chroma, and the aspect
the metadata stage works out from them."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import make_video, needs_ffmpeg
from hypothesis import given
from hypothesis import strategies as st

from dvt import aspect, chroma, letterbox

# ---- aspect -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("4:3", 4 / 3), ("16:9", 16 / 9), ("1:1", 1.0), (None, None), ("4/3", None),
     ("0:1", None), ("a:b", None), ("-4:3", None), (1.5, None)],
)  # fmt: skip
def test_parse_ratio(text: object, expected: float | None) -> None:
    assert aspect.parse_ratio(text) == expected


def test_nearest_standard() -> None:
    near, dist = aspect.nearest_standard(1.80)
    assert (near, round(dist, 6)) == (1.78, 0.02)


def test_aspect_without_a_container() -> None:
    rec = aspect.aspect_fields(4 / 3, 640, 480, None)
    assert rec == {"sar_ratio": None, "ar_known": False, "display_ar": 4 / 3,
                   "ar_nearest": 1.33, "ar_dist": pytest.approx(4 / 3 - 1.33),
                   "ar_break": False, "ar_note": "not in metadata.csv"}  # fmt: skip


def test_anamorphic() -> None:
    container = {"sar": "32:27", "dar": "16:9", "width": 720, "height": 480}
    rec = aspect.aspect_fields(720 / 480, 720, 480, container)
    assert rec["ar_known"]
    assert rec["sar_ratio"] == pytest.approx(32 / 27)
    assert rec["display_ar"] == pytest.approx(16 / 9)
    assert rec["ar_nearest"] == 1.78
    assert rec["ar_note"] == ""


def test_sar_from_dar() -> None:
    rec = aspect.aspect_fields(1.5, 720, 480, {"sar": None, "dar": "16:9", "width": 720,
                                               "height": 480})  # fmt: skip
    assert rec["sar_ratio"] == pytest.approx((16 / 9) / 1.5)
    assert rec["ar_note"] == "sar derived from dar"


def test_inconsistent_container() -> None:
    rec = aspect.aspect_fields(1.5, 720, 480, {"sar": "1:1", "dar": "16:9", "width": 720,
                                               "height": 480})  # fmt: skip
    assert rec["ar_break"]
    assert not rec["ar_known"]
    assert rec["ar_note"] == "dar 1.7778 != (w/h)*sar 1.5000"


def test_other_dimensions() -> None:
    rec = aspect.aspect_fields(1.5, 720, 480, {"sar": "1:1", "dar": "3:2", "width": 640,
                                               "height": 480})  # fmt: skip
    assert rec["ar_note"] == "dims 720x480 != metadata.csv 640x480"
    assert not rec["ar_known"]


def test_no_usable_sar() -> None:
    rec = aspect.aspect_fields(1.5, 720, 480, {"sar": None, "dar": None})
    assert rec["ar_note"] == "no usable sar"


@given(st.floats(0.5, 3.0), st.integers(16, 4000), st.integers(16, 4000))
def test_square_pixels_keep_the_content_aspect(ar: float, w: int, h: int) -> None:
    rec = aspect.aspect_fields(ar, w, h, {"sar": "1:1", "dar": None, "width": w, "height": h})
    assert rec["display_ar"] == pytest.approx(ar)
    assert rec["ar_dist"] == pytest.approx(abs(rec["ar_nearest"] - ar))
    assert rec["ar_nearest"] in aspect.STANDARD_AR


# ---- letterbox profiles ---------------------------------------------------------


def profile(n: int, length: int, near: int, far: int, level: int = 120) -> np.ndarray:
    """n identical profiles of `length` with `near` and `far` dark pixels at
    the two ends."""
    p = np.full((n, length), level, dtype=np.uint8)
    p[:, :near] = 0
    if far:
        p[:, length - far :] = 0
    return p


def test_edge_runs() -> None:
    near, far, lit = letterbox.edge_runs(profile(3, 100, 10, 20), letterbox.THR)
    assert near.tolist() == [10, 10, 10]
    assert far.tolist() == [20, 20, 20]
    assert lit.all()
    near, far, lit = letterbox.edge_runs(np.zeros((2, 50), np.uint8), letterbox.THR)
    assert not lit.any()


def test_edge_runs_pass_over_a_specks() -> None:
    p = profile(1, 200, 30, 0)
    p[0, 5] = 200
    near, _, _ = letterbox.edge_runs(p, letterbox.THR)
    assert near.tolist() == [30]


@given(st.integers(0, 40), st.integers(0, 40))
def test_edge_runs_find_any_bars(near_px: int, far_px: int) -> None:
    near, far, lit = letterbox.edge_runs(profile(2, 120, near_px, far_px), letterbox.THR)
    assert lit.all()
    assert near.tolist() == [near_px] * 2
    assert far.tolist() == [far_px] * 2


def test_block_spread() -> None:
    assert letterbox.block_spread(np.full(40, 7)) == 0.0
    assert letterbox.block_spread(np.r_[np.zeros(20), np.full(20, 6)]) == 6.0
    assert letterbox.block_spread(np.array([3])) == 0.0


def test_crop_from_profiles() -> None:
    rows, cols = profile(60, 120, 12, 14), profile(60, 160, 0, 0)
    rec = letterbox.crop_from_profiles(rows, cols, 160, 120, 64)
    assert (rec["crop_top"], rec["crop_bottom"], rec["crop_left"], rec["crop_right"]) == (
        12, 14, 0, 0)  # fmt: skip
    assert rec["crop_usable"]
    assert rec["crop_stable"]
    assert not rec["crop_union"]
    assert (rec["crop_n_sampled"], rec["crop_n_usable"]) == (64, 60)
    assert rec["crop_dead_frac"] == pytest.approx(1 - 60 / 64)
    assert rec["crop_asym_px"] == 2
    assert not rec["crop_odd"]
    assert rec["content_ar"] == pytest.approx(160 / 94)


def test_crop_that_moves() -> None:
    rows = np.concatenate([profile(30, 120, 10, 10), profile(30, 120, 20, 20)])
    rec = letterbox.crop_from_profiles(rows, profile(60, 160, 0, 0), 160, 120)
    assert rec["crop_union"]
    assert not rec["crop_stable"]
    assert rec["crop_drift_px"] == 10.0
    assert rec["crop_top"] == 10


def test_too_few_frames() -> None:
    rec = letterbox.crop_from_profiles([], [], 160, 120)
    assert not rec["crop_usable"]
    assert rec["crop_dead_frac"] == 1.0
    assert rec["content_ar"] == pytest.approx(160 / 120)


def read(
    frame: int = 0,
    dead: bool = False,
    f_chroma: float = 0.0,
    f_sat: float = 0.0,
    row_mean: list[int] | None = None,
    col_mean: list[int] | None = None,
) -> letterbox.Read:
    """A sampled frame as sample_evidence records one it could read."""
    return {"frame": frame, "ok": True, "f_min": 0, "f_max": 255, "f_mean": 0.5, "f_std": 0.2,
            "f_chroma": f_chroma, "f_sat": f_sat, "dead": dead, "row_max": [], "col_max": [],
            "row_mean": row_mean or [], "col_mean": col_mean or []}  # fmt: skip


UNREAD: letterbox.Unread = {"frame": 0, "ok": False}


def test_profiles_from_rows() -> None:
    assert letterbox.profiles_from_rows([UNREAD, read(dead=True)]) == (None, None)
    rows, cols = letterbox.profiles_from_rows([read(row_mean=[1, 2], col_mean=[3])])
    assert rows is not None
    assert cols is not None
    assert (rows.tolist(), cols.tolist(), rows.dtype) == ([[1, 2]], [[3]], np.uint8)


# ---- on a video -----------------------------------------------------------------


@needs_ffmpeg
def test_sample_evidence(tmp_path: Path) -> None:
    video = make_video(tmp_path / "v.mp4", seconds=6, bars=(16, 16, 0, 0))
    rows, meta = letterbox.sample_evidence(video, samples=12)
    assert meta == {"n_frames": 60, "fps": 10.0, "img_width": 160, "img_height": 152}
    assert rows is not None
    assert len(rows) == 12
    first = rows[0]
    assert first["ok"] is True
    assert not first["dead"]
    assert len(first["row_mean"]) == 152
    assert len(first["col_mean"]) == 160
    assert max(first["row_mean"][:14]) < letterbox.THR < min(first["row_mean"][20:130])


@needs_ffmpeg
def test_sample_evidence_of_junk(tmp_path: Path) -> None:
    junk = tmp_path / "v.mp4"
    junk.write_text("no")
    assert letterbox.sample_evidence(junk) == (None, {})


@needs_ffmpeg
def test_create_letterbox_and_chroma(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = make_video(tmp_path / "v.mp4", seconds=6, bars=(16, 16, 0, 0))
    od = tmp_path / "out" / "Film"
    letterbox.create_letterbox(video, od, samples=20)
    df = pl.read_parquet(od / "letterbox.parquet")
    assert df.height == 20
    assert set(df["fname"]) == {"Film"}
    assert df["img_height"][0] == 152
    assert df["version"][0] == 1
    chroma.create_chroma(video, od)
    c = pl.read_parquet(od / "chroma.parquet").row(0, named=True)
    assert c["fname"] == "Film"
    # chroma reads the frames letterbox sampled, kept by the video's path.
    assert c["n_usable"] == 20
    assert not c["bw"]
    out = capsys.readouterr().out
    assert "[Film:letterbox] 20 sampled, 20 usable" in out
    assert "[Film:chroma] colour" in out
    letterbox.create_letterbox(video, od)
    chroma.create_chroma(video, od)
    assert capsys.readouterr().out == (
        "[Film:letterbox] Already exists\n[Film:chroma] Already exists\n"
    )


@needs_ffmpeg
def test_chroma_of_a_grey_film(tmp_path: Path) -> None:
    video = make_video(tmp_path / "g.mp4", seconds=3, gray=True)
    od = tmp_path / "out" / "Grey"
    chroma.create_chroma(video, od)
    assert pl.read_parquet(od / "chroma.parquet")["bw"].to_list() == [True]


def test_missing_film(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    od = tmp_path / "Film"
    letterbox.create_letterbox(tmp_path / "none.mp4", od)
    chroma.create_chroma(tmp_path / "none.mp4", od)
    assert capsys.readouterr().out.count("Missing film") == 2
    assert not od.exists()


def test_summarise() -> None:
    assert chroma.summarise([]) == {"n_sampled": 0, "n_usable": 0, "chroma": 0.0,
                                    "chroma_p90": 0.0, "sat": 0.0}  # fmt: skip
    rows: list[letterbox.Sample] = [read(f_chroma=float(i), f_sat=0.5) for i in range(11)]
    rows.append(UNREAD)
    rec = chroma.summarise(rows)
    assert (rec["n_sampled"], rec["n_usable"], rec["chroma"], rec["sat"]) == (12, 11, 5.0, 0.5)
    assert rec["chroma_p90"] == pytest.approx(9.0)
