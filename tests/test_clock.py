"""Times in seconds must follow the container, not frame / fps.

The clip made here has frames of two durations and a video stream that starts
a tenth of a second into the file, the two things the corpus holds that
frame / fps gets wrong.
"""

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from dvt import clock
from dvt.segment import segment
from dvt.tools import ffprobe, mp4

LATE = 0.1
FAST, SLOW = 1 / 30, 1 / 15
N_FAST = 30
CUT = 40


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clock") / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=d=3:r=30:s=160x120",
            "-f",
            "lavfi",
            "-i",
            "sine=d=3",
            "-vf",
            f"select='lt(n,{N_FAST})+not(mod(n,2))',setpts=PTS+{LATE}/TB",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def expected(n: int) -> NDArray[np.float64]:
    gaps = np.where(np.arange(n - 1) < N_FAST, FAST, SLOW)
    return LATE + np.concatenate(([0.0], np.cumsum(gaps)))


def test_frame_times_follow_the_container(clip: Path) -> None:
    t = np.array(ffprobe.frame_times(clip))
    assert len(t) > CUT
    assert np.allclose(t, expected(len(t)), atol=1e-3)
    assert ffprobe.start(clip, "video") == pytest.approx(LATE, abs=1e-3)
    assert ffprobe.start(clip, "audio") == pytest.approx(0.0, abs=1e-3)


def test_the_index_gives_the_times_ffprobe_gives(clip: Path) -> None:
    assert mp4.index_times(clip) == pytest.approx(ffprobe.frame_times(clip), abs=1e-5)
    assert mp4.frame_times(clip) == mp4.index_times(clip)


def test_a_file_the_index_cannot_time_goes_to_ffprobe(clip: Path, tmp_path: Path) -> None:
    fragmented = tmp_path / "fragmented.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(clip),
            "-c",
            "copy",
            "-movflags",
            "frag_keyframe+empty_moov",
            str(fragmented),
        ],
        check=True,
    )
    with pytest.raises(mp4.Unhandled):
        mp4.index_times(fragmented)
    assert mp4.frame_times(fragmented) == pytest.approx(ffprobe.frame_times(fragmented))


def test_decoder_hands_over_as_many_frames_as_are_timed(clip: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    cap, n = cv2.VideoCapture(str(clip)), 0
    while cap.grab():
        n += 1
    assert n == len(mp4.frame_times(clip))


def test_shots_tracks_and_motion_take_their_seconds_from_t(clip: Path) -> None:
    t = np.array(ffprobe.frame_times(clip))
    n = len(t)
    sfp = np.zeros(n)
    sfp[CUT] = 0.99
    frames = pl.DataFrame(
        {
            "frame": range(n),
            "t": t,
            "sfp": sfp,
            "afp": np.zeros(n),
            "mean": np.full(n, 0.5),
            "std": np.full(n, 0.2),
            "lum_std": np.full(n, 0.2),
        }
    )
    fps = n / (t[-1] - t[0])
    times = clock.Clock(frames["t"].to_numpy(), fps=fps)
    shots = segment(frames, times)

    assert shots["f0"].to_list() == [0, CUT]
    assert shots["t0"].to_list() == pytest.approx([t[0], t[CUT]])
    assert shots["t1"].to_list() == pytest.approx([t[CUT], t[-1] + SLOW], abs=1e-3)
    # The same cut by frame / fps lands a sixth of a second away.
    assert abs(CUT / fps - t[CUT]) > 0.15


def test_clock_without_t_falls_back_and_says_so(clip: Path, tmp_path: Path) -> None:
    n = len(mp4.frame_times(clip))
    pl.DataFrame({"frame": range(n)}).write_parquet(tmp_path / "frames.parquet")
    assert clock.of(tmp_path, clip, 24.0).source == clock.FROM_VIDEO
    assert clock.of(tmp_path, None, 24.0).source == clock.FROM_FPS
    pl.DataFrame({"frame": range(n - 1)}).write_parquet(tmp_path / "frames.parquet")
    assert clock.of(tmp_path, clip, 24.0).source == clock.FROM_FPS


def test_decoded_frames_are_placed_by_the_time_the_decoder_reports(clip: Path) -> None:
    t: NDArray[np.float64] = np.array(mp4.frame_times(clip))
    # A decoder that skipped frames 20-24 and never handed over the last three,
    # reporting times counted from the first frame and rounded to the millisecond.
    kept = np.r_[0:20, 25 : len(t) - 3]
    shown = (t[kept] - t[0]).round(3)
    assert clock.places(shown, t).tolist() == kept.tolist()

    times = clock.Clock(t[clock.places(shown, t)])
    assert float(times.at(20)) == pytest.approx(t[25])

    with pytest.raises(ValueError, match="match no time"):
        clock.places(np.append(shown, shown[-1] + 0.011), t)
