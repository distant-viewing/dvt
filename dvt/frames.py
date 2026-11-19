"""The frames stage: one row a frame, and the shot boundary TransNetV2 gives each."""

import json
import time
from pathlib import Path

# The settings every annotator runs under, before torch is imported.
# isort: off
from . import runtime as runtime

# isort: on
import cv2
import numpy as np
import polars as pl
import torch

from . import clock, config, frame_ops
from .annotators import stamp
from .log import log_finished, log_status
from .tools import mp4
from .tools.values import StrPath
from .transnet import TransNetV2

MID_W, MID_H = frame_ops.MID_W, frame_ops.MID_H
MOTION_W, MOTION_H = frame_ops.MOTION_W, frame_ops.MOTION_H
THUMB_W, THUMB_H = frame_ops.THUMB_W, frame_ops.THUMB_H
HIST_BINS = frame_ops.HIST_BINS
MIN_DRESP = frame_ops.MIN_DRESP
FULL_BOX = frame_ops.FULL_BOX
motion_thumb = frame_ops.motion_thumb
mid_crop_box = frame_ops.mid_crop_box
source_crop_box = frame_ops.source_crop_box
box_scale = frame_ops.box_scale
frame_metrics = frame_ops.frame_metrics


def create_frames(ip: StrPath, od: Path) -> None:

    frames_path = od / "frames.parquet"
    if frames_path.exists():
        log_status(od, "frames", "Already exists")
        return

    t0 = time.monotonic()

    meta_path = od / "metadata.parquet"
    if not meta_path.exists():
        raise ValueError(
            f"{od.name}: frames needs metadata.parquet for the letterbox crop. "
            f"Run the metadata stage first."
        )
    meta = pl.read_parquet(meta_path)
    if "crop_top" not in meta.columns:
        raise ValueError(
            f"{od.name}: metadata.parquet predates letterbox detection. "
            f"Delete it and rerun the metadata stage."
        )
    box = mid_crop_box(meta)
    ctop, cbot, cleft, cright = box

    # Read before the decode so an unreadable index costs nothing.
    frame_t = mp4.frame_times(ip)

    transnetv2 = TransNetV2()
    transnetv2.load_state_dict(torch.load(config.models() / "detect_shots.pt"))
    transnetv2.eval()

    vinput = cv2.VideoCapture(str(ip))
    n_frames = int(vinput.get(cv2.CAP_PROP_FRAME_COUNT))
    thumbnails = torch.empty(n_frames, THUMB_H, THUMB_W, 3, dtype=torch.uint8)

    fnum = 0
    shown: list[float] = []
    mean_vals: list[float] = []
    std_vals: list[float] = []
    # The motion from the frame before; the first frame has none.
    dx_vals: list[float | None] = []
    dy_vals: list[float | None] = []
    dresp_vals: list[float | None] = []
    dabs_vals: list[float | None] = []
    metric_vals: list[frame_ops.Metrics] = []
    prev_mthumb = None
    while True:
        status, img = vinput.read()
        if not status:
            break
        shown.append(vinput.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gmean, gstd = cv2.meanStdDev(frame_gray)
        mean_vals.append(float(gmean[0, 0]) / 255.0)
        std_vals.append(float(gstd[0, 0]) / 255.0)

        mid = cv2.resize(img, (MID_W, MID_H), interpolation=cv2.INTER_AREA)

        content = mid[ctop:cbot, cleft:cright]

        mthumb = motion_thumb(content)
        if prev_mthumb is None:
            dx_vals.append(None)
            dy_vals.append(None)
            dresp_vals.append(None)
            dabs_vals.append(None)
        else:
            (dx, dy), resp = cv2.phaseCorrelate(prev_mthumb, mthumb)
            dx_vals.append(dx)
            dy_vals.append(dy)
            dresp_vals.append(resp)
            dabs_vals.append(float(np.abs(mthumb - prev_mthumb).mean()))
        prev_mthumb = mthumb

        metric_vals.append(frame_metrics(content))

        thumb = cv2.resize(content, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        thumbnails[fnum] = torch.as_tensor(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        fnum += 1

    if fnum != len(frame_t):
        # The decoder skipped damaged video or kept back its last frames, so
        # the frames are placed in the index by the times it reported.
        try:
            place = clock.places(shown, frame_t)
        except ValueError as e:
            raise ValueError(
                f"{od.name}: decoded {fnum} frames but the container times "
                f"{len(frame_t)}, and {e}, so no frame can be given a time. "
                f"Nothing was written."
            ) from e
        skipped = int((np.diff(place) > 1).sum()) + int(place[0] > 0)
        log_status(
            od,
            "frames",
            f"{len(frame_t) - fnum} of {len(frame_t)} frames never decoded: "
            f"{skipped} damaged stretch(es) skipped, "
            f"{len(frame_t) - 1 - int(place[-1])} lost at the end",
        )
        frame_t = [frame_t[int(i)] for i in place]

    thumbnails = thumbnails[:fnum]
    sfp, afp = transnetv2.predict_frames(thumbnails, label=od.name)
    sfp = np.roll(sfp, 1)
    afp = np.roll(afp, 1)
    sfp[0] = afp[0] = 0
    (lum_mean, lum_std, lum_p5, lum_p50, lum_p95, edge, sat, lab_a, lab_b, hist) = zip(
        *metric_vals, strict=True
    )
    df = pl.DataFrame(
        {
            "frame": range(len(sfp)),
            "t": pl.Series(frame_t, dtype=pl.Float64),
            "mean": mean_vals,
            "std": std_vals,
            "sfp": sfp,
            "afp": afp,
            "dx": pl.Series(dx_vals, dtype=pl.Float32),
            "dy": pl.Series(dy_vals, dtype=pl.Float32),
            "dresp": pl.Series(dresp_vals, dtype=pl.Float32),
            "dabs": pl.Series(dabs_vals, dtype=pl.Float32),
            "lum_mean": pl.Series(lum_mean, dtype=pl.Float32),
            "lum_std": pl.Series(lum_std, dtype=pl.Float32),
            "lum_p5": pl.Series(lum_p5, dtype=pl.Float32),
            "lum_p50": pl.Series(lum_p50, dtype=pl.Float32),
            "lum_p95": pl.Series(lum_p95, dtype=pl.Float32),
            "edge": pl.Series(edge, dtype=pl.Float32),
            "sat": pl.Series(sat, dtype=pl.Float32),
            "lab_a": pl.Series(lab_a, dtype=pl.Float32),
            "lab_b": pl.Series(lab_b, dtype=pl.Float32),
            "hist16": pl.Series(hist, dtype=pl.List(pl.Float32)),
        }
    )

    sx, sy = box_scale(box, meta["img_width"][0], meta["img_height"][0])
    stamp(df, "frames").write_parquet(
        frames_path,
        metadata={
            "frames": json.dumps(
                {
                    "fname": od.name,
                    "n_frames": fnum,
                    "crop_source": [
                        int(meta[f"crop_{s}"][0]) for s in ("top", "bottom", "left", "right")
                    ],
                    "crop_mid": list(box),
                    "crop_frac": float(meta["crop_frac"][0]),
                    "boxed": box != FULL_BOX,
                    "thumb_cropped": True,
                    "motion_cropped": True,
                    "dx_scale": sx,
                    "dy_scale": sy,
                    "thumb": [THUMB_W, THUMB_H],
                    "motion_thumb": [MOTION_W, MOTION_H],
                    "hist_bins": HIST_BINS,
                    "cv2": cv2.__version__,
                    "torch": torch.__version__,
                }
            )
        },
    )
    log_finished(od, "frames", t0)
