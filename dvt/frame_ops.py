"""Per-frame reductions and descriptors, shared by everything that decodes."""

from typing import TYPE_CHECKING

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cv2.typing import MatLike

# A crop as top, bottom, left and right: the rows and columns kept.
Box = tuple[int, int, int, int]
# What frame_metrics measures of one frame: the luma mean, spread and 5th,
# 50th and 95th percentiles, the edge strength, the saturation, the Lab a
# and b means, and the luma histogram.
Metrics = tuple[float, float, float, float, float, float, float, float, float, NDArray[np.float32]]

MID_W, MID_H = 192, 108
MOTION_W, MOTION_H = 64, 36
THUMB_W, THUMB_H = 48, 27

HIST_BINS = 16

MIN_DRESP = 0.5

FULL_BOX = (0, MID_H, 0, MID_W)


def motion_thumb(mid: "MatLike") -> "MatLike":
    small = cv2.resize(mid, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small.astype(np.float32), cv2.COLOR_BGR2GRAY)


def chroma_stats(img: "MatLike") -> tuple[float, float]:
    small = cv2.resize(img, (MID_W, MID_H), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    ab = lab[:, :, 1:].astype(np.float32) - 128.0
    chroma = float(np.hypot(ab[:, :, 0], ab[:, :, 1]).mean())
    sat = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0
    return chroma, sat


def mid_crop_box(meta: pl.DataFrame) -> Box:
    sy = MID_H / meta["img_height"][0]
    sx = MID_W / meta["img_width"][0]
    top = int(round(meta["crop_top"][0] * sy))
    bottom = MID_H - int(round(meta["crop_bottom"][0] * sy))
    left = int(round(meta["crop_left"][0] * sx))
    right = MID_W - int(round(meta["crop_right"][0] * sx))
    if bottom - top < 8 or right - left < 8:
        return FULL_BOX
    return top, bottom, left, right


MIN_CROP_FRAC = 0.005


def source_crop_box(meta: pl.DataFrame, width: int, height: int) -> Box | None:
    top = int(meta["crop_top"][0])
    bottom = height - int(meta["crop_bottom"][0])
    left = int(meta["crop_left"][0])
    right = width - int(meta["crop_right"][0])

    if bottom - top < height // 8 or right - left < width // 8:
        return None
    if (
        top + (height - bottom) < MIN_CROP_FRAC * height
        and left + (width - right) < MIN_CROP_FRAC * width
    ):
        return None
    return top, bottom, left, right


def box_scale(
    box: Box, img_width: float, img_height: float, out_w: int = MOTION_W, out_h: int = MOTION_H
) -> tuple[float, float]:
    top, bottom, left, right = box
    return (
        (right - left) * img_width / MID_W / out_w,
        (bottom - top) * img_height / MID_H / out_h,
    )


def frame_metrics(mid: "MatLike") -> Metrics:
    gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    p5, p50, p95 = np.percentile(gray, (5, 50, 95))

    lmean, lstd = cv2.meanStdDev(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(cv2.magnitude(gx, gy).mean()) / 255.0

    sat = float(cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0
    lab = cv2.cvtColor(mid, cv2.COLOR_BGR2Lab)

    hist = cv2.calcHist([gray], [0], None, [HIST_BINS], [0, 256]).ravel()

    return (
        float(lmean[0, 0]) / 255.0,
        float(lstd[0, 0]) / 255.0,
        float(p5) / 255.0,
        float(p50) / 255.0,
        float(p95) / 255.0,
        edge,
        sat,
        float(lab[:, :, 1].mean()) - 128.0,
        float(lab[:, :, 2].mean()) - 128.0,
        hist / hist.sum(),
    )
