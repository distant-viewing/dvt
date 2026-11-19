from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from .kalman_filter import Floats

if TYPE_CHECKING:
    from .byte_tracker import STrack

# Row or column numbers: an array, or for an empty cost matrix a tuple.
Indices = NDArray[np.integer] | tuple[int, ...]


def linear_assignment(
    cost_matrix: Floats, thresh: float
) -> tuple[NDArray[np.integer], Indices, Indices]:
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    cost = np.asarray(cost_matrix, dtype=float)
    big = 1e6
    cost_capped = np.where(cost > thresh, big, cost)
    rows, cols = linear_sum_assignment(cost_capped)
    keep = cost[rows, cols] <= thresh
    matches: NDArray[np.integer] = (
        np.stack([rows[keep], cols[keep]], axis=1) if keep.any() else np.empty((0, 2), dtype=int)
    )
    unmatched_a = np.setdiff1d(np.arange(int(cost.shape[0])), matches[:, 0])
    unmatched_b = np.setdiff1d(np.arange(int(cost.shape[1])), matches[:, 1])
    return matches, unmatched_a, unmatched_b


def bbox_ious(atlbrs: ArrayLike, btlbrs: ArrayLike) -> Floats:
    a: Floats = np.asarray(atlbrs, dtype=float)
    b: Floats = np.asarray(btlbrs, dtype=float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1 + 1, 0, None)
    ih = np.clip(iy2 - iy1 + 1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def ious(atlbrs: Sequence[Floats], btlbrs: Sequence[Floats]) -> Floats:
    empty: Floats = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if empty.size == 0:
        return empty
    return bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float), np.ascontiguousarray(btlbrs, dtype=float)
    )


def _tlbrs(items: "Sequence[STrack] | Sequence[Floats]") -> list[Floats]:
    """The boxes of tracks, or the boxes themselves when given boxes."""
    return [t if isinstance(t, np.ndarray) else t.tlbr for t in items]


def iou_distance(
    atracks: "Sequence[STrack] | Sequence[Floats]", btracks: "Sequence[STrack] | Sequence[Floats]"
) -> Floats:
    return 1 - ious(_tlbrs(atracks), _tlbrs(btracks))


def fuse_score(cost_matrix: Floats, detections: "Sequence[STrack]") -> Floats:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim
