"""Box geometry shared by the detectors: containment, IoU and non-maximum suppression."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TrackParams:
    track_thresh: float = 0.4
    match_thresh: float = 0.8
    buffer_s: float = 0.5
    min_track_frames: int = 3
    embed_gate: float = 0.7
    emb_min_norm: float = 17.0

    face_in_head_min: float = 0.4
    nms_iou: float = 0.5
    head_wr: float = 1.75
    head_hr: float = 1.55
    head_dy: float = -0.17
    dup_track_iou: float = 0.3

    split_dist: float = 0.65
    split_persist: int = 3
    split_min_faces: int = 5

    stitch_gap_s: float = 2.0
    stitch_dist: float = 0.45
    stitch_space: float = 2.0
    stitch_space_per_frame: float = 0.1

    norm_ref: float = 20.0
    trim_dist: float = 0.6
    min_faces_pool: int = 3
    pose_k_max: int = 6
    pose_min_per_k: int = 8
    k_exemplars: int = 5
    frontal_yaw_max: float = 0.5


def containment(faces_xyxy: Array, heads_xyxy: Array) -> Array:
    """For every (face, head) pair, the share of the face's area inside the head."""
    f, h = faces_xyxy, heads_xyxy
    ix1 = np.maximum(f[:, None, 0], h[None, :, 0])
    iy1 = np.maximum(f[:, None, 1], h[None, :, 1])
    ix2 = np.minimum(f[:, None, 2], h[None, :, 2])
    iy2 = np.minimum(f[:, None, 3], h[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    farea = np.maximum((f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1]), 1e-6)
    return inter / farea[:, None]


def _area(x: Array) -> Array:
    return (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])


def pairwise_iou(a: Array, b: Array) -> Array:
    ix1 = np.maximum(a[:, 0], b[:, 0])
    iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2])
    iy2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum(_area(a) + _area(b) - inter, 1e-9)


def nms(boxes: Array, scores: Array, iou_thr: float) -> npt.NDArray[np.int_]:
    order = np.argsort(-scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)
