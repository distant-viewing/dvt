import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dvt.tools import boxes

coords = st.floats(min_value=0, max_value=1000, allow_nan=False, width=32)


def box_arrays(n: int) -> st.SearchStrategy[np.ndarray]:
    def to_boxes(raw: np.ndarray) -> np.ndarray:
        lo = np.minimum(raw[:, :2], raw[:, 2:])
        hi = np.maximum(raw[:, :2], raw[:, 2:]) + 1
        return np.concatenate([lo, hi], axis=1)

    return arrays(np.float64, (n, 4), elements=coords).map(to_boxes)


def test_iou_examples() -> None:
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float)
    assert boxes.pairwise_iou(a, b) == pytest.approx([1.0, 1 / 3])


@given(box_arrays(5), box_arrays(5))
def test_iou_is_symmetric_and_bounded(a: np.ndarray, b: np.ndarray) -> None:
    iou = boxes.pairwise_iou(a, b)
    assert np.allclose(iou, boxes.pairwise_iou(b, a))
    assert np.all((iou >= 0) & (iou <= 1 + 1e-9))


def test_containment() -> None:
    faces = np.array([[2, 2, 4, 4], [20, 20, 22, 22]], float)
    heads = np.array([[0, 0, 10, 10], [3, 3, 13, 13]], float)
    assert boxes.containment(faces, heads) == pytest.approx(np.array([[1.0, 0.25], [0.0, 0.0]]))


def test_nms() -> None:
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = boxes.nms(b, np.array([0.5, 0.9, 0.7]), 0.5)
    assert keep.tolist() == [1, 2]
    assert keep.dtype == int


@given(box_arrays(8), arrays(np.float64, 8, elements=st.floats(0, 1)))
def test_nms_keeps_boxes_that_do_not_overlap(b: np.ndarray, scores: np.ndarray) -> None:
    keep = boxes.nms(b, scores, 0.5)
    assert len(set(keep.tolist())) == len(keep)
    kept = b[keep]
    iou = boxes.pairwise_iou(kept[:, None, :].repeat(len(kept), 1).reshape(-1, 4),
                             kept[None, :, :].repeat(len(kept), 0).reshape(-1, 4))  # fmt: skip
    off_diagonal = ~np.eye(len(kept), dtype=bool).ravel()
    assert np.all(iou[off_diagonal] <= 0.5)
