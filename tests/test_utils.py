import numpy as np
import pandas as pd
import cv2
import pytest

from dvt.annotate.core import FrameInput, FrameBatch
from dvt.utils import (
    DictFrame,
    stack_dict_frames,
    pd_to_dict_frame,
    get_batch,
    sub_image,
)


class TestDictFrame:
    def test_empty_dict(self):

        empty = DictFrame()
        assert empty.shape == (0, 0)

    def test_to_data_frame(self):

        dframe = DictFrame(
            {
                "x": [0, 1, 2],
                "y": ["a", "b", "c"],
                "embed": np.random.normal(0, 1, (3, 20)),
            }
        )
        exp_cols = ["x", "y"] + ["embed-{0:d}".format(x) for x in range(20)]
        pdf = dframe.todf()

        assert set(pdf.columns) == set(exp_cols)
        assert issubclass(type(pdf), pd.core.frame.DataFrame)

    def test_scalar_input(self):

        dframe = DictFrame({"x": 0, "y": ["a"]})
        assert dframe["x"] == [0]
        assert dframe["y"] == ["a"]

    def test_from_data_frame(self):

        dframe1 = DictFrame({"x": [0, 1, 2], "y": ["a", "b", "c"]})
        pdf = dframe1.todf()
        dframe2 = pd_to_dict_frame(pdf)

        assert set(pdf.columns) == set(["x", "y"])
        assert dframe1 == dframe2

    def test_stack(self):

        dframe1 = DictFrame(
            {
                "x": [0, 1, 2],
                "y": ["a", "b", "c"],
                "embed": np.random.normal(0, 1, (3, 20)),
            }
        )
        dframe2 = DictFrame(
            {"x": [0, 1], "y": ["a", "b"], "embed": np.random.normal(0, 1, (2, 20))}
        )

        dframe3 = stack_dict_frames([dframe1, dframe2])

        assert set(dframe3.keys()) == set(["x", "y", "embed"])
        assert issubclass(type(dframe3["embed"]), np.ndarray)
        assert dframe3.shape == (5, 3)

    def test_add(self):

        dframe1 = DictFrame(
            {
                "x": [0, 1, 2],
                "y": ["a", "b", "c"],
                "embed": np.random.normal(0, 1, (3, 20)),
            }
        )
        dframe2 = DictFrame(
            {"x": [0, 1], "y": ["a", "b"], "embed": np.random.normal(0, 1, (2, 20))}
        )

        dframe3 = dframe1 + dframe2

        assert set(dframe3.keys()) == set(["x", "y", "embed"])
        assert issubclass(type(dframe3["embed"]), np.ndarray)
        assert dframe3.shape == (5, 3)

    def test_bad_length(self):

        # check assertion on input
        with pytest.raises(AssertionError):
            dframe = DictFrame({"x": [0, 1], "y": ["a", "b", "c"]})

        # check assertion after the fact
        dframe = DictFrame({"x": [0, 1], "y": ["a", "b", "c"]}, check=False)
        with pytest.raises(AssertionError):
            dframe.verify()


class TestOtherUtils:
    def test_get_batch(self):

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        batch = get_batch(finput, batch_num=3)

        assert issubclass(type(batch), FrameBatch)
        assert batch.bnum == 3
        assert batch.bsize == 8

    def test_sub_image(self):

        finput = FrameInput("test-data/video-clip.mp4", bsize=1)
        batch = get_batch(finput, batch_num=0)
        img = batch.get_frames()[0, :, :, :]

        simg = sub_image(img, top=0, right=128, bottom=64, left=0)
        assert simg.shape == (64, 128, 3)

    def test_sub_image_fct(self):

        finput = FrameInput("test-data/video-clip.mp4", bsize=1)
        batch = get_batch(finput, batch_num=0)
        img = batch.get_frames()[0, :, :, :]

        simg = sub_image(img, top=0, right=128, bottom=64, left=0, fct=1.5)
        assert simg.shape == (80, 160, 3)

    def test_sub_image_reshape(self):

        finput = FrameInput("test-data/video-clip.mp4", bsize=1)
        batch = get_batch(finput, batch_num=0)
        img = batch.get_frames()[0, :, :, :]

        simg = sub_image(img, 32, 64, 64, 32, output_shape=(100, 100))
        assert simg.shape == (100, 100, 3)


if __name__ == "__main__":
    pytest.main([__file__])
