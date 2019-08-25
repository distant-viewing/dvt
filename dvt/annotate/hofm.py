# -*- coding: utf-8 -*-
"""Annotator to extract dense Optical Flow.

Uses the opencv Gunnar Farneback’s algorithm and represent it as a
histogram of optical flow orientation and magnitude (HOFM).
"""

from importlib import import_module

from numpy import array, digitize, stack, zeros
from cv2 import cartToPolar

from ..abstract import FrameAnnotator
from .opticalflow import _get_optical_flow
from ..utils import _proc_frame_list, _which_frames


class HOFMAnnotator(FrameAnnotator):
    """Annotator to extract dense Optical Flow using the opencv
    Gunnar Farneback’s algorithm and represent it as a
    histogram of optical flow orientation and magnitude (HOFM).

    The annotator will return the optical flow describing the motion in
    two subsequent frames as a HOFM feature.

    Attributes:
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will computer every other frame in the batch.
        blocks (int): How many spatial blocks to divide the frame in, in each
            dimension. Default is 3, which results in 9 spatial blocks.
        mag_buckets (list of ints): List of bounds for magnitude
            buckets. Default is [0, 20, 40, 60, 80, 100].
        ang_buckets (list of ints): List of bounds for angle
            buckets. Default is [0, 45, 90, 135, 180, 225, 270, 315, 360].
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "hofm"

    def __init__(self, **kwargs):

        self.skutil = import_module("skimage.util")
        self.freq = kwargs.get("freq", 1)
        self.blocks = kwargs.get("blocks", 3)
        self.mag_buckets = kwargs.get("mag_buckets", [0, 20, 40, 60, 80, 100])
        self.ang_buckets = kwargs.get(
            "ang_buckets",
            [0, 45, 90, 135, 180, 225, 270, 315, 360]
        )
        self.frames = _proc_frame_list(kwargs.get("frames", None))

        super().__init__(**kwargs)

    def annotate(self, batch):
        """Annotate the batch of frames with the optical flow annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, and the
            HOFM optical flow representation of length (len(blocks) *
            len(blocks) * len(mag_buckets) * len(ang_buckets))
        """
        # determine which frames to work on
        frames = _which_frames(batch, self.freq, self.frames)
        if not frames:
            return None

        # run the optical flow analysis on each frame
        hofm = []
        for fnum in frames:
            flow = _get_optical_flow(batch, fnum)

            hofm.append(
                _make_block_hofm(
                    flow,
                    self.blocks,
                    self.mag_buckets,
                    self.ang_buckets,
                    self.skutil,
                ).flatten()
            )

        obj = {"hofm": stack(hofm)}

        # Add video and frame metadata
        obj["frame"] = array(batch.get_frame_names())[list(frames)]

        return obj


def _make_block_hofm(flow, blocks, mag_buckets, ang_buckets, skutil):
    mag, ang = cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    mag_digit = digitize(mag, mag_buckets)
    # mod so 360 falls into first bucket
    ang_digit = digitize(ang % 360, ang_buckets)

    mag_blocks = skutil.view_as_blocks(mag_digit, (blocks, blocks))
    ang_blocks = skutil.view_as_blocks(ang_digit, (blocks, blocks))

    histogram = zeros(
        (blocks, blocks, len(mag_buckets), len(ang_buckets) - 1)
    )

    for i in range(blocks):
        for j in range(blocks):
            for mblock, ablock in zip(
                mag_blocks[:, :, i, j].flatten(),
                ang_blocks[:, :, i, j].flatten(),
            ):
                histogram[i, j, mblock - 1, ablock - 1] += 1
        # normalize by block size (h,w)
        histogram[i, j, :, :] /= mag_blocks[:, :, i, j].size
    return histogram
