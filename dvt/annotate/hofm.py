# -*- coding: utf-8 -*-
"""Annotator to extract dense Optical Flow using the opencv
Gunnar Farneback’s algorithm and represent it as a
histogram of optical flow orientation and magnitude (HOFM),
as described in https://doi.org/10.1109/SIBGRAPI.2015.21
"""

import importlib

import numpy as np
import cv2

from .core import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames


class HOFMAnnotator(FrameAnnotator):
    """Annotator to extract dense Optical Flow using the opencv
    Gunnar Farneback’s algorithm and represent it as a
    histogram of optical flow orientation and magnitude (HOFM),
    as described in https://doi.org/10.1109/SIBGRAPI.2015.21

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
    """

    name = "hofm"

    def __init__(
        self, freq=1,
        blocks=3,
        mag_buckets=None,
        ang_buckets=None,
        frames=None,
    ):

        if mag_buckets is None:
            mag_buckets = [0, 20, 40, 60, 80, 100]

        if ang_buckets is None:
            ang_buckets = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        self.skutil = importlib.import_module("skimage.util")
        self.freq = freq
        self.blocks = blocks
        self.mag_buckets = mag_buckets
        self.ang_buckets = ang_buckets
        self.frames = _proc_frame_list(frames)
        super().__init__()

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
            return

        # run the optical flow analysis on each frame
        hofm = []
        for fnum in frames:
            current_gray = cv2.cvtColor(
                batch.img[fnum, :, :, :], cv2.COLOR_RGB2GRAY
            )
            next_gray = cv2.cvtColor(
                batch.img[fnum + 1, :, :, :], cv2.COLOR_RGB2GRAY
            )

            flow = _get_optical_flow(current_gray, next_gray)

            hofm.append(
                _make_block_hofm(
                    flow,
                    self.blocks,
                    self.mag_buckets,
                    self.ang_buckets,
                    self.skutil,
                ).flatten()
            )

        obj = {"hofm": np.stack(hofm)}

        # Add video and frame metadata
        obj["video"] = [batch.vname] * len(frames)
        obj["frame"] = np.array(batch.get_frame_names())[list(frames)]

        return [obj]


def _get_optical_flow(current_frame, next_frame):

    return cv2.calcOpticalFlowFarneback(
        current_frame,
        next_frame,
        flow=None,
        pyr_scale=0.5,
        levels=1,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )


def _make_block_hofm(flow, blocks, mag_buckets, ang_buckets, skutil):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    mag_digit = np.digitize(mag, mag_buckets)
    # mod so 360 falls into first bucket
    ang_digit = np.digitize(ang % 360, ang_buckets)

    mag_blocks = skutil.view_as_blocks(mag_digit, (blocks, blocks))
    ang_blocks = skutil.view_as_blocks(ang_digit, (blocks, blocks))

    histogram = np.zeros(
        (blocks, blocks, len(mag_buckets), len(ang_buckets) - 1)
    )

    for x in range(blocks):
        for y in range(blocks):
            for m, a in zip(
                mag_blocks[:, :, x, y].flatten(),
                ang_blocks[:, :, x, y].flatten(),
            ):
                histogram[x, y, m - 1, a - 1] += 1
        # normalize by block size (h,w)
        histogram[x, y, :, :] /= mag_blocks[:, :, x, y].size
    return histogram
