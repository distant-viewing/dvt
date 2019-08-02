# -*- coding: utf-8 -*-
"""Annotator to extract dense Optical Flow using the opencv
Gunnar Farneback’s algorithm.
"""

import os

import numpy as np
import cv2

from .core import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames


class OpticalFlowAnnotator(FrameAnnotator):
    """Annotator to extract dense Optical Flow using the opencv Gunnar
    Farneback’s algorithm.

    The annotator will return an image or flow field describing the motion in
    two subsequent frames.

    Attributes:
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will computer every other frame in the batch.
        raw (bool): Return optical flow as color image by default, raw returns
            the raw output as produced by the opencv algorithm.
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If set
            to something other than None, the freq input is ignored.
        output_dir (string): optional location to store the computed images.
            Only used if raw is set to False.
    """

    name = "opticalflow"

    def __init__(self, freq=1, raw=False, frames=None, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

        self.freq = freq
        self.raw = raw
        self.frames = _proc_frame_list(frames)
        self.output_dir = output_dir
        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the optical flow annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, and the
            optical flow representation. The latter has the same spatial
            dimensions as the input.
        """
        # determine which frames to work on
        frames = _which_frames(batch, self.freq, self.frames)
        if not frames:
            return

        # run the optical flow analysis on each frame
        flow = []
        for fnum in frames:
            current_gray = cv2.cvtColor(
                batch.img[fnum, :, :, :], cv2.COLOR_RGB2GRAY
            )
            next_gray = cv2.cvtColor(
                batch.img[fnum + 1, :, :, :], cv2.COLOR_RGB2GRAY
            )

            flow += [_get_optical_flow(current_gray, next_gray)]

            if not self.raw:
                flow[-1] = _flow_to_color(flow[-1])
                if self.output_dir is not None:
                    opath = os.path.join(
                        self.output_dir, "frame-{0:6d}.png".format(fnum)
                    )
                    cv2.imwrite(filename=opath, img=flow[-1])

        obj = {"opticalflow": np.stack(flow)}

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


# Optical flow to color image conversion code adapted from:
# https://github.com/tomrunia/OpticalFlow_Visualization


def _make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def _flow_compute_color(u, v):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Attributes:
        u (np.ndarray): horizontal flow.
        v (np.ndarray): vertical flow.
    """

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = _make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        flow_image[:, :, i] = np.floor(255 * col)

    return flow_image


def _flow_to_color(flow_uv, clip_flow=None):
    """
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Attributes:
        flow_uv (np.ndarray): np.ndarray of optical flow with shape [H,W,2]
        clip_flow (float): maximum clipping value for flow
    """

    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return _flow_compute_color(u, v)
