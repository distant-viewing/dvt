# -*- coding: utf-8 -*-
"""Annotators to extract clutter or visual complexity values.
"""

import importlib

import numpy as np
import cv2

from .core import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames


class ClutterAnnotator(FrameAnnotator):
    """Annotator for extracting the clutter or visual complexity value from a
    frame. The clutter value is calculated based on the Subband Entropy
    algorithm proposed by Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano. in
    "Measuring Visual Clutter". Journal of Vision, 7(2), 2007.

    The annotator will return a a single value per frame, describing its
    clutter value.

    Attributes:
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will computer every other frame in the batch.
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
    """

    name = "clutter"

    def __init__(self, freq=1, frames=None):

        self.freq = freq
        self.frames = _proc_frame_list(frames)
        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the cielab annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, and the
            clutter value extracted from the frame.
        """
        # determine which frames to work on
        frames = _which_frames(batch, self.freq, self.frames)
        if not frames:
            return

        # run the clutter analysis on each frame
        clutter_val = []
        for fnum in frames:
            clutter_val += [_get_clutter(batch.img[fnum, :, :, :])]

        obj = {"clutter": np.vstack(clutter_val)}

        # Add video and frame metadata
        obj["video"] = [batch.vname] * len(frames)
        obj["frame"] = np.array(batch.get_frame_names())[list(frames)]

        return [obj]


def _entropy(samples):
    nsamples = samples.size
    nbins = int(np.ceil(np.sqrt(nsamples)))

    bincount, _ = np.histogram(samples, nbins)
    H = bincount[np.where(bincount > 0)]
    H = H / float(sum(H))

    return -sum(H * np.log2(H))


def _band_entropy(img, wlevels, wor):
    pt = importlib.import_module("pyrtools")
    sfpyr = pt.pyramids.SteerablePyramidFreq(img, wlevels, wor - 1)

    en_band = np.zeros((sfpyr.num_scales, sfpyr.num_orientations))

    for i in range(sfpyr.num_scales):
        for b in range(sfpyr.num_orientations):
            band = sfpyr.pyr_coeffs[(i, b)]
            en_band[i, b] = _entropy(np.ravel(band))

    return en_band


def _get_clutter(img, wlevels=3, wght_chrom=0.0625):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    wor = 4
    L = img[:, :, 0]

    en_band = _band_entropy(L, wlevels, wor)
    clutter_se = np.mean(en_band)

    for i in range(1, 3):
        chrom = img[:, :, i]
        if (np.max(chrom) - np.min(chrom)) < 0.008:     # pragma: no cover
            chrom = np.zeros(chrom.shape)
        en_band = _band_entropy(chrom, wlevels, wor)
        clutter_se += wght_chrom * np.mean(en_band)

    clutter_se /= 1 + 2 * wght_chrom
    return clutter_se
