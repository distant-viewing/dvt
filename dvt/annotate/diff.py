# -*- coding: utf-8 -*-
"""This module illustrates something.
"""

import cv2
import numpy as np

from .core import FrameAnnotator


class DiffAnnotator(FrameAnnotator):
    """Here"""

    name = 'diff'

    def __init__(self, quantiles=None, bins=16, size=32):
        if not quantiles:
            quantiles = []

        self.quantiles = quantiles
        self.bins = bins
        self.size = size
        super().__init__()

    def annotate(self, batch):
        """Here

        :param batch:

        """

        output = {'video': [batch.vname] * batch.bsize}
        output['frame'] = batch.get_frame_nums()
        output['avg_value'] = _average_value_batch(batch)

        for quant in self.quantiles:
            key = "q{0:d}".format(quant)
            output[key] = l1_quantile(batch, quantile=quant, size=self.size)

            key = "h{0:d}".format(quant)
            output[key] = hist_diffs(batch, quantile=quant, bins=self.bins)

        return [output]


def l1_quantile(batch, quantile=50, size=32):
    """Here

    :param batch:
    :param quantile:  (Default value = 50)
    :param size:  (Default value = 32)

    """
    bsize = batch.bsize
    msize = bsize + 1
    assert msize <= batch.img.shape[0]

    simg = np.zeros((msize, size, size, 3))
    for iran in range(msize):
        fsmall = cv2.resize(batch.img[iran, :, :, :], (size, size))
        fsmall_hsv = cv2.cvtColor(fsmall, cv2.COLOR_RGB2HSV)
        simg[iran, :, :, :] = fsmall_hsv

    norm = simg[slice(0, bsize), :, :, :] - simg[slice(1, bsize + 1), :, :, :]

    return np.percentile(np.abs(norm), q=quantile, axis=(1, 2, 3))


def hist_diffs(batch, quantile=50, bins=16):
    """Here

    :param batch:
    :param quantile:  (Default value = 50)
    :param bins:  (Default value = 16)

    """
    bsize = batch.bsize
    msize = bsize + 1
    assert msize <= batch.img.shape[0]

    hist_vals = hsv_hist(batch, msize, bins=bins)
    norm = hist_vals[slice(0, bsize), :] - hist_vals[slice(1, bsize + 1), :]
    norm = norm / np.prod(batch.img.shape[1:4]) * 100

    return np.percentile(np.abs(norm), q=quantile, axis=(1))


def hsv_hist(batch, msize, bins=16):
    """Here

    :param batch:
    :param msize:
    :param bins:  (Default value = 16)

    """
    hist_vals = np.zeros((msize, bins * 3))
    for iran in range(msize):
        hsv = cv2.cvtColor(batch.img[iran, :, :, :], cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256]).flatten()
            hist_vals[iran, slice(i * bins, (i + 1) * bins)] = hist

    return hist_vals


def _average_value_batch(batch):
    """Here

    :param batch:

    """
    img = batch.get_batch()
    return np.mean(img, axis=(1, 2, 3))
