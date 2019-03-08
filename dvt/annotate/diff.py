# -*- coding: utf-8 -*-

import numpy as np
import cv2

from .core import FrameAnnotator


class DiffAnnotator(FrameAnnotator):
    name = 'diff'

    def __init__(self, quantiles=None, bins=16, size=32):
        if not quantiles:
            quantiles = []

        self.quantiles = quantiles
        self.bins = bins
        self.size = size
        super().__init__()

    def annotate(self, batch):

        output = {'video': [batch.vname] * batch.bsize}
        output['frame'] = batch.get_frame_nums()
        output['avg_value'] = _average_value_batch(batch)

        for quant in self.quantiles:
            key = "q{0:d}".format(quant)
            output[key] = l1_quantile(batch, q=quant, size=self.size)

            key = "h{0:d}".format(quant)
            output[key] = hist_diffs(batch, q=quant, bins=self.bins)



        return [output]


def l1_quantile(batch, q=50, size=32):
    n = batch.bsize
    m = n + 1
    assert m <= batch.img.shape[0]

    simg = np.zeros((m, size, size, 3))
    for iran in range(m):
        fsmall = cv2.resize(batch.img[iran, :, :, :], (size, size))
        fsmall_hsv = cv2.cvtColor(fsmall, cv2.COLOR_RGB2HSV)
        simg[iran, :, :, :] = fsmall_hsv

    l1 = simg[slice(0, n), :, :, :] - simg[slice(1, n + 1), :, :, :]

    return np.percentile(np.abs(l1), q=q, axis=(1, 2, 3))


def hist_diffs(batch, q=50, bins=16):
    n = batch.bsize
    m = n + 1
    assert m <= batch.img.shape[0]

    hist_vals = hsv_hist(batch, m, bins=bins)
    l1 = hist_vals[slice(0, n), :] - hist_vals[slice(1, n + 1), :]
    l1 = l1 / np.prod(batch.img.shape[1:4]) * 100

    return np.percentile(np.abs(l1), q=q, axis=(1))


def hsv_hist(batch, m, bins=16):
    hist_vals = np.zeros((m, bins * 3))
    for iran in range(m):
        hsv = cv2.cvtColor(batch.img[iran, :, :, :], cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256]).flatten()
            hist_vals[iran, slice(i * bins, (i + 1) * bins)] = hist

    return hist_vals


def _average_value_batch(batch):
    img = batch.get_batch()
    return np.mean(img, axis=(1, 2, 3))
