# -*- coding: utf-8 -*-

import numpy as np
import cv2

from .core import FrameAnnotator
from .utils import combine_list_dicts


class VirtualCutDetector(FrameAnnotator):
    name = 'virtual-cut'

    def annotate(self, batch, pipeline):
        return []


class SimpleCutDetector(VirtualCutDetector):
    name = 'simple-cut'

    def __init__(self, l40=4, h16=100, train=False):
        self.l40 = l40
        self.h16 = h16
        self.train = train
        super().__init__()

    def annotate(self, batch):
        vals_l40 = l1_quantile(batch, diffs = [1], q=40)[1]
        vals_h16 = hist_diffs(batch, diffs = [1])[1]
        avg_value = average_value(batch)

        return [{'frame': batch.get_frame_nums(),
                 'vals_l40': vals_l40, 'vals_h16': vals_h16,
                 'avg_value': avg_value}]

    def collect(self, output):
        op = combine_list_dicts(output)
        if self.train:
            return op
        else:
            pass


def l1_quantile(batch, diffs, q=50, height=32, width=32):
    n = batch.bsize
    m = n + np.max(diffs)
    assert m <= batch.img.shape[0]

    simg = np.zeros((m, height, width, 3))
    for iran in range(m):
        fsmall = cv2.resize(batch.img[iran, :, :, :], (height, width))
        fsmall_hsv = cv2.cvtColor(fsmall, cv2.COLOR_RGB2HSV)
        simg[iran, :, :, :] = fsmall_hsv

    out = {}
    for d in diffs:
        l1 = simg[slice(0, n), :, :, :] - simg[slice(d, n + d), :, :, :]
        out[d] = np.percentile(np.abs(l1), q=q, axis=(1, 2, 3))

    return out


def hist_diffs(batch, diffs, q=50, bins=16):
    n = batch.bsize
    m = n + np.max(diffs)
    assert m <= batch.img.shape[0]

    hist_vals = hsv_hist(batch, m, bins=bins)

    out = {}
    for d in diffs:
        l1 = hist_vals[slice(0, n), :] - hist_vals[slice(d, n + d), :]
        l1 = l1 / np.prod(batch.img.shape[1:2])
        out[d] = np.percentile(np.abs(l1), q=q, axis=(1))

    return out


def hsv_hist(batch, m, bins=16):
    hist_vals = np.zeros((m, bins * 3))
    for iran in range(m):
        hsv = cv2.cvtColor(batch.img[iran, :, :, :], cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256]).flatten()
            hist_vals[iran, slice(i * bins, (i + 1) * bins)] = hist

    return hist_vals


def average_value(batch):
    img = batch.get_batch()
    return np.mean(img, axis=(1, 2, 3))



