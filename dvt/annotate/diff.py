# -*- coding: utf-8 -*-
"""Annotators for finding difference between subsequent frames.

The annotators here detect differences from one frame to the next. They all
collect basic summaries of the overall brightness and saturation of a frame.
These are all useful for detecting shot boundaries and scene breaks.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of the DiffAnnotator over two batches of the input.

    >>> fp = FrameProcessor()
    >>> fp.load_annotator(DiffAnnotator(quantiles=[40]))
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    INFO:root:processed batch 00:00:00,000 to 00:00:17,083 with annotator: 'embed'
    INFO:root:processed batch 00:00:17,083 to 00:00:25,625 with annotator: 'embed'

    Then, collect the output from the annotator and display as a pandas data
    frame. Here, just the head of the data is shown for brevity.

    >>> fp.collect("diff").todf().head()
       q40      video  frame  h40  avg_value
    0  0.0  input.mp4      0  0.0   2.552584
    1  0.0  input.mp4      1  0.0   2.921136
    2  0.0  input.mp4      2  0.0   2.697502
    3  0.0  input.mp4      3  0.0   2.830626
    4  0.0  input.mp4      4  0.0   2.560596

    The output can be further processed with related aggregator classes to
    detect shot breaks.
"""

import cv2
import numpy as np

from .core import FrameAnnotator


class DiffAnnotator(FrameAnnotator):
    """Annotator for detecting differences between frames.

    The annotator will return data for each frame in a given batch by showing
    how much a frame differs compares to the next one. It computes two kinds of
    differences. The first down samples the image to a small square and takes
    pixel-by-pixel differences between the frames. The second computes a
    histogram in HSV space and compares the histogram counts between frames.
    Results for both are given by taking quantiles of the differences. The
    histogram differences are normalized to account for comparisons across
    different image sizes.

    Additionally, the average value (darkness) of each frame is provided to
    assist with putting the differences in context.

    Attributes:
        quantiles (list): A list of integers or floats giving the quantiles to
            return. Set to None to only return the average value of each frame.
            Inputs are given as percentiles, so [50] will return the median.
        size (int): Size of one side of the square used for down sampling the
            image.
        bins (int): How many bins to include in the histogram differences. Will
            make this many bins for each of hue, saturation, and value.
    """

    name = "diff"

    def __init__(self, quantiles=None, size=32, bins=16):
        if not quantiles:
            quantiles = []

        self.quantiles = quantiles
        self.size = size
        self.bins = bins
        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the difference annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, average
            value, and any requested quantile and histogram differences.
        """

        output = {"video": [batch.vname] * batch.bsize}
        output["frame"] = batch.get_frame_names()
        output["avg_value"] = _average_value_batch(batch)

        for quant in self.quantiles:
            key = "q{0:d}".format(quant)
            output[key] = _l1_quantile(batch, quantile=quant, size=self.size)

            key = "h{0:d}".format(quant)
            output[key] = _hist_diffs(batch, quantile=quant, bins=self.bins)

        return [output]


def _l1_quantile(batch, quantile=50, size=32):
    """Compute differences between subsequent frames in a batch.
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


def _hist_diffs(batch, quantile=50, bins=16):
    """Compute differences between HSV histograms across a batch.
    """
    bsize = batch.bsize
    msize = bsize + 1
    assert msize <= batch.img.shape[0]

    hist_vals = _hsv_hist(batch, msize, bins=bins)
    norm = hist_vals[slice(0, bsize), :] - hist_vals[slice(1, bsize + 1), :]
    norm = norm / np.prod(batch.img.shape[1:4]) * 100

    return np.percentile(np.abs(norm), q=quantile, axis=(1))


def _hsv_hist(batch, msize, bins=16):
    """Compute histogram counts from a batch of images.
    """
    hist_vals = np.zeros((msize, bins * 3))
    for iran in range(msize):
        hsv = cv2.cvtColor(batch.img[iran, :, :, :], cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256]).flatten()
            hist_vals[iran, slice(i * bins, (i + 1) * bins)] = hist

    return hist_vals


def _average_value_batch(batch):
    """Compute the average value across a batch of images.
    """
    img = batch.get_batch()
    return np.mean(img, axis=(1, 2, 3))
