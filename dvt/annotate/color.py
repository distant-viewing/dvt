# -*- coding: utf-8 -*-
"""Annotators to extract colors.
"""

import numpy as np
import cv2

from .core import FrameAnnotator


class ColorAnnotator(FrameAnnotator):
    """Annotator for detecting dominant colors in an image.

    The annotator will return the proportion of an image with specific values
    for hue, saturation, and value.

    Attributes:
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will computer every other frame in the batch.
        num_buckets (tuple): A tuple of three numbers giving the maximum number
            of buckets in each color channel, hue, saturation, and value. These
            should each be divisible by 256. Default is (16, 4, 4).
    """

    name = "color"

    def __init__(self, freq=1, num_buckets=(16, 4, 4)):

        self.freq = freq
        self.num_buckets = num_buckets
        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the color annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, bounding
            box coordinates (top, bottom, left, and right). If an embedding is
            included, the result will also contain a numpy array of the
            embedding for each face.
        """

        # run the color analysis on each frame
        hgrams = []
        for fnum in range(0, batch.bsize, self.freq):
            hgrams += [_get_color_histogram(batch.img[fnum, :, :, :],
                       self.num_buckets)]

        obj = {'color': np.vstack(hgrams)}

        # Add video and frame metadata
        frames = range(0, batch.bsize, self.freq)
        obj["video"] = [batch.vname] * len(frames)
        obj["frame"] = np.array(batch.get_frame_names())[list(frames)]

        return [obj]


def _get_color_histogram(img, num_buckets):

    img_hsv = np.int64(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    max_sizes = tuple([int(256 / x) for x in num_buckets])

    z = img_hsv[:, :, 0] // max_sizes[0] + \
        (img_hsv[:, :, 1] // max_sizes[1]) * num_buckets[0] + \
        (img_hsv[:, :, 2] // max_sizes[2]) * num_buckets[0] * num_buckets[1]

    msize = num_buckets[0]*num_buckets[1]*num_buckets[2]

    return np.bincount(z.flatten(), minlength=msize)
