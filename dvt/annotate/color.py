# -*- coding: utf-8 -*-
"""Annotators to extract CIELAB color histograms.
"""

from numpy import vstack, stack, dtype, full, zeros, uint8, array, float32
from cv2 import (
    calcHist,
    cvtColor,
    COLOR_LAB2RGB,
    COLOR_RGB2LAB,
    COLOR_RGB2HSV,
    COLOR_RGB2LUV
)
from scipy.cluster.vq import kmeans

from ..abstract import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames


class ColorHistogramAnnotator(FrameAnnotator):
    """Annotator for constructing a color histogram.

    The annotator will return a histogram describing the color distribution
    of an image.

    Attributes:
        freq (int): How often to perform the annotator. For example, setting
            the frequency to 2 will computer every other frame in the batch.
        colorspace: What color space to use. Currently supports "hsv", "lab",
            and "luv". Default is "hsv".
        num_buckets (tuple): A tuple of three numbers giving the maximum number
            of buckets in each color channel, Lightness, A, B. These
            should each be a power of 2. Default is (16, 16, 16).
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "colorhist"

    def __init__(self, **kwargs):

        self.freq = kwargs.get("freq", 1)
        self.num_buckets = kwargs.get("num_buckets", (16, 16, 16))
        self.colorspace = kwargs.get("colorspace", "hsv")
        self.frames = _proc_frame_list(kwargs.get("frames", None))

        super().__init__(**kwargs)

    def annotate(self, batch):
        """Annotate the batch of frames with the color histogram annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, the
            histogram of length (num_buckets[0] * num_buckets[1] *
            num_buckets[2]).
        """
        # determine which frames to work on
        frames = _which_frames(batch, self.freq, self.frames)
        if not frames:
            return None

        # what color space to use?
        max_size = 255
        if self.colorspace == "lab":
            cspace = COLOR_RGB2LAB
        elif self.colorspace == "luv":
            cspace = COLOR_RGB2LUV
        else:
            cspace = COLOR_RGB2HSV
            max_size = 180
            self.colorspace = "hsv"

        # run the color analysis on each frame
        hgrams = []
        for fnum in frames:
            img_convert = cvtColor(batch.img[fnum, :, :, :], cspace)
            hgrams += [_get_histogram(img_convert, self.num_buckets, max_size)]

        obj = {self.colorspace: vstack(hgrams)}

        # Add video and frame metadata
        obj["frame"] = array(batch.get_frame_names())[list(frames)]

        return obj


class DominantColorAnnotator(FrameAnnotator):
    """Annotator for extracting the dominant colours for an image.

    The annotator will return a list of the most dominant colors.

    Attributes:
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will computer every other frame in the batch.
        num_dominant (int): Number of dominant colors to extract. Default is 5.
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "domcolor"

    def __init__(self, **kwargs):

        self.freq = kwargs.get("freq", 1)
        self.num_dominant = kwargs.get("num_dominant", 5)
        self.frames = _proc_frame_list(kwargs.get("frames", None))

        super().__init__(**kwargs)

    def annotate(self, batch):
        """Annotate the batch of frames with dominant colors.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            An array of dominant colors given as hex strings.
        """
        # determine which frames to work on
        frames = _which_frames(batch, self.freq, self.frames)
        if not frames:
            return None

        # run the color analysis on each frame
        dominant = []
        for fnum in frames:
            img_convert = cvtColor(batch.img[fnum, :, :, :], COLOR_RGB2LAB)
            dominant += [_get_dominant(img_convert, self.num_dominant)]

        obj_rgb = cvtColor(stack(dominant), COLOR_LAB2RGB)
        shp = (obj_rgb.shape[0], self.num_dominant)
        out = full(shp, "#000000", dtype=dtype("<U7"))
        for i, obj_frame in enumerate(obj_rgb):
            for j, occ in enumerate(obj_frame):
                out[i, j] = "#{0:02x}{1:02x}{2:02x}".format(
                    occ[0], occ[1], occ[2]
                )

        obj = {"dominant_colors": out}

        # Add video and frame metadata
        obj["frame"] = array(batch.get_frame_names())[list(frames)]

        return obj


def _get_histogram(img, num_buckets, max_size):

    return calcHist(
        [img], [0, 1, 2], None, num_buckets, [0, max_size, 0, 256, 0, 256]
    ).reshape(-1)


def _get_dominant(img, num_dominant):
    img_flat = img.reshape(-1, 3).astype(float32)

    # increasing iter would give 'better' clustering, at the cost of speed
    dominant_colors, _ = kmeans(img_flat, num_dominant, iter=5)
    #kmeans_code = vq(img_flat, dominant_colors)

    if dominant_colors.shape[0] != num_dominant:         # pragma: no cover
        diff = num_dominant - dominant_colors.shape[0]
        dominant_colors = vstack([
            dominant_colors,
            zeros((diff, dominant_colors.shape[1]))
        ])

    return dominant_colors.astype(uint8)
