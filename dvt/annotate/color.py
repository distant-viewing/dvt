# -*- coding: utf-8 -*-
"""Annotators to extract colors.
"""

import numpy as np
import cv2

from .core import FrameAnnotator
from ..utils import stack_dict_frames, sub_image, _trim_bbox


class ColorAnnotator(FrameAnnotator):
    """Annotator for detecting dominant colors in an image.

    The annotator will return the proportion of an image with specific values
    for hue, saturation, and value.

    Attributes:
        detector: An object with a method called detect that takes an image
            and returns a set of detect faces. Can be set to None (default) as
            a pass-through option for testing.
        embedding: An object with a method embed that takes an image along with
            a set of bounding boxed and returns embeddings of the faces as a
            numpy array. Set to None (default) to only run the face detector.
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will embed every other frame in the batch.
    """

    name = "face"

    def __init__(self, freq=1):

        self.freq = freq
        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the face annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, bounding
            box coordinates (top, bottom, left, and right). If an embedding is
            included, the result will also contain a numpy array of the
            embedding for each face.
        """

        f_faces = []
        for fnum in range(0, batch.bsize, self.freq):
            img = batch.img[fnum, :, :, :]
            t_faces = stack_dict_frames(self.detector.detect(img))
            if t_faces:
                frame = batch.get_frame_names()[fnum]
                t_faces["video"] = [batch.vname] * len(t_faces["top"])
                t_faces["frame"] = [frame] * len(t_faces["top"])
                if self.embedding is not None:
                    t_faces["embed"] = self.embedding.embed(img, t_faces)
                f_faces.append(t_faces)

        return f_faces


def get_color_histogram(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.int64( (img_hsv[:, :, 1] > 100) & (img_hsv[:, :, 1] > 100))
    vals = np.histogram(img_hsv[:, :, 0], weights=mask)
