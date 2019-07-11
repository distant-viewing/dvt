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
    """

    name = "face"

    def __init__(self, freq=1):

        self.freq = freq
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
    np.int32(vals[0] * 1000 / np.sum(vals[0]))

    img_hsv = np.int64(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    img_hsv[:, :, 0] = img_hsv[:, :, 0] // 16
    img_hsv[:, :, 1] = (img_hsv[:, :, 1] // 64) * 16
    img_hsv[:, :, 2] = (img_hsv[:, :, 2] // 64) * (16*4)


    np.sum(img_hsv, axis=2)
