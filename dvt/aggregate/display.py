# -*- coding: utf-8 -*-
"""Display detected objects and faces in output image files.

The aggregator functions here takes detected faces and objects. It draws
bounding boxes over the repective frames and saves the png files locally.
Requires that the PNG annotator was run and the original images are saved.
"""

import os

import cv2
import numpy as np

from ..core import Aggregator
from ..utils import _check_out_dir

class DisplayAggregator(Aggregator):
    """Display detected faces and objects as overlay over image frames.

    Attributes:
        min_obj_score (float): minimum confidence score to include a detected
            object in the computation
        min_face_score (float): minimum confidence score to include a detected
            face in the computation
        shot_names (list): a list of shot names, from the longest shot to the
            tightest. Set to None to use the default settings.
        shot_sizes (list): as list of shot size cut-offs given as a proportion
            (vertical) of face size to the entire shot. Should be an increasing
            list starting at zero and the same length as shot_names. Set to
            None to use the default settings.
    """

    name = "display"

    def __init__(self, **kwargs):

        self.input_dir = _check_out_dir(kwargs.get("input_dir"), True)
        self.output_dir = _check_out_dir(kwargs.get("output_dir"))
        self.frames = kwargs.get('frames', None)

    def aggregate(self, ldframe):
        """Create output png files showing the annotated data.

        Args:
            ldframe (dict): A dictionary of DictFrames from a FrameAnnotator.
            frames (list): An optional list of frames. Otherwise, will annotate
                any frame with a detected face or object.
        """

        # what frames to include?
        frames = self.frames
        if frames is None:
            frames = set()
            if "face" in ldframe:
                frames = frames.union(ldframe["face"]["frame"])
            if "obj" in ldframe:
                frames = frames.union(ldframe["obj"]["frame"])
        frames = sorted(frames)

        for frame in frames:
            print(frame)
            _add_annotations_to_image(
                self.input_dir, self.output_dir, frame, ldframe
            )


def _add_annotations_to_image(input_dir, output_dir, frame, pipeline_data):
    # get input and file paths
    input_file = os.path.join(input_dir, "frame-{0:06d}.png".format(frame))
    output_file = os.path.join(output_dir, "frame-{0:06d}.png".format(frame))

    # define colours
    box_color = (255, 165, 0)
    face_color = (22, 75, 203)
    white_color = (255, 255, 255)

    img = cv2.imread(input_file)

    if pipeline_data.get("obj") is not None:
        img = _add_bbox(img, frame, pipeline_data["obj"], box_color, 2)
        img = _add_box_text(
            img,
            frame,
            pipeline_data["obj"],
            "category",
            color=white_color,
            bgc=box_color,
            size=0.5,
        )

    if pipeline_data.get("face") is not None:
        img = _add_bbox(img, frame, pipeline_data["face"], face_color, 1)

    _ = cv2.imwrite(output_file, img)


def _add_bbox(img, frame, pdf, color=(255, 255, 255), thickness=2):

    for ind in np.nonzero(np.array(pdf["frame"]) == frame)[0]:
        # grab values from data
        top = pdf["top"][ind]
        right = pdf["right"][ind]
        bottom = pdf["bottom"][ind]
        left = pdf["left"][ind]

        # plot the rectangle
        img = cv2.rectangle(
            img, (left, top), (right, bottom), color, thickness
        )

    return img


def _add_box_text(img, frame, pdf, lvar, color=(0, 0, 0), bgc=None, size=0.5):

    font = cv2.FONT_HERSHEY_SIMPLEX
    for ind in np.nonzero(np.array(pdf["frame"]) == frame)[0]:
        # grab values from data
        bottom = pdf["bottom"][ind]
        left = pdf["left"][ind]
        msg = pdf[lvar][ind]

        if bgc:
            # make a text box with background color bg
            (text_width, text_height) = cv2.getTextSize(
                msg, font, fontScale=size, thickness=1
            )[0]
            text_offset_x = left
            text_offset_y = bottom
            box_coords = (
                (text_offset_x, text_offset_y + 1),
                (
                    text_offset_x + text_width + 5,
                    text_offset_y - text_height - 10,
                ),
            )
            img = cv2.rectangle(
                img, box_coords[0], box_coords[1], bgc, cv2.FILLED
            )

        # plot text and text box
        img = cv2.putText(
            img,
            msg,
            (left + 5, bottom - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            1,
            cv2.LINE_AA,
        )

    return img
