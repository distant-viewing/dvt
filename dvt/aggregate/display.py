# -*- coding: utf-8 -*-
"""Display detected objects and faces in output image files.

The aggregator functions here take as input detected faces and objects. It
draws bounding boxes over the repective frames and saves the png files
locally. Requires that the PNG annotator was run and the original images are
saved somewhere locally.
"""

from os.path import join

from cv2 import (
    getTextSize,
    imread,
    imwrite,
    putText,
    rectangle,
    resize,
    FILLED,
    FONT_HERSHEY_SIMPLEX,
    LINE_AA
)
from numpy import array, nonzero

from ..abstract import Aggregator
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
        size (int): What should the size of the output images be? Set to
            None, the default, to preserve the size as given in the input file.
            Given as the desired height; the width will be scaled to keep the
            aspect ratio.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "display"

    def __init__(self, **kwargs):

        self.input_dir = _check_out_dir(kwargs.get("input_dir"), True)
        self.output_dir = _check_out_dir(kwargs.get("output_dir"))
        self.frames = kwargs.get('frames', None)
        self.size = kwargs.get('size', None)

        super().__init__(**kwargs)

    def aggregate(self, ldframe, **kwargs):
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
            self._add_annotations_to_image(frame, ldframe)

    def _add_annotations_to_image(self, frame, pipeline_data):

        # get input and file paths
        input_file = join(self.input_dir, "frame-{0:06d}.png".format(frame))
        output_file = join(self.output_dir, "frame-{0:06d}.png".format(frame))

        # define colours
        box_color = (255, 165, 0)
        face_color = (22, 75, 203)
        white_color = (255, 255, 255)

        img = imread(input_file)

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

        if self.size is not None:
            scale = img.shape[0] / self.size
            new_size = (int(img.shape[1] // scale), int(self.size))
            img_resize = resize(img, new_size)
            _ = imwrite(filename=output_file, img=img_resize)
        else:
            _ = imwrite(output_file, img)


def _add_bbox(img, frame, pdf, color=(255, 255, 255), thickness=2):

    for ind in nonzero(array(pdf["frame"]) == frame)[0]:
        # grab values from data
        top = pdf["top"][ind]
        right = pdf["right"][ind]
        bottom = pdf["bottom"][ind]
        left = pdf["left"][ind]

        # plot the rectangle
        img = rectangle(
            img, (left, top), (right, bottom), color, thickness
        )

    return img


def _add_box_text(img, frame, pdf, lvar, color=(0, 0, 0), bgc=None, size=0.5):

    for ind in nonzero(array(pdf["frame"]) == frame)[0]:
        # grab values from data
        bottom = pdf["bottom"][ind]
        left = pdf["left"][ind]
        msg = pdf[lvar][ind]

        if bgc:
            # make a text box with background color bg
            (text_width, text_height) = getTextSize(
                msg, FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=1
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
            img = rectangle(
                img, box_coords[0], box_coords[1], bgc, FILLED
            )

        # plot text and text box
        img = putText(
            img,
            msg,
            (left + 5, bottom - 5),
            FONT_HERSHEY_SIMPLEX,
            size,
            color,
            1,
            LINE_AA,
        )

    return img
