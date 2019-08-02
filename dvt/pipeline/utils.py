# -*- coding: utf-8 -*-
"""Utility functions for working with data pipelines.
"""

import os

import cv2
import numpy as np

from ..annotate.core import FrameProcessor, FrameInput, ImageInput
from ..annotate.diff import DiffAnnotator
from ..aggregate.cut import CutAggregator


def _get_cuts(finput):
    fpobj = FrameProcessor()
    fpobj.load_annotator(DiffAnnotator(quantiles=[40]))
    fri = FrameInput(finput, bsize=128)
    fpobj.process(fri)
    obj_out = fpobj.collect_all()

    ca = CutAggregator(cut_vals={"q40": 10}, min_len=30)
    agg = ca.aggregate(obj_out)
    agg["frame_start"] = np.array(agg["frame_start"])
    agg["frame_end"] = np.array(agg["frame_end"])
    agg["mpoint"] = (
        agg["frame_start"] + (agg["frame_end"] - agg["frame_start"]) // 2
    )
    return agg


def _add_annotations_to_image(input_file, output_dir, frame, pipeline_data):
    # get output file path
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    # define colours
    box_color = (255, 165, 0)
    face_color = (22, 75, 203)
    white_color = (255, 255, 255)

    fname = "frame-{0:06d}.png".format(frame)
    img = cv2.imread(input_file)

    if pipeline_data["object"] is not None:
        img = _add_bbox(img, frame, pipeline_data["object"], box_color, 2)
        img = _add_box_text(
            img,
            frame,
            pipeline_data["object"],
            "class",
            color=white_color,
            bgc=box_color,
            size=0.5,
        )

    if pipeline_data["face"] is not None:
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
