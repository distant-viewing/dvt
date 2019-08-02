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
