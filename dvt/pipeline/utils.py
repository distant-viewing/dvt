# -*- coding: utf-8 -*-
"""Utility functions for working with data pipelines.
"""

import numpy as np

from ..annotate.core import FrameProcessor, FrameInput
from ..annotate.diff import DiffAnnotator
from ..annotate.meta import MetaAnnotator
from ..aggregate.cut import CutAggregator


def _get_cuts(finput, diff_co=10, cut_min_length=30):
    fpobj = FrameProcessor()
    fpobj.load_annotator(DiffAnnotator(quantiles=[40]))
    fri = FrameInput(finput, bsize=128)
    fpobj.process(fri)
    obj_out = fpobj.collect_all()

    ca = CutAggregator(cut_vals={"q40": diff_co}, min_len=cut_min_length)
    agg = ca.aggregate(obj_out)
    agg["frame_start"] = np.array(agg["frame_start"])
    agg["frame_end"] = np.array(agg["frame_end"])
    agg["mpoint"] = (
        agg["frame_start"] + (agg["frame_end"] - agg["frame_start"]) // 2
    )
    return agg


def _get_meta(finput):
    fpobj = FrameProcessor()
    fpobj.load_annotator(MetaAnnotator())
    fri = FrameInput(finput, bsize=2)
    fpobj.process(fri, max_batch=1)
    return fpobj.collect("meta")
