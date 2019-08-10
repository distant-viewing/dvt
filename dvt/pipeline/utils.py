# -*- coding: utf-8 -*-
"""Utility functions for working with data pipelines.
"""

import numpy as np

from ..core import DataExtraction
from ..input import FrameInput
from ..annotate.diff import DiffAnnotator
from ..aggregate.cut import CutAggregator


def _get_cuts(finput, diff_co=10, cut_min_length=30):
    de = DataExtraction(FrameInput(input_path=finput, bsize=128))
    de.run_annotators([DiffAnnotator(quantiles=[40])])
    de.run_aggregator(
        CutAggregator(cut_vals={"q40": diff_co}, min_len=cut_min_length)
    )

    agg = de.get_data()['cut']
    agg["mpoint"] = (agg.frame_start.values +
        (agg.frame_end.values - agg.frame_start.values) / 2
    )
    return agg


def _get_meta(finput):
    de = DataExtraction(FrameInput(input_path=finput, bsize=2))
    de.run_annotators([], max_batch=1)

    return de.get_data()["meta"]
