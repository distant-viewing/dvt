# -*- coding: utf-8 -*-
"""Utility functions for working with data pipelines.
"""

from ..core import DataExtraction
from ..inputs import FrameInput
from ..annotate.diff import DiffAnnotator
from ..aggregate.cut import CutAggregator


def _get_cuts(finput, diff_co=10, cut_min_length=30):
    dextra = DataExtraction(FrameInput(input_path=finput, bsize=128))
    dextra.run_annotators([DiffAnnotator(quantiles=[40])])
    dextra.run_aggregator(
        CutAggregator(cut_vals={"q40": diff_co}, min_len=cut_min_length)
    )

    agg = dextra.get_data()['cut']
    agg["mpoint"] = (
        agg.frame_start.values +
        (agg.frame_end.values - agg.frame_start.values) / 2
    )
    return agg


def _get_meta(finput):
    dextra = DataExtraction(FrameInput(input_path=finput, bsize=2))
    dextra.run_annotators([], max_batch=1)

    return dextra.get_data()["meta"]
