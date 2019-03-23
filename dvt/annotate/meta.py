# -*- coding: utf-8 -*-
"""Annotators to supply metadata about the input.

This module contains functions that return metadata about the input sources.
Generally, these annotators only return information on the first batch of data
for the entire input. While not directly working with the batch of data, they
are organized as annotators to ease their inclusion into a pipeline of data
annotations.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the sample usage of FaceAnnotator over two batches of the input. A CNN
    model from dlib is used to detect the faces and the VGGFace2 algorithm is
    used to embed the faces into a 2048-dimensional space. The algorithm is
    applied to every 128 frames.

    >>> fp = FrameProcessor()
    >>> fp.load_annotator(VideoMetaAnnotator())
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    INFO:root:processed 00:00:00,000 to 00:00:17,083 with annotator: 'meta'
    INFO:root:processed 00:00:17,083 to 00:00:25,625 with annotator: 'meta'

    Then, collect the output from the annotator and display as a pandas data
    frame.

    >>> fp.collect("meta").todf()
            fps      vname  height   type  width  frames
    0  29.97003  input.mp4     480  video    708   42058

    The metadata output always gives a single line of data. These can be
    combined row-wise between mutiple runs of a pipeline.
"""

import numpy as np

from .core import FrameAnnotator


class MetaAnnotator(FrameAnnotator):
    """Return table of metadata about a input object.

    Attributes:
        meta (dict): A dictionary containing the metadata.
        pass_flag (bool): Indicates whether metadata has already been passed to
            the FrameAnnotator.
    """

    name = "meta"

    def __init__(self):
        self.meta = {}
        self.pass_flag = False
        super().__init__()

    def start(self, ival):
        """Grab metadata from the input.

        Args:
            ival: A FrameInput object.
        """
        self.meta = ival.meta
        self.meta["vname"] = ival.vname

    def annotate(self, batch):
        """Return metadata.

        Args:
            batch (FrameBatch): A batch of images to annotate. Not used in the
                function.

        Returns:
            Returns the metadata (as a list with one element) on the first call
            to this function and None otherwise.
        """
        if self.pass_flag:
            return None

        self.pass_flag = True
        return [self.meta]
