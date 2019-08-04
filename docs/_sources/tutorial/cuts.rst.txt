
.. highlight:: python


Annotator Tutorial
#####################

This tutorial shows how to detect cuts in an input video source using the
annotators and cut aggregator.
The following commands assume that you have installed the dvt toolkit, are
in a running Python environment, and have a video file [video-clip.mp4](https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4)
in your working directory.

To start, load the following from the toolkit and turn on logging::

    from dvt.annotate.core import FrameProcessor, FrameInput
    from dvt.annotate.diff import DiffAnnotator
    from dvt.annotate.face import FaceAnnotator, FaceDetectDlib, FaceEmbedVgg2
    from dvt.annotate.meta import MetaAnnotator
    from dvt.annotate.png import PngAnnotator
    from dvt.aggregate.cut import CutAggregator

    import logging
    logging.basicConfig(level='DEBUG')
