# -*- coding: utf-8 -*-
"""A pipeline for building an interactive website from a video file.
"""

import json
import os
import os.path

import numpy as np

from ..annotate.cielab import CIElabAnnotator
from ..annotate.diff import DiffAnnotator
from ..annotate.core import FrameProcessor, FrameInput, ImageInput
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceDetectDlib
from ..annotate.embed import EmbedAnnotator, EmbedFrameKerasResNet50
from ..annotate.meta import MetaAnnotator
from ..annotate.object import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.png import PngAnnotator
from ..aggregate.cut import CutAggregator
from ..aggregate.display import DisplayAggregator
from ..aggregate.length import ShotLengthAggregator
from ..utils import setup_tensorflow, _format_time
from .utils import _get_cuts


class VideoPipeline:
    """Contains
    """

    def __init__(self, finput, doutput=None):
        # setup tensorflow
        setup_tensorflow()

        # find absolute path to the input and determine where the output will
        # live
        finput = os.path.abspath(finput)
        if doutput is None:
            doutput = os.path.join(os.getcwd(), "dvt-output")
        self.finput = finput
        self.doutput = doutput
        self.cuts = None
        self.pipeline_data = None

    def run(self):
        if not os.path.exists(self.doutput):
            os.makedirs(self.doutput)

        self.cuts = _get_cuts(self.finput)
        self._run_pipeline()
        self._annotate_images()
        self._make_json()

        _copy_web()

    def _run_pipeline(self):
        frames = self.cuts["mpoint"]

        fpobj = FrameProcessor()
        fpobj.load_annotator(MetaAnnotator())
        fpobj.load_annotator(
            PngAnnotator(
                output_dir=os.path.join(self.doutput, "img"), frames=frames
            )
        )
        fpobj.load_annotator(CIElabAnnotator(frames=frames, num_dominant=5))
        fpobj.load_annotator(
            FaceAnnotator(detector=FaceDetectMtcnn(), frames=frames)
        )
        fpobj.load_annotator(
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), frames=frames)
        )

        fri = FrameInput(self.finput, bsize=128)
        fpobj.process(fri)
        self.pipeline_data = fpobj.collect_all()

        self.pipeline_data["length"] = ShotLengthAggregator().aggregate(
            self.pipeline_data, frames
        )

    def _annotate_images(self):
        img_output_dir = os.path.join(self.doutput, "img-anno")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        da = DisplayAggregator(
            input_dir=os.path.join(self.doutput, "img"),
            output_dir=img_output_dir,
        )
        da.aggregate(self.pipeline_data, self.cuts["mpoint"])

    def _make_json(self):
        nframes = len(self.cuts["mpoint"])
        fps = self.pipeline_data["meta"]["fps"][0]
        ldata = self.pipeline_data["length"]

        output = []
        for iter in range(nframes):
            frame = self.cuts["mpoint"][iter]

            output += [
                {
                    "frame_start": int(self.cuts["frame_start"][iter]),
                    "frame_end": int(self.cuts["frame_end"][iter]),
                    "time_start": _format_time(
                        float(self.cuts["frame_start"][iter]) / fps * 1000,
                        include_msec=False,
                    ),
                    "time_end": _format_time(
                        float(self.cuts["frame_end"][iter]) / fps * 1000,
                        include_msec=False,
                    ),
                    "img_path": os.path.join(
                        "img",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"][iter]),
                    ),
                    "dominant_colors": self.pipeline_data["cielab"][
                        "dominant_colors"
                    ][iter].tolist(),
                    "num_faces": int(ldata["num_faces"][iter]),
                    "num_people": int(ldata["num_people"][iter]),
                    "largest_face": int(ldata["largest_face"][iter]),
                    "largest_body": int(ldata["largest_body"][iter]),
                    "obj_list": ldata["objects"][iter],
                    "shot_length": ldata["shot_length"][iter],
                }
            ]

        with open(os.path.join(self.doutput, "data.json"), "w") as fout:
            json.dump(output, fout, sort_keys=True, indent=4)


def _copy_web():
    pass
