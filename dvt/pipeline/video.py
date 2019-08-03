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
from ..annotate.meta import MetaAnnotator
from ..annotate.object import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.opticalflow import OpticalFlowAnnotator
from ..annotate.png import PngAnnotator
from ..aggregate.cut import CutAggregator
from ..aggregate.display import DisplayAggregator
from ..aggregate.length import ShotLengthAggregator
from ..utils import setup_tensorflow, _format_time, DictFrame
from .utils import _get_cuts


class VideoPipeline:
    """Contains a predefined annotators to process an input video file.
    """

    def __init__(self, finput, doutput=None, diff_co=10, cut_min_length=30):
        setup_tensorflow()

        # find absolute path to the input and determine the output location
        finput = os.path.abspath(finput)
        if doutput is None:
            doutput = os.path.join(os.getcwd(), "dvt-output")

        self.finput = finput
        self.doutput = doutput
        self.diff_co = diff_co
        self.cut_min_length = cut_min_length
        self.cuts = None
        self.pipeline_data = None

    def make_breaks(self, freq=0):
        if freq <= 0:
            self.cuts = _get_cuts(
                self.finput, self.diff_co, self.cut_min_length
            )
        else:
            nframes = _get_meta(self.finput)["frames"]
            cuts = DictFrame(
                {"frame_start": list(range(0, nframes - 1, freq))}
            )
            cuts["frame_end"] = [x + 1 for x in cuts["frame_start"][-1]]
            cuts["frame_end"].extend([nframes - 1])
            cuts["mpoint"] = (
                cuts["frame_start"]
                + (cuts["frame_end"] - cuts["frame_start"]) // 2
            )
            self.cuts = cuts

    def run(self, level=2):
        if not os.path.exists(self.doutput):
            os.makedirs(self.doutput)

        self._run_pipeline()
        self._make_json()

        if level >= 1:
            self._annotate_images()

        if level >= 2:
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
        fpobj.load_annotator(
            OpticalFlowAnnotator(
                output_dir=os.path.join(self.doutput, "img-flow"),
                frames=frames,
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
                    "anno_path": os.path.join(
                        "img-anno",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"][iter]),
                    ),
                    "flow_path": os.path.join(
                        "img-flow",
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
