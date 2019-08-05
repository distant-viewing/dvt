# -*- coding: utf-8 -*-
"""A pipeline for building an interactive website from a video file.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of the VideoPipeline.

    >>> wp = VideoPipeline("video-clip.mp4", dname)
    >>> wp.make_breaks()
    >>> wp.run()

    After running there will be a new directory names "dvt-output-data" in the
    working directory. It contains a JSON file with extracted metadata, the
    extracted frames, annotated frames, and visualizations of the dense optical
    annotator. If you start a web browser on your machine:

    python3 -m http.server --directory dvt-output-data

    An interactive visualization of the output will be available by navigating
    to http://0.0.0.0:8000/ in your browser.
"""

import json
import os
import os.path

import numpy as np

from ..annotate.cielab import CIElabAnnotator
from ..annotate.core import FrameProcessor, FrameInput
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..annotate.meta import MetaAnnotator
from ..annotate.obj import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.opticalflow import OpticalFlowAnnotator
from ..annotate.png import PngAnnotator
from ..aggregate.display import DisplayAggregator
from ..aggregate.people import PeopleAggregator, make_fprint_from_images
from ..aggregate.length import ShotLengthAggregator
from ..utils import setup_tensorflow, _format_time, DictFrame
from .utils import _get_cuts, _get_meta
from .data import INDEX_MAIN, INDEX_PAGE, DVT_CSS, DVT_JS, DVT_MAIN_JS


class VideoPipeline:
    """Contains a predefined annotators to process an input video file.

    Attributes:
        finput (str): path to the input video file
        doutput (str): output directory. If set to None (default), will be
            a directory named "dvt-output-data" in the current working
            directory
        diff_co (int): difference cutoff value; integer from 0-256; higher
            values produce fewer cuts.
        cut_min_length (int): minimum length of a detected cut in frames;
            higher values produce few cuts.
        path_to_faces (str): Path to directory containing protype faces
            (optional). See tutorial on the commandline interface for more
            details.
    """

    def __init__(
        self,
        finput,
        doutput=None,
        diff_co=10,
        cut_min_length=30,
        path_to_faces=None
    ):

        setup_tensorflow()

        # find absolute path to the input and determine the output location
        finput = os.path.abspath(finput)
        fname = os.path.splitext(os.path.basename(finput))[0]
        if doutput is None:
            doutput = os.path.join(os.getcwd(), "dvt-output-data")

        self.finput = finput
        self.fname = fname
        self.nframes = _get_meta(self.finput)["frames"][0]
        self.doutput = os.path.join(doutput, fname)
        self.diff_co = diff_co
        self.cut_min_length = cut_min_length
        self.path_to_faces = path_to_faces

        self.cuts = None
        self.pipeline_data = None

    def make_breaks(self, freq=0):
        """Determine what frames to include in the output.

        Args:
            freq (int): set to a positive integer to select images based on
                frequency rather than detecting cuts; integer gives frequency
                of the sampling
        """
        if freq <= 0:
            self.cuts = _get_cuts(
                self.finput, self.diff_co, self.cut_min_length
            )
        else:
            cuts = DictFrame(
                {"frame_start": list(range(0, self.nframes - 1, freq))}
            )
            cuts["frame_end"] = [x + 1 for x in cuts["frame_start"][1:]]
            cuts["frame_end"].extend([self.nframes - 1])

            cuts["frame_start"] = np.array(cuts["frame_start"])
            cuts["frame_end"] = np.array(cuts["frame_end"])

            cuts["mpoint"] = (
                cuts["frame_start"]
                + (cuts["frame_end"] - cuts["frame_start"]) // 2
            )
            self.cuts = cuts

    def run(self, level=2):
        """Run the pipeline over the input video file.

        Args:
            level (int): interger code (0, 1, or 2) describing how much data
            to parse. 0 creates just metadata, 1 creates just images and
            metadata, 2 or more creates the interactive website
        """
        if not os.path.exists(self.doutput):
            os.makedirs(self.doutput)

        self._run_pipeline()
        self._make_json()

        if level >= 1:
            self._annotate_images()

        if level >= 2:
            self._json_toc()
            self._copy_web()

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

        if self.path_to_faces is not None:
            fembed, fnames = make_fprint_from_images(self.path_to_faces)
            fpobj.load_annotator(
                FaceAnnotator(
                    detector=FaceDetectMtcnn(),
                    embedding=FaceEmbedVgg2()
                    frames=frames
                )
            )
        else:
            fpobj.load_annotator(
                FaceAnnotator(detector=FaceDetectMtcnn(), frames=frames)
            )

        fri = FrameInput(self.finput, bsize=128)
        fpobj.process(fri)
        self.pipeline_data = fpobj.collect_all()

        self.pipeline_data["length"] = ShotLengthAggregator().aggregate(
            self.pipeline_data, frames=frames
        )

    def _annotate_images(self):
        img_output_dir = os.path.join(self.doutput, "img-anno")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        dagg = DisplayAggregator(
            input_dir=os.path.join(self.doutput, "img"),
            output_dir=img_output_dir,
        )
        dagg.aggregate(self.pipeline_data, frames=self.cuts["mpoint"])

    def _make_json(self):
        nframes = len(self.cuts["mpoint"])
        fps = self.pipeline_data["meta"]["fps"][0]
        ldata = self.pipeline_data["length"]

        if self.path_to_faces is not None:
            people = PeopleAggregator(self.pipeline_data)

        output = []
        for fnum in range(nframes):

            output += [
                {
                    "frame_start": int(self.cuts["frame_start"][fnum]),
                    "frame_end": int(self.cuts["frame_end"][fnum]),
                    "time_start": _format_time(
                        float(self.cuts["frame_start"][fnum]) / fps * 1000,
                    ),
                    "time_end": _format_time(
                        float(self.cuts["frame_end"][fnum]) / fps * 1000,
                    ),
                    "img_path": os.path.join(
                        "img",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"][fnum]),
                    ),
                    "anno_path": os.path.join(
                        "img-anno",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"][fnum]),
                    ),
                    "flow_path": os.path.join(
                        "img-flow",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"][fnum]),
                    ),
                    "dominant_colors": self.pipeline_data["cielab"][
                        "dominant_colors"
                    ][fnum].tolist(),
                    "num_faces": int(ldata["num_faces"][fnum]),
                    "num_people": int(ldata["num_people"][fnum]),
                    "largest_face": int(ldata["largest_face"][fnum]),
                    "largest_body": int(ldata["largest_body"][fnum]),
                    "obj_list": ldata["objects"][fnum],
                    "shot_length": ldata["shot_length"][fnum],
                }
            ]

        with open(os.path.join(self.doutput, "data.json"), "w") as fout:
            json.dump(output, fout, sort_keys=True, indent=4)

    def _json_toc(self):
        toc_path = os.path.join(os.path.dirname(self.doutput), "toc.json")

        data = []
        if os.path.exists(toc_path):
            with open(toc_path, "r") as finput:
                data = json.load(finput)

            data = [x for x in data if x["video_name"] != self.fname]

        miter = len(self.cuts["mpoint"]) // 2
        data.extend(
            [
                {
                    "thumb_path": os.path.join(
                        self.fname,
                        "img",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"][miter]),
                    ),
                    "video_name": self.fname,
                }
            ]
        )

        with open(toc_path, "w") as finput:
            json.dump(data, finput, indent=4)

    def _copy_web(self):
        toc_level = os.path.dirname(self.doutput)

        if not os.path.isdir(os.path.join(toc_level, "js")):
            os.makedirs(os.path.join(toc_level, "js"))

        if not os.path.isdir(os.path.join(toc_level, "css")):
            os.makedirs(os.path.join(toc_level, "css"))

        with open(os.path.join(self.doutput, "index.html"), "w") as fout:
            fout.write(INDEX_PAGE)

        with open(os.path.join(toc_level, "index.html"), "w") as fout:
            fout.write(INDEX_MAIN)

        with open(os.path.join(toc_level, "css", "dvt.css"), "w") as fout:
            fout.write(DVT_CSS)

        with open(os.path.join(toc_level, "js", "dvt.js"), "w") as fout:
            fout.write(DVT_JS)

        with open(os.path.join(toc_level, "js", "dvt-main.js"), "w") as fout:
            fout.write(DVT_MAIN_JS)
