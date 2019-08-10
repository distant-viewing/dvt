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
from pandas import DataFrame

from ..core import DataExtraction
from ..input import FrameInput
from ..annotate.cielab import CIElabAnnotator
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..annotate.obj import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.opticalflow import OpticalFlowAnnotator
from ..annotate.png import PngAnnotator
from ..aggregate.display import DisplayAggregator
from ..aggregate.people import PeopleAggregator, make_fprint_from_images
from ..aggregate.length import ShotLengthAggregator
from ..utils import setup_tensorflow
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
            frame_start = np.array(range(0, self.nframes - 1, freq))
            frame_end = np.append(frame_start[1:] + 1, self.nframes - 1)

            self.cuts = DataFrame({
                "frame_start": frame_start,
                "frame_end": frame_end,
                "mpoint": frame_start + (frame_end - frame_start) // 2
            })

        self.cuts["mpoint"] = np.int32(self.cuts["mpoint"].values)

    def run(self, level=2):
        """Run the pipeline over the input video file.

        Args:
            level (int): interger code (0, 1, or 2) describing how much data
            to parse. 0 creates just metadata, 1 creates just images and
            metadata, 2 or more creates the interactive website
        """
        if not os.path.exists(self.doutput):
            os.makedirs(self.doutput)

        self._run_pipeline(level=level)
        self._make_json()

        if level >= 2:
            self._json_toc()
            self._copy_web()

    def _run_pipeline(self,level):
        frames = self.cuts["mpoint"].values

        if self.path_to_faces is not None:
            face_anno = FaceAnnotator(
                    detector=FaceDetectMtcnn(),
                    embedding=FaceEmbedVgg2(),
                    frames=frames
            )
        else:
            face_anno = FaceAnnotator(
                detector=FaceDetectMtcnn(), frames=frames
            )

        de = DataExtraction(FrameInput(input_path=self.finput, bsize=128))
        de.run_annotators([
            PngAnnotator(
                output_dir=os.path.join(self.doutput, "img"), frames=frames
            ),
            OpticalFlowAnnotator(
                output_dir=os.path.join(self.doutput, "img-flow"),
                frames=frames,
            ),
            CIElabAnnotator(frames=frames, num_dominant=5),
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), frames=frames),
            face_anno
        ])

        de.run_aggregator(ShotLengthAggregator(frames=frames))

        if self.path_to_faces is not None:
            fembed, fnames = make_fprint_from_images(self.path_to_faces)

            de.run_aggregator(PeopleAggregator(
                face_names=fnames,
                fprint=fembed,
                frames=frames
            ))

        self.pipeline_data = de.get_data()

        if level >= 1:
            img_output_dir = os.path.join(self.doutput, "img-anno")
            if not os.path.exists(img_output_dir):
                os.makedirs(img_output_dir)

            de.run_aggregator(DisplayAggregator(
                input_dir=os.path.join(self.doutput, "img"),
                output_dir=img_output_dir,
                frames=frames
            ))

    def _make_json(self):
        nframes = len(self.cuts["mpoint"])
        fps = self.pipeline_data["meta"]["fps"].values[0]
        ldata = self.pipeline_data["length"]

        output = []
        for fnum in range(nframes):

            output += [
                {
                    "frame_start": int(self.cuts["frame_start"].values[fnum]),
                    "frame_end": int(self.cuts["frame_end"].values[fnum]),
                    "time_start": float(
                        self.cuts["frame_start"].values[fnum]
                    ) / fps * 1000,
                    "time_end": float(
                        self.cuts["frame_end"].values[fnum]
                    ) / fps * 1000,
                    "img_path": os.path.join(
                        "img",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"].values[fnum]),
                    ),
                    "anno_path": os.path.join(
                        "img-anno",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"].values[fnum]),
                    ),
                    "flow_path": os.path.join(
                        "img-flow",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"].values[fnum]),
                    ),
                    "dominant_colors": self.pipeline_data["cielab"][
                        "dominant_colors"
                    ].values[fnum].tolist(),
                    "num_faces": int(ldata["num_faces"].values[fnum]),
                    "num_people": int(ldata["num_people"].values[fnum]),
                    "largest_face": int(ldata["largest_face"].values[fnum]),
                    "largest_body": int(ldata["largest_body"].values[fnum]),
                    "obj_list": ldata["objects"].values[fnum],
                    "people_list": ldata["people"].values[fnum],
                    "shot_length": ldata["shot_length"].values[fnum],
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

        miter = len(self.cuts["mpoint"].values) // 2
        data.extend(
            [
                {
                    "thumb_path": os.path.join(
                        self.fname,
                        "img",
                        "frame-{0:06d}.png".format(self.cuts["mpoint"].values[miter]),
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
