# -*- coding: utf-8 -*-
"""A pipeline for building an interactive website from a video file.

Offers similar functionality to the command line interface from within Python.
"""

from json import dump, load
from os import getcwd, makedirs
from os.path import abspath, basename, dirname, exists, isdir, join, splitext
from shutil import copyfile

from numpy import array, append, int32
from pandas import DataFrame

from ..core import DataExtraction, FrameInput
from ..annotate.diff import DiffAnnotator
from ..annotate.cielab import CIElabAnnotator
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..annotate.obj import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.opticalflow import OpticalFlowAnnotator
from ..annotate.png import PngAnnotator
from ..aggregate.cut import CutAggregator
from ..aggregate.display import DisplayAggregator
from ..aggregate.people import PeopleAggregator, make_fprint_from_images
from ..aggregate.length import ShotLengthAggregator
from ..utils import setup_tensorflow, get_data_location, _expand_path
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
        level (int): interger code (0, 1, or 2) describing how much data
            to parse. 0 creates just metadata, 1 creates just images and
            metadata, 2 or more creates the interactive website
        freq (int): set to a positive integer to select images based on
            frequency rather than detecting cuts; integer gives frequency
            of the sampling
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
        level=2,
        freq=0,
        path_to_faces=None
    ):

        setup_tensorflow()

        # create data extraction object and get metadata for the input
        input_path, _, fname, _ = _expand_path(finput)
        self.dextra = DataExtraction(FrameInput(
            input_path=input_path,
            bsize=128
        ))
        self.dextra.run_annotators([], max_batch=1)
        self.level = level

        # process and prepare the output directory
        if doutput is None:
            doutput = join(getcwd(), "dvt-output-data")

        if not exists(doutput):
            makedirs(doutput)

        # grab parameters and store as class attribute
        self.attrib = {
            "finput": finput,
            "fname": fname,
            "doutput": join(doutput, fname),
            "diff_co": diff_co,
            "cut_min_length": cut_min_length,
            "path_to_faces": path_to_faces,
            "freq": freq,
        }

    def run(self):
        """Run the pipeline over the input video object.
        """
        # determine frames to process and run the main annotations
        self._make_breaks()
        self._run_annotation()

        # run aggregators
        if self.attrib['path_to_faces'] is not None:
            self._proc_faces()

        self.dextra.run_aggregator(ShotLengthAggregator(
            frames=self.dextra.data['cut']['mpoint']
        ))

        self.dextra.run_aggregator(DisplayAggregator(
            input_dir=join(self.attrib['doutput'], "img"),
            output_dir=join(self.attrib['doutput'], "img-display"),
            frames=self.dextra.data['cut']['mpoint'],
            size=250
        ))

        # output dataset as a JSON file
        self.dextra.get_json(
            join(self.attrib['doutput'], "data.json"),
            exclude_set=["opticalflow"]
        )
        self._json_toc()

        # if desired, push web-specific files
        if self.level >= 2:
            self._copy_web()

    def _make_breaks(self):
        """Determine what frames to include in the output.
        """

        if self.attrib['freq'] <= 0:
            self.dextra.run_annotators([DiffAnnotator(quantiles=[40])])
            self.dextra.run_aggregator(
                CutAggregator(
                    cut_vals={"q40": self.attrib['diff_co']},
                    min_len=self.attrib['cut_min_length'])
            )
        else:
            nframes = self.dextra.get_data()["meta"]['frames'].values[0]
            frame_start = array(range(0, nframes - 1, self.attrib['freq']))
            frame_end = append(frame_start[1:] + 1, nframes - 1)

            self.dextra.data['cut'] = DataFrame({
                "frame_start": frame_start,
                "frame_end": frame_end
            })

        fs = self.dextra.data['cut'].frame_start.values
        fe = self.dextra.data['cut'].frame_end.values

        self.dextra.data['cut']['mpoint'] = int32(
            fs + (fe - fs) // 2
        )

    def _run_annotation(self):
        frames = self.dextra.data['cut']['mpoint']

        annotators = [
            CIElabAnnotator(frames=frames, num_dominant=5),
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), frames=frames)
        ]

        if self.level >= 1:
            annotators.append(PngAnnotator(
                output_dir=join(self.attrib['doutput'], "img"), frames=frames
            ))
            thumb = PngAnnotator(
                output_dir=join(self.attrib['doutput'], "img-thumb"),
                frames=frames,
                size=150
            )
            thumb.name = "thumb"
            annotators.append(thumb)
            annotators.append(OpticalFlowAnnotator(
                output_dir=join(self.attrib['doutput'], "img-flow"),
                frames=frames,
                size=150
            ))

        if self.attrib['path_to_faces'] is not None:
            annotators.append(FaceAnnotator(
                    detector=FaceDetectMtcnn(),
                    embedding=FaceEmbedVgg2(),
                    frames=frames
            ))
        else:
            annotators.append(FaceAnnotator(
                detector=FaceDetectMtcnn(), frames=frames
            ))

        self.dextra.run_annotators(annotators)

    def _proc_faces(self):
        fembed, fnames = make_fprint_from_images(self.attrib['path_to_faces'])

        frames = self.dextra.data['cut']['mpoint']
        self.dextra.run_aggregator(PeopleAggregator(
            face_names=fnames,
            fprint=fembed,
            frames=frames
        ))

    def _json_toc(self):
        toc_path = join(dirname(self.attrib['doutput']), "toc.json")

        data = []
        if exists(toc_path):
            with open(toc_path, "r") as finput:
                data = load(finput)

            data = [x for x in data if x["video_name"] != self.attrib['fname']]

        miter = len(self.dextra.get_data()["cut"]["mpoint"].values) // 2
        mframe = self.dextra.get_data()["cut"]["mpoint"].values[miter]
        data.extend(
            [
                {
                    "thumb_path": join(
                        self.attrib['fname'],
                        "img",
                        "frame-{0:06d}.png".format(mframe),
                    ),
                    "video_name": self.attrib['fname'],
                    "video_name_long": self.attrib['fname']
                }
            ]
        )

        with open(toc_path, "w") as finput:
            dump(data, finput, indent=4)

    def _copy_web(self):
        data_dir = get_data_location()
        toc_level = dirname(self.attrib['doutput'])

        if not isdir(join(toc_level, "js")):
            makedirs(join(toc_level, "js"))

        if not isdir(join(toc_level, "css")):
            makedirs(join(toc_level, "css"))

        copyfile(join(data_dir, "html", "index-main.html"),
                 join(toc_level, "index.html"))

        copyfile(join(data_dir, "html", "index-video.html"),
                 join(self.attrib['doutput'], "index.html"))

        copyfile(join(data_dir, "css", "dvt.css"),
                 join(toc_level, "css", "dvt.css"))

        copyfile(join(data_dir, "js", "dvt.js"),
                 join(toc_level, "js", "dvt.js"))

        copyfile(join(data_dir, "js", "dvt-main.js"),
                 join(toc_level, "js", "dvt-main.js"))
