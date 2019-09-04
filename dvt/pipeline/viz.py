# -*- coding: utf-8 -*-
"""A pipeline for building an interactive website from media files.

Offers similar functionality to the command line interface from within Python.
"""

from argparse import ArgumentParser
from json import dump, load
from os import getcwd, makedirs
from os.path import dirname, exists, isdir, join
from shutil import copyfile

from numpy import array, append, int32
from pandas import DataFrame

from ..abstract import Pipeline
from ..core import DataExtraction, FrameInput, ImageInput
from ..annotate.diff import DiffAnnotator
from ..annotate.cielab import CIElabAnnotator
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..annotate.img import ImgAnnotator
from ..annotate.obj import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.opticalflow import OpticalFlowAnnotator
from ..annotate.png import PngAnnotator
from ..aggregate.audio import SpectrogramAggregator, PowerToneAggregator
from ..aggregate.cut import CutAggregator
from ..aggregate.display import DisplayAggregator
from ..aggregate.people import PeopleAggregator, make_fprint_from_images
from ..aggregate.length import ShotLengthAggregator
from ..utils import (
    setup_tensorflow,
    get_data_location,
    _expand_path,
    _check_exists,
    _check_exists_dir,
    _find_containing_images
)


class VideoVizPipeline(Pipeline):
    """Contains a predefined annotators to process an input video file.

    Attributes:
        finput (str): path to the input video file
        dirout (str): output directory. If set to None (default), will be
            a directory named "dvt-output-data" in the current working
            directory
        pipeline_level (int): interger code (0, 1, or 2) describing how much
            data to parse. 0 creates just metadata, 1 creates just images and
            metadata, 2 or more creates the interactive website
        diff_cutoff (int): difference cutoff value; integer from 0-256; higher
            values produce fewer cuts.
        cut_min_length (int): minimum length of a detected cut in frames;
            higher values produce few cuts.
        freq (int): set to a positive integer to select images based on
            frequency rather than detecting cuts; integer gives frequency
            of the sampling
        path_to_faces (str): Path to directory containing protype faces
            (optional). See tutorial on the commandline interface for more
            details.
        path_to_audio (str): Path to a wav file with audio data. See tutorial
            on the commandline interface for more details.
        path_to_subtitle (str): Path to a src file with subtitle data. See
            tutorial on the commandline interface for more details.
    """

    def __init__(
        self,
        finput,
        dirout=None,
        pipeline_level=2,
        diff_cutoff=10,
        cut_min_length=30,
        frequency=0,
        path_to_faces=None,
        path_to_audio=None,
        path_to_subtitle=None
    ):

        setup_tensorflow()

        # create data extraction object and get metadata for the input
        input_path, _, fname, _ = _expand_path(finput)
        self.dextra = DataExtraction(FrameInput(
            input_path=input_path,
            bsize=128
        ), ainput=path_to_audio, sinput=path_to_subtitle)
        self.dextra.run_annotators([], max_batch=1, msg=None)
        self.pipeline_level = pipeline_level

        # process and prepare the output directory
        if dirout is None:
            dirout = join(getcwd(), "dvt-output-data")

        if not exists(dirout):
            makedirs(dirout)

        # grab parameters and store as class attribute
        self.attrib = {
            "finput": _check_exists(finput),
            "fname": fname,
            "dirout": join(dirout, fname),
            "diff_cutoff": diff_cutoff,
            "cut_min_length": cut_min_length,
            "path_to_faces": path_to_faces,
            "path_to_audio": path_to_audio,
            "path_to_subtitle": path_to_subtitle,
            "frequency": frequency,
        }

        super().__init__()

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
            input_dir=join(self.attrib['dirout'], "img", "frames"),
            output_dir=join(self.attrib['dirout'], "img", "display"),
            frames=self.dextra.data['cut']['mpoint'],
            size=250
        ))

        # if audio file, process
        if self.attrib['path_to_audio'] is not None:
            breaks = [0] + self.dextra.data['cut']['frame_end'].tolist()
            spec_output = join(self.attrib['dirout'], "img", "spec")
            tone_output = join(self.attrib['dirout'], "img", "tone")

            self.dextra.run_audio_annotator()
            self.dextra.run_aggregator(SpectrogramAggregator(
                output_dir=spec_output, breaks=breaks
            ))
            self.dextra.run_aggregator(PowerToneAggregator(
                output_dir=tone_output, breaks=breaks
            ))
            self.dextra.data.pop('audio')
            self.dextra.data.pop('audiometa')

        # if subtitle file, process
        if self.attrib['path_to_subtitle'] is not None:
            self.dextra.run_subtitle_annotator()

        # output dataset as a JSON file
        self.dextra.get_json(
            join(self.attrib['dirout'], "data", "viz-data.json"),
            exclude_set=["opticalflow"]
        )
        miter = len(self.dextra.get_data()["cut"]["mpoint"].values) // 2
        _json_toc(
            self.attrib['dirout'],
            self.attrib['fname'],
            self.dextra.get_data()["cut"]["mpoint"].values[miter]
        )

        # if desired, push web-specific files
        if self.pipeline_level >= 2:
            _copy_web(self.attrib['dirout'], "index-video.html")

    @staticmethod
    def get_argparser():
        parser = ArgumentParser(
            prog="python3 -m dvt video-viz",
            description="Given a single video file, this pipeline selects a "
            "set of frames (by default, one frame in each shot) and extracts "
            "various metadata using the toolkit's annotators and aggregators. "
            "The output is stored locally as JSON and PNG files that can be "
            "viewed locally using the interactive web interface.",
        )
        parser.add_argument(
            "finput", type=str, help="path to the video file to process"
        )
        parser.add_argument(
            "--dirout",
            "-d",
            type=str,
            help="base directory in which to store the output",
            default="dvt-output-data",
        )
        parser.add_argument(
            "--pipeline-level",
            "-l",
            type=int,
            help="interger code (0, 1, or 2) describing how much data to "
            "parse; 0 creates just metadata, 1 creates just images and "
            "metadata, 2 or more creates the interactive website "
            "(default: 2)",
            default=2,
        )
        parser.add_argument(
            "--diff-cutoff",
            "-dc",
            type=int,
            help="difference cutoff value; integer from 0-256; higher values "
            "produce fewer cuts (default: 10)",
            default=10,
        )
        parser.add_argument(
            "--cut-min-length",
            "-cml",
            type=int,
            help="minimum length of a detected cut, frames; higher values "
            "produce fewer cuts (default: 30)",
            default=30,
        )
        parser.add_argument(
            "--frequency",
            "-f",
            type=int,
            help="set to a positive integer to select images based on "
            "frequency rather than detecting cuts; integer gives frequency of "
            "the sampling (default: 0)",
            default=0,
        )
        parser.add_argument(
            "--path-to-faces",
            type=str,
            default=None,
            help="Path to directory containing protype faces (optional). See "
            "tutorial on the commandline interface for more details.",
        )
        parser.add_argument(
            "--path-to-audio",
            type=str,
            default=None,
            help="Path to a wav file corresponding to the video input "
            "(optional). See tutorial on the commandline interface for more "
            "details.",
        )
        parser.add_argument(
            "--path-to-subtitle",
            type=str,
            default=None,
            help="Path to a srt file corresponding to the subtitle input "
            "(optional). See tutorial on the commandline interface for more "
            "details.",
        )

        return parser

    def _make_breaks(self):
        """Determine what frames to include in the output.
        """

        if self.attrib['frequency'] <= 0:
            self.dextra.run_annotators([
                DiffAnnotator(quantiles=[40])
            ], msg=None)
            self.dextra.run_aggregator(
                CutAggregator(
                    cut_vals={"q40": self.attrib['diff_cutoff']},
                    min_len=self.attrib['cut_min_length'])
            )
        else:
            nframes = self.dextra.get_data()["meta"]['frames'].values[0]
            frame_start = array(range(
                0, nframes - 1, self.attrib['frequency']
            ))
            frame_end = append(frame_start[1:] + 1, nframes - 1)

            self.dextra.data['cut'] = DataFrame({
                "frame_start": frame_start,
                "frame_end": frame_end
            })

        fstart = self.dextra.data['cut'].frame_start.values
        fend = self.dextra.data['cut'].frame_end.values

        self.dextra.data['cut']['mpoint'] = int32(
            fstart + (fend - fstart) // 2
        )

    def _run_annotation(self):
        frames = self.dextra.data['cut']['mpoint']

        annotators = [
            CIElabAnnotator(frames=frames, num_dominant=5),
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), frames=frames)
        ]

        if self.pipeline_level >= 1:
            annotators.append(PngAnnotator(
                output_dir=join(self.attrib['dirout'], "img", "frames"),
                frames=frames
            ))
            thumb = PngAnnotator(
                output_dir=join(self.attrib['dirout'], "img", "thumb"),
                frames=frames,
                size=150
            )
            thumb.name = "thumb"
            annotators.append(thumb)
            annotators.append(OpticalFlowAnnotator(
                output_dir=join(self.attrib['dirout'], "img", "flow"),
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

        self.dextra.run_annotators(
            annotators,
            msg="Processing annotators: "
        )

    def _proc_faces(self):
        fembed, fnames = make_fprint_from_images(self.attrib['path_to_faces'])

        frames = self.dextra.data['cut']['mpoint']
        self.dextra.run_aggregator(PeopleAggregator(
            face_names=fnames,
            fprint=fembed,
            frames=frames
        ))


class ImageVizPipeline(Pipeline):
    """Contains a predefined annotators to process a corpus of images.

    Attributes:
        dinput (str): directory of the input images. Will take all image
            files (png, jpg, jpeg, tif, tiff) from within the parent directory.
        dirout (str): output directory. If set to None (default), will be
            a directory named "dvt-output-data" in the current working
            directory
        pipeline_level (int): interger code (0, 1, or 2) describing how much
            data to parse. 0 creates just metadata, 1 creates just images and
            metadata, 2 or more creates the interactive website
        path_to_faces (str): Path to directory containing protype faces
            (optional). See tutorial on the commandline interface for more
            details.
    """

    def __init__(
        self,
        dinput,
        dirout=None,
        pipeline_level=2,
        path_to_faces=None
    ):

        setup_tensorflow()

        # create data extraction object and get metadata for the input
        dinput = _check_exists_dir(dinput)
        finput = _find_containing_images(dinput)
        fname = _expand_path(dinput)[1]

        self.dextra = DataExtraction(ImageInput(input_paths=finput))
        self.dextra.run_annotators([], max_batch=1, msg=None)
        self.pipeline_level = pipeline_level

        # process and prepare the output directory
        if dirout is None:
            dirout = join(getcwd(), "dvt-output-data")

        if not exists(dirout):
            makedirs(dirout)

        # grab parameters and store as class attribute
        self.attrib = {
            "fname": fname,
            "finput": finput,
            "dirout": join(dirout, fname),
            "path_to_faces": path_to_faces
        }

        super().__init__()

    def run(self):
        """Run the pipeline over the input image objects.
        """
        # run the main annotations
        self._run_annotation()

        # run aggregators
        if self.attrib['path_to_faces'] is not None:
            self._proc_faces()

        self.dextra.run_aggregator(ShotLengthAggregator(
            frames=list(range(len(self.attrib['finput'])))
        ))

        self.dextra.run_aggregator(DisplayAggregator(
            input_dir=join(self.attrib['dirout'], "img", "frames"),
            output_dir=join(self.attrib['dirout'], "img", "display"),
            size=250
        ))

        # output dataset as a JSON file
        self.dextra.get_json(
            join(self.attrib['dirout'], "data", "viz-data.json")
        )
        _json_toc(
            self.attrib['dirout'],
            self.attrib['fname'],
            0
        )

        # if desired, push web-specific files
        if self.pipeline_level >= 2:
            _copy_web(self.attrib['dirout'], "index-image.html")

    @staticmethod
    def get_argparser():
        parser = ArgumentParser(
            prog="python3 -m dvt video-viz",
            description="Given a single video file, this pipeline selects a "
            "set of frames (by default, one frame in each shot) and extracts "
            "metadata using the toolkit's annotators and aggregators. "
            "The output is stored locally as JSON and PNG files that can be "
            "viewed locally using the interactive web interface.",
        )
        parser.add_argument(
            "dinput",
            type=str,
            help="directory of the input images. Will take all image files "
                 "(png, jpg, jpeg, tif, tiff) from within this parent"
                 "directory."
        )
        parser.add_argument(
            "--dirout",
            "-d",
            type=str,
            help="base directory in which to store the output",
            default="dvt-output-data",
        )
        parser.add_argument(
            "--pipeline-level",
            "-l",
            type=int,
            help="interger code (0, 1, or 2) describing how much data to "
            "parse; 0 creates just metadata, 1 creates just images and "
            "metadata, 2 or more creates the interactive website "
            "(default: 2)",
            default=2,
        )
        parser.add_argument(
            "--path-to-faces",
            type=str,
            default=None,
            help="Path to directory containing protype faces (optional). See "
            "tutorial on the commandline interface for more details.",
        )

        return parser

    def _run_annotation(self):

        annotators = [
            CIElabAnnotator(num_dominant=5),
            ObjectAnnotator(detector=ObjectDetectRetinaNet()),
            ImgAnnotator()
        ]

        if self.pipeline_level >= 1:
            annotators.append(PngAnnotator(
                output_dir=join(self.attrib['dirout'], "img", "frames")
            ))
            thumb = PngAnnotator(
                output_dir=join(self.attrib['dirout'], "img", "thumb"),
                size=150
            )
            thumb.name = "thumb"
            annotators.append(thumb)

        if self.attrib['path_to_faces'] is not None:
            annotators.append(FaceAnnotator(
                    detector=FaceDetectMtcnn(),
                    embedding=FaceEmbedVgg2()
            ))
        else:
            annotators.append(FaceAnnotator(
                detector=FaceDetectMtcnn()
            ))

        self.dextra.run_annotators(
            annotators,
            msg="Processing annotators: "
        )

    def _proc_faces(self):
        fembed, fnames = make_fprint_from_images(self.attrib['path_to_faces'])

        self.dextra.run_aggregator(PeopleAggregator(
            face_names=fnames,
            fprint=fembed
        ))


def _json_toc(dirout, fname, frame_num):
    toc_path = join(dirname(dirout), "toc.json")

    data = []
    if exists(toc_path):
        with open(toc_path, "r") as finput:
            data = load(finput)

        data = [x for x in data if x["video_name"] != fname]

    data.extend(
        [
            {
                "thumb_path": join(
                    fname,
                    "img",
                    "frames",
                    "frame-{0:06d}.png".format(frame_num),
                ),
                "video_name": fname,
                "video_name_long": fname
            }
        ]
    )

    with open(toc_path, "w") as finput:
        dump(data, finput, indent=4)


def _copy_web(dirout, index_name="index-video.html"):
    data_dir = get_data_location()
    toc_level = dirname(dirout)

    if not isdir(join(toc_level, "js")):
        makedirs(join(toc_level, "js"))

    if not isdir(join(toc_level, "css")):
        makedirs(join(toc_level, "css"))

    copyfile(join(data_dir, "html", "index-main.html"),
             join(toc_level, "index.html"))

    copyfile(join(data_dir, "html", index_name),
             join(dirout, "index.html"))

    copyfile(join(data_dir, "css", "dvt.css"),
             join(toc_level, "css", "dvt.css"))

    copyfile(join(data_dir, "js", "dvt.js"),
             join(toc_level, "js", "dvt.js"))
