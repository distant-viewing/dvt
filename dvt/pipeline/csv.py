# -*- coding: utf-8 -*-
"""A pipeline for building an interactive website from media files.

Offers similar functionality to the command line interface from within Python.
"""

from argparse import ArgumentParser
from os import getcwd, makedirs
from os.path import exists, join

from numpy import array, append, int32
from pandas import DataFrame

from ..abstract import Pipeline
from ..core import DataExtraction, FrameInput
from ..annotate.diff import DiffAnnotator
from ..annotate.cielab import CIElabAnnotator
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..annotate.obj import ObjectAnnotator, ObjectDetectRetinaNet
from ..aggregate.audio import PowerToneAggregator
from ..aggregate.cut import CutAggregator
from ..aggregate.people import PeopleAggregator, make_fprint_from_images
from ..aggregate.length import ShotLengthAggregator
from ..utils import (
    setup_tensorflow,
    _expand_path,
    _check_exists,
    _check_out_dir
)


class VideoCsvPipeline(Pipeline):
    """Contains annotators to return structured data about video inputs.

    Attributes:
        finput (str): path to the input video file
        dirout (str): output directory. If set to None (default), will be
            a directory named "dvt-output-data" in the current working
            directory
        diff_cutoff (int): difference cutoff value; integer from 0-256; higher
            values produce fewer cuts.
        cut_min_length (int): minimum length of a detected cut in frames;
            higher values produce few cuts.
        freq (int): set to a positive integer to select images based on
            frequency rather than detecting cuts; integer gives frequency
            of the sampling
        path_to_faces (str): Path to directory containing protype faces
            (optional).
        path_to_audio (str): Path to a wav file with audio data. See tutorial
            on the commandline interface for more details.
        path_to_subtitle (str): Path to a src file with subtitle data. See
            tutorial on the commandline interface for more details.
    """

    def __init__(
        self,
        finput,
        dirout=None,
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

        # if audio file, process
        if self.attrib['path_to_audio'] is not None:
            breaks = [0] + self.dextra.data['cut']['frame_end'].tolist()

            self.dextra.run_audio_annotator()
            self.dextra.run_aggregator(PowerToneAggregator(breaks=breaks))
            self.dextra.data.pop('audio')
            self.dextra.data.pop('audiometa')

        # if subtitle file, process
        if self.attrib['path_to_subtitle'] is not None:
            self.dextra.run_subtitle_annotator()

        if not exists(join(self.attrib['dirout'], "csv")):
            makedirs(join(self.attrib['dirout'], "csv"))

        # output dataset as CSV files
        ldframe = self.dextra.get_data()
        _check_out_dir(join(self.attrib['dirout'], "data"))
        for key, value in ldframe.items():
            value.to_csv(
                path_or_buf=join(self.attrib['dirout'], "data", key + ".csv"),
                index=False
            )

    @staticmethod
    def get_argparser():
        parser = ArgumentParser(
            prog="python3 -m dvt video-viz",
            description="Given a single video file, this pipeline extracts "
            "metadata and saves the output as a set of CSV files. This is a "
            "useful format for later computational analysis.",
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
