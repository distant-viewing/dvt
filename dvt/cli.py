# -*- coding: utf-8 -*-
"""Command line tool for evoking the toolkit.

This tool allows users to quickly apply pre-build pipelines to one or more
media files.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the default usage of the command line interface:

    > python3 -m dvt video-clip.mp4

    This may take several minutes to complete. Some minimal logging information
    should display the annotators progress in your terminal. Once finished, you
    should have a new directory dvt-output-data that contains extracted
    metadata and frames from the source material. You can view the extracted
    information by starting a local http server:

    > python3 -m http.server --directory dvt-output-data

    And opening the following: http://0.0.0.0:8000/.
"""

from argparse import ArgumentParser
from glob import glob
from logging import basicConfig
from os.path import abspath, splitext, basename

from .pipeline.video import VideoPipeline


def run_cli():
    """Command line tool for evoking the toolkit.

    This functions is not intended to be called directly from an external
    program. Run with caution.
    """
    parser = _get_arg_parse()
    args = parser.parse_args()

    if not args.quiet:
        basicConfig(level="INFO")

    for vin in _expand_path_and_check(args.inputs):

        vpipe = VideoPipeline(
            vin,
            doutput=args.dirout,
            diff_co=args.diff_cutoff,
            cut_min_length=args.cut_min_length,
            path_to_faces=args.path_to_faces
        )
        vpipe.make_breaks(freq=args.frequency)
        vpipe.run(level=args.pipeline_level)


def _get_arg_parse():
    parser = ArgumentParser(
        prog="python3 -m dvt",
        description="The Distant Viewing Toolkit command line interface makes "
        "it possible to quickly apply pre-build pipelines to one or more "
        "media files. Several options exist to control the way the output is "
        "produced. See the lower-level functions for more flexibility.",
    )
    parser.add_argument(
        "inputs", nargs="+", help="one or more media files to process"
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
        help="interger code (0, 1, or 2) describing how much data to parse; "
        "0 creates just metadata, 1 creates just images and metadata, 2 or "
        "more creates the interactive website (default: 2)",
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
        help="minimum length of a detected cut, frames; higher values produce"
        "fewer cuts (default: 30)",
        default=30,
    )
    parser.add_argument(
        "--frequency",
        "-f",
        type=int,
        help="set to a positive integer to select images based on frequency "
        "rather than detecting cuts; integer gives frequency of the sampling "
        "(default: 0)",
        default=0,
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="flag to indicate that the pipeline should run quietly",
    )
    parser.add_argument(
        "--path-to-faces",
        type=str,
        default=None,
        help="Path to directory containing protype faces (optional). See "
        "tutorial on the commandline interface for more details.",
    )

    return parser


def _expand_path_and_check(fnames):
    """Expand paths from the input and ensure that all files exist.
    """
    video_files = []
    for this_name in fnames:
        video_files.extend(glob(this_name, recursive=True))

    video_files = sorted((abspath(x) for x in video_files))
    base_names = [splitext(basename(x))[0] for x in video_files]

    if len(set(base_names)) != len(base_names):
        raise AssertionError(
            "Processing input files with duplicate basenames is not supported."
        )

    if not video_files:
        raise FileNotFoundError("No valid input files found.")

    return video_files
