# -*- coding: utf-8 -*-
"""Command line tool for evoking the toolkit.

This tool allows users to quickly apply pre-build pipelines to one or more
media files.
"""

import argparse
import glob
import logging
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
        logging.basicConfig(level="INFO")

    for vin in _expand_path_and_check(args.inputs):

        wp = VideoPipeline(
            vin,
            doutput=args.dirout,
            diff_co=args.diff_cutoff,
            cut_min_length=args.cut_min_length,
        )
        wp.make_breaks(freq=args.frequency)
        wp.run(level=args.pipeline_level)


def _get_arg_parse():
    parser = argparse.ArgumentParser(
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

    return parser


def _expand_path_and_check(fnames):
    """Expand paths from the input and ensure that all files exist
    """
    video_files = []
    for this_name in fnames:
        video_files.extend(glob.glob(this_name, recursive=True))

    video_files = sorted((abspath(x) for x in video_files))
    base_names = [splitext(basename(x))[0] for x in video_files]

    if len(set(base_names)) != len(base_names):
        raise AssertionError(
            "Processing input files with duplicate basenames is not supported."
        )

    if not video_files:
        raise FileNotFoundError("No valid input files found.")

    return video_files
