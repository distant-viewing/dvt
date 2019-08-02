# -*- coding: utf-8 -*-
"""Command line tool for evoking the toolkit.

This tool allows users to quickly apply pre-build pipelines to one or more
media files.
"""

import argparse
import glob
import logging
from os.path import abspath, isfile, join, splitext, basename

from .pipeline.video import VideoPipeline


def run_cli():
    """Command line tool for evoking the toolkit.

    This functions is not intended to be called directly from an external
    program. Run with caution.
    """
    parser = _get_arg_parse()
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level='INFO')

    video_files, base_names = _expand_path_and_check(args.inputs)

    for vin, bin in zip(video_files, base_names):

        wp = WebPipeline(vin, doutput=join(args.dirout, '', bin))
        wp.run()


def _get_arg_parse():
    parser = argparse.ArgumentParser(
        prog="python3 -m dvt",
        description="The Distant Viewing Toolkit command line interface makes "
        "it possible to quickly apply pre-build pipelines to one or more "
        "media files. Several options exist to control the way the output is "
        "produced. See the lower-level functions for more flexibility.",
    )
    parser.add_argument(
        "inputs",
        nargs='+',
        help="one or more media files to process",
    )
    parser.add_argument(
        "--dirout",
        "-d",
        type=str,
        help="base directory in which to store the output",
        default="dvt-output-data"
    )
    parser.add_argument(
        "--pipeline-level",
        "-l",
        type=int,
        help="interger code (0, 1, or 2) describing how much data to parse; "
        "0 creates just metadata, 1 creates just images and metadata, 2 or "
        "more creates the interactive website (default: 2)",
        default=2
    )
    parser.add_argument(
        "--diff-cutoff",
        "-dc",
        type=int,
        help="difference cutoff value; integer from 0-256; higher values "
        "produce fewer cuts (default: 10)",
        default=10
    )
    parser.add_argument(
        "--hue-cutoff",
        "-hc",
        type=int,
        help="hue histogram cutoff value; positive integer; higher values"
        "produce fewer cuts (default: 10)",
        default=10
    )
    parser.add_argument(
        "--frequency",
        "-f",
        type=int,
        help="set to a positive integer to select images based on frequency "
        "rather than detecting cuts; integer gives frequency of the sampling "
        "(default: 0)",
        default=0
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action='store_true',
        help="flag to indicate that the pipeline should run quietly"
    )

    return parser


def _expand_path_and_check(fnames):
    """Expand paths from the input and ensure that all files exist
    """
    video_files = []
    for this_name in fnames:
        video_files.extend(glob.glob(this_name, recursive=True))

    video_files = sorted(set([abspath(x) for x in video_files]))
    for this_file in video_files:
        if not isfile(this_file):
            raise FileNotFoundError("Input file '{0:s}' not found!".format(
                this_file
            ))

    base_names = [splitext(basename(x))[0] for x in video_files]
    if len(set(base_names)) != len(base_names):
        raise AssertionError(
            "Processing input files with duplicate basenames is supported."
        )

    return video_files, base_names
