# -*- coding: utf-8 -*-
"""Command line tool for evoking the toolkit.

This tool allows users to quickly apply pre-build pipelines to one or more
media files.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the default usage of the command line interface:

    > python3 -m dvt video-viz video-clip.mp4

    This may take several minutes to complete. Some minimal logging information
    should display the annotators progress in your terminal. Once finished, you
    should have a new directory dvt-output-data that contains extracted
    metadata and frames from the source material. You can view the extracted
    information by starting a local http server:

    > python3 -m http.server --directory dvt-output-data

    And opening the following: http://0.0.0.0:8000/
"""

from sys import argv

from .pipeline.viz import VideoVizPipeline
from .pipeline.csv import VideoCsvPipeline


def run_cli(args=None):
    """Command line tool for evoking the toolkit.

    This function is not intended to be called directly from an external
    program. Run with caution.
    """

    if args is None:
        args = argv

    try:
        key = args[1]
        if key == "video-viz":
            pipeline = VideoVizPipeline.create_from_cli(args[2:])
        elif key == "video-csv":
            pipeline = VideoCsvPipeline.create_from_cli(args[2:])
        else:
            raise IndexError("Unknown pipeline: " + key)

        pipeline.run()

    except IndexError:
        print_help()


def print_help():
    """Print useful help message for the toolkit commandline interface.
    """

    msg = """usage: python3 -m dvt pipeline"

The Distant Viewing Toolkit command line interface makes it possible
to quickly apply pre-built pipelines to input media files.

Currently available pipelines:

  video-viz      given a single video file, selects a set of frames
                 (by default, one frame in each shot) and extracts
                 various metadata using the toolkit. The output is
                 stored locally as JSON and PNG files that can be
                 viewed locally using the interactive web interface.

  video-csv      a pipeline used when extracting metadata for the
                 analysis of visual style across a larger corpus.
                 Output is stored as a collection of CSV files.
                 When set to processes all of the frames in the
                 input, this pipeline is very computationally
                 intensive.

Running each pipeline with no other options will display their
respective help pages. For more examples, see the command line
tutorial included in the documentation:

    https://distant-viewing.github.io/dvt/tutorial/cli.html

"""

    print(msg)
