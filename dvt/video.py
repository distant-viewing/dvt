# -*- coding: utf-8 -*-
"""Class for working on the frames within a video file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import numpy as np
import cv2

from .utils import AnnotatorStatus, iso8601
from .frame import FrameAnnotator


class VideoProcessor:
    """Run frame annotators over a video file
    """

    pipeline = collections.OrderedDict()
    output_path = ""
    video_path = ""
    video_name = ""

    def __init__(self):
        self.pipeline = collections.OrderedDict()

    def load_annotator(self, annotator):
        assert(issubclass(type(annotator), FrameAnnotator))
        self.pipeline.update({annotator.name: annotator})

    def clear_annotators(self):
        self.pipeline = collections.OrderedDict()

    def setup_input(self, video_path, output_path=None, video_name=None):
        self.video_path = os.path.abspath(os.path.expanduser(video_path))

        if video_name is None:
            self.video_name = os.path.splitext(os.path.basename(video_path))[0]

        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.video_path),
                                       self.video_name + "-output.json")
        else:
            output_path = os.path.expanduser(output_path)

        self.output_path = os.path.abspath(output_path)

    def process(self, verbose=True, max_frame=None):
        # load the video file using OpenCV
        cap = cv2.VideoCapture(self.video_path)

        # get video metadata
        video_meta = {'video': self.video_name,
                      'type': 'video',
                      'fps': cap.get(cv2.CAP_PROP_FPS),
                      'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                      'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                      'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}

        # cycle through the frames one by one until asked
        # to stop by an annotator (or reaching the end)
        with open(self.output_path, 'w') as fout:

            print(json.dumps(video_meta), file=fout)
            continue_video = True
            fcount = 0

            while continue_video:

                continue_video, frame = cap.read()
                if not continue_video:
                    break
                if max_frame is not None:
                    if fcount >= max_frame:
                        break
                foutput = {'video': self.video_name, 'type': 'frame',
                           'frame': fcount,
                           'time': round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                         3)}

                for anno_name, anno in self.pipeline.items():

                    res = anno.process_next(frame, foutput)
                    if len(res) > 0:
                        foutput[anno.name] = res
                    status = anno.status()

                    if verbose:
                        msg = "[{0:s}] processed frame {1:06d}:" + \
                              "annotator: {2:s}"
                        print(msg.format(iso8601(), fcount, anno.name))

                    if status == AnnotatorStatus.NEXT_FRAME:
                        break

                print(json.dumps(foutput), file=fout)
                fcount = fcount + 1
