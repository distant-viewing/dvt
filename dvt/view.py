# -*- coding: utf-8 -*-
"""Class for viewing annotations over raw video file
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

COL_BL = [255, 165, 0]
COL_OR = [0, 165, 255]
COL_DR = [20, 20, 20]


class EndDataException(Exception):
    pass


class VideoViewer:
    """View annotations over raw video file
    """

    input_path = ""
    video_path = ""
    dframe = 0
    vframe = 0
    fv = None

    def __init__(self):
        self.pipeline = collections.OrderedDict()
        self.fv = FrameViewer()

    def setup_input(self, video_path, input_path):
        self.video_path = os.path.abspath(os.path.expanduser(video_path))
        self.input_path = os.path.abspath(os.path.expanduser(input_path))

    def next_frame_data(self, fin):
        while True:
            line = fin.readline()
            if line == '':
                raise EndDataException

            output = json.loads(line)
            if output['type'] == 'frame':
                self.dframe = output['frame']
                break

        return output

    def next_frame(self, cap):
        frame = None
        while self.vframe < self.dframe:
            ret, frame = cap.read()
            self.vframe = self.vframe + 1
            if not ret:
                raise EndDataException

        return frame

    def run(self, output=None, start=0):
        if output is not None:
            video_out = True
            vout_path = os.path.expanduser(output)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vout = cv2.VideoWriter(vout_path, fourcc, 20.0, (640, 480))
        else:
            video_out = False

        # load the video file using OpenCV
        cap = cv2.VideoCapture(self.video_path)
        self.dframe = -1
        self.vframe = -1

        # cycle through the frames data
        continue_video = True
        with open(self.input_path, 'r') as fin:
            for i in range(start):
                video_meta = json.loads(fin.readline())

            video_meta = json.loads(fin.readline())

            while continue_video:

                # attempt to get the next frame's data and image; if either
                # stream ends, finish and return
                try:
                    fdata = self.next_frame_data(fin)
                    frame = self.next_frame(cap)
                except EndDataException:
                    break

                if self.dframe != self.vframe:
                    raise Exception("Frame data output of order")

                self.fv.set_frame(frame)
                self.fv.set_name(video_meta['video'])

                for key, value in fdata.items():

                    if key == "diff":
                        self.fv.set_diff(value['decile'][4])

                    if key == "oflow":
                        self.fv.set_oflow(value)

                    if key == "object":
                        self.fv.set_objects(value, self.dframe)

                    if key == "places":
                        self.fv.set_places(value)

                    if key == "face":
                        self.fv.set_faces(value, self.dframe)

                self.fv.render_objects(self.dframe)
                self.fv.render_faces(self.dframe)
                self.fv.render_places()

                vis = self.fv.render_output()
                self.fv.reset_panels()
                if video_out:
                    vout.write(vis)
                else:
                    cv2.imshow('frame', vis)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

        cap.release()
        cv2.destroyAllWindows()
        if video_out:
            vout.release()


class FrameViewer:
    """Internal class for representing the output of one frame
    """

    def __init__(self):
        self.reset_panels()
        self.dv = np.zeros((30), dtype=np.int32)
        self.ov = dict()
        self.objs = []
        self.faces = []
        self.places = []

    def reset_panels(self):
        self.panel = np.zeros((480, 480, 3), dtype=np.uint8)
        self.tbars = np.zeros((200, 720, 3), dtype=np.uint8)
        self.pname = np.zeros((200, 480, 3), dtype=np.uint8)

    def render_output(self):
        fr = cv2.resize(self.frame, (720, 480),
                        interpolation=cv2.INTER_CUBIC)
        v = np.concatenate((np.concatenate((fr, self.panel), axis=1),
                            np.concatenate((self.tbars, self.pname), axis=1)),
                           axis=0)
        return v

    def set_frame(self, img):
        self.frame = img

    def set_name(self, name):
        cv2.putText(self.pname, name, (300, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, COL_BL, 1)

    def set_diff(self, next_val):
        self.dv = np.concatenate(([next_val], self.dv[:-1]))
        for i, (v, u) in enumerate(zip(self.dv[:-1], self.dv[1:])):
            cv2.line(self.tbars, (360 - (i-1)*12, 190 - v),
                     (360 - i*12, 190 - u), COL_BL, 2)

    def set_oflow(self, new_flow):
        self.ov = {k: v for k, v in self.ov.items() if k in new_flow}

        for k, v in new_flow.items():
            if k not in self.ov:
                self.ov[k] = [new_flow[k]]
            else:
                self.ov[k] = [new_flow[k]] + self.ov[k][:10]

        for key, value in self.ov.items():
            for u, v in zip(value[1:], value[:-1]):
                cv2.line(self.frame, (u[0], u[1]), (v[0], v[1]),
                         COL_OR, 2)

    def set_places(self, places):
        self.places = places['categories'][:5]

    def render_places(self):
        for i, pl in enumerate(self.places):
            cv2.putText(self.pname, "{0:s} ({1:01.03f})".format(pl[1], pl[0]),
                        (50, (i + 1) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_BL, 1)

    def set_objects(self, new_objs, frame):
        # if frame detector was run, we want to clear any old objects
        self.objs = [(frame, x) for x in new_objs]

    def render_objects(self, frame):
        self.objs = [(x, y) for x, y in self.objs if x > frame - 60]

        for frame, obj in self.objs:
            cv2.rectangle(self.frame, (obj['box']['left'], obj['box']['top']),
                          (obj['box']['right'], obj['box']['bottom']),
                          COL_OR, 1)
            cv2.rectangle(self.frame, (obj['box']['left'], obj['box']['top']),
                          (obj['box']['right'], obj['box']['top'] - 25),
                          COL_OR, -1)
            cv2.putText(self.frame, obj['class'],
                        (int(obj['box']['left'] + 10),
                        obj['box']['top'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, COL_BL, 1)

    def set_faces(self, new_objs, frame):
        # if frame detector was run, we want to clear any old objects
        self.faces = [(frame, x) for x in new_objs]

    def render_faces(self, frame):
        self.faces = [(x, y) for x, y in self.faces if x > frame - 60]

        for frame, obj in self.faces:
            cv2.rectangle(self.frame, (obj['box']['left'], obj['box']['top']),
                          (obj['box']['right'], obj['box']['bottom']),
                          COL_BL, 2)
