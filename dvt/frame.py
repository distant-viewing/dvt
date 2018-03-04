# -*- coding: utf-8 -*-
"""Frame level annotator classes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from keras import backend as K
from keras.models import load_model

import cv2
import face_recognition as fr

from .darknet import yolo_eval, yolo_head
from .utils import AnnotatorStatus, get_file


class FrameAnnotator:
    name = 'base'

    def __init__(self):
        pass

    def process_next(self, img, foutput):
        return []

    def clear(self):
        pass

    def status(self):
        return AnnotatorStatus.NEXT_ANNOTATOR


class DiffFrameAnnotator(FrameAnnotator):
    name = 'diff'
    last_frame = None
    last_avg_value = 0

    def process_next(self, img, foutput):
        self.frame_number = foutput['frame']

        frame_small = cv2.resize(img, (32, 32))
        frame_small_hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

        self.last_avg_value = int(np.mean(frame_small_hsv[:, :, 2]))
        if self.last_frame is not None:
            diffs = np.abs(frame_small_hsv - self.last_frame)
            diff_q = np.percentile(diffs, q=list(range(0, 110, 10)))
            output = {'decile': [int(x) for x in diff_q]}
        else:
            output = {}

        self.last_frame = frame_small_hsv

        return output

    def clear(self):
        self.last_frame = None
        self.last_avg_value = 0


class TerminateFrameAnnotator(FrameAnnotator):
    name = 'terminate'
    last_frame_allowed = 0
    sflag = AnnotatorStatus.NEXT_ANNOTATOR

    def __init__(self):
        self.last_frame_allowed = 0

    def process_next(self, img, foutput):
        # only proceed if the frame has changed a lot or
        # if there has not been a change in a while (but
        # again, only if the scene is not too dark)
        fnum = foutput['frame']
        try:
            if foutput['diff']['decile'][4] > 15 and self.last_frame_allowed \
                                                     + 24 < fnum:
                self.sflag = AnnotatorStatus.NEXT_ANNOTATOR
                self.last_frame_allowed = fnum
            elif self.last_frame_allowed + 24 * 10 < fnum:
                self.sflag = AnnotatorStatus.NEXT_ANNOTATOR
                self.last_frame_allowed = fnum
            else:
                self.sflag = AnnotatorStatus.NEXT_FRAME
        except KeyError:
            self.sflag = AnnotatorStatus.NEXT_ANNOTATOR

        return {}

    def status(self):
        return self.sflag


class FlowFrameAnnotator(FrameAnnotator):
    name = 'oflow'
    fp = None
    lk_params = None
    old_gray = None
    min_track = 5

    p0 = None
    tnum = 0
    these_nums = None

    def __init__(self):
        self.old_gray = None
        self.tnum = 0
        self.fp = feature_params = dict(maxCorners=100,
                                        qualityLevel=0.1,
                                        minDistance=40,
                                        blockSize=7)

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=0,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def process_next(self, img, foutput):
        output = {}
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # is this image the first frame?
        if self.old_gray is None:
            self.old_gray = frame_gray
            return output

        # do we need new features to track?
        if (self.p0 is None) or (len(self.p0) < self.min_track):
            self.p0 = self.good_features()
            if self.p0 is not None:
                self.these_nums = np.arange(len(self.p0)) + self.tnum
                self.tnum = self.tnum + len(self.p0)

        # do we have any features to track? (even after running
        # cv2.goodFeaturesToTrack, we still may not have any, such
        # as when the screen is blacked out)
        if self.p0 is not None:
            p1, st0, err = self.optical_flow(frame_gray)
            p0r, st1, err = self.optical_flow(frame_gray, reverse=True)

            # which paths are "good"?
            d = abs(self.p0-p0r).reshape(-1, 2).max(-1)
            ok = (st0.flatten() == 1) & (st0.flatten() == 1) & (d < 20)
            self.p0 = self.p0[ok]
            p1 = p1[ok]
            self.these_nums = self.these_nums[ok]

            output = {'t' + str(z): [int(x[0][0]), int(x[0][1])] for x, z in
                      zip(p1, self.these_nums)}

            self.p0 = p1

        self.old_gray = frame_gray.copy()
        return output

    def good_features(self):
        return cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.fp)

    def optical_flow(self, new_frame, reverse=False):
        if not reverse:
            return cv2.calcOpticalFlowPyrLK(self.old_gray, new_frame,
                                            self.p0, None, **self.lk_params)
        else:
            return cv2.calcOpticalFlowPyrLK(self.old_gray, new_frame,
                                            self.p0, None, **self.lk_params)


class ObjectCocoFrameAnnotator(FrameAnnotator):
    name = 'object'
    sess = None
    yolo_model = None
    model_image_size = None
    input_image_shape = None
    boxes = None
    scores = None
    classes = None

    def __init__(self):
        model_path = get_file(fname="yolo.h5",
                              origin="https://github.com/statsmaths/dvt/" +
                                     "releases/download/0.1.0/yolo.h5")
        score_threshold = 0.3
        iou_threshold = .5

        model_path = os.path.expanduser(model_path)

        self.sess = K.get_session()

        self.class_names = ['person', 'bicycle', 'car', 'motorbike',
                            'aeroplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light', 'fire hydrant', 'stop sign',
                            'parking meter', 'bench', 'bird', 'cat', 'dog',
                            'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella',
                            'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                            'snowboard', 'sports ball', 'kite', 'baseball bat',
                            'baseball glove', 'skateboard', 'surfboard',
                            'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana',
                            'apple', 'sandwich', 'orange', 'broccoli',
                            'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                            'chair', 'sofa', 'pottedplant', 'bed',
                            'diningtable', 'toilet', 'tvmonitor', 'laptop',
                            'mouse', 'remote', 'keyboard', 'cell phone',
                            'microwave', 'oven', 'toaster', 'sink',
                            'refrigerator', 'book', 'clock', 'vase',
                            'scissors', 'teddy bear', 'hair drier',
                            'toothbrush']
        anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                   5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        anchors = np.array(anchors).reshape(-1, 2)

        self.yolo_model = load_model(model_path)
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]

        yolo_outputs = yolo_head(self.yolo_model.output, anchors,
                                 len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold)

    def process_next(self, img, foutput):

        ir = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(ir, tuple(reversed(self.model_image_size)))
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        feed_dict = {self.yolo_model.input: image_data,
                     self.input_image_shape: [ir.shape[0], ir.shape[1]],
                     K.learning_phase(): 0}
        obox, oscore, oclass = self.sess.run([self.boxes, self.scores,
                                              self.classes],
                                             feed_dict=feed_dict)

        oname = [self.class_names[x] for x in oclass]
        output = []

        for box, score, cname in zip(obox, oscore, oname):
            box = box.tolist()
            output.append({
                    'box': {'top': int(box[0]),
                            'bottom': int(box[2]),
                            'left': int(box[1]),
                            'right': int(box[3])},
                    'score': round(score.tolist(), 3),
                    'class': cname})

        return output

    def clear(self):
        pass


class FaceFrameAnnotator(FrameAnnotator):
    name = 'face'

    def process_next(self, img, foutput):

        faces = fr.face_locations(img, 1, model="cnn")
        embed = fr.face_encodings(img, faces, num_jitters=10)

        output = []
        for face, em in zip(faces, embed):
            output.append({
                    'box': {'top': face[0], 'bottom': face[2],
                            'left': face[3], 'right': face[1]},
                    'embed': [round(x, 4) for x in em.tolist()]})

        return(output)
