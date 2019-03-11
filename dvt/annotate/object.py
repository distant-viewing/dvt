# -*- coding: utf-8 -*-
"""This module illustrates something.
"""

import numpy as np

from .core import FrameAnnotator
from ..utils import stack_dict_frames


class ObjectAnnotator(FrameAnnotator):
    """Here"""

    name = 'object'

    def __init__(self, detector, freq=1):
        self.freq = freq
        self.detector = detector
        super().__init__()

    def annotate(self, batch):
        """Here

        :param batch:

        """

        f_obj = []
        for fnum in range(0, batch.bsize, self.freq):
            img = batch.img[fnum, :, :, :]
            t_obj = stack_dict_frames(self.detector.detect(img))
            if t_obj:
                frame = batch.get_frame_nums()[fnum]
                t_obj['video'] = [batch.vname] * len(t_obj['top'])
                t_obj['frame'] = [frame] * len(t_obj['top'])
                f_obj.append(t_obj)

        return f_obj


class ObjectDetectRetinaNet():
    """Here"""

    def __init__(self, model_path, cutoff=0.5):
        from keras_retinanet import models
        from keras_retinanet.utils.image import preprocess_image, resize_image

        self.preprocess_image = preprocess_image
        self.resize_image = resize_image
        self.cutoff = cutoff
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.lcodes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                       4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
                       8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                       11: 'stop sign', 12: 'parking meter', 13: 'bench',
                       14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
                       18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                       22: 'zebra', 23: 'giraffe', 24: 'backpack',
                       25: 'umbrella', 26: 'handbag', 27: 'tie',
                       28: 'suitcase', 29: 'frisbee', 30: 'skis',
                       31: 'snowboard', 32: 'sports ball', 33: 'kite',
                       34: 'baseball bat', 35: 'baseball glove',
                       36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                       39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
                       43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                       47: 'apple', 48: 'sandwich', 49: 'orange',
                       50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                       53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                       57: 'couch', 58: 'potted plant', 59: 'bed',
                       60: 'dining table', 61: 'toilet', 62: 'tv',
                       63: 'laptop', 64: 'mouse', 65: 'remote',
                       66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                       69: 'oven', 70: 'toaster', 71: 'sink',
                       72: 'refrigerator', 73: 'book', 74: 'clock',
                       75: 'vase', 76: 'scissors', 77: 'teddy bear',
                       78: 'hair drier', 79: 'toothbrush', -1: 'unknown'}

    def detect(self, img):
        """Here

        :param img:

        """
        # process the input image
        img = self.preprocess_image(img)
        img, scale = self.resize_image(img)
        img = np.expand_dims(img, axis=0)

        # make predictions and scale back to original
        boxes, scores, labels = self.model.predict_on_batch(img)
        boxes /= scale

        # arrange output as list of dictionaries for processing
        objs = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score > self.cutoff:
                objs += [{'top': int(box[1]), 'right': int(box[2]),
                          'bottom': int(box[3]), 'left': int(box[0]),
                          'score': score, 'class': self.lcodes[label]}]

        return objs
