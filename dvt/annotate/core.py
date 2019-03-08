# -*- coding: utf-8 -*-

import collections
import logging
import numpy as np
import cv2
import os

from ..utils import _format_time, stack_dict_frames


class FrameProcessor:

    def __init__(self, pipeline={}):
        self.output = collections.OrderedDict()
        self.pipeline = collections.OrderedDict()
        for anno in pipeline.values():
            self.load_annotator(anno)

    def load_annotator(self, annotator):
        assert issubclass(type(annotator), FrameAnnotator)
        self.pipeline.update({annotator.name: annotator})
        self.output.update({annotator.name: []})

    def process(self, input_obj, max_batch=None):
        assert input_obj.fcount == 0 # make sure there is a fresh input

        # clear and start each annotator
        for anno in self.pipeline.values():
            anno.clear()
            anno.start(input_obj)

        # cycle through batches and process the file
        while input_obj.continue_read:
            batch = input_obj.next_batch()
            for anno in self.pipeline.values():
                next_values =  anno.annotate(batch)
                if next_values is not None:
                    self.output[anno.name] += next_values
                msg = "processed batch {0:s} to {1:s} with annotator: '{2:s}'"
                logging.info(msg.format(_format_time(batch.start),
                                        _format_time(batch.end), anno.name))
            if max_batch is not None:
                if batch.frame >= (max_batch - 1) * batch.bsize:
                    return

    def clear(self):
        for annotator in self.pipeline.values():
            annotator.clear()

        self.pipeline = collections.OrderedDict()

    def collect(self, aname):
        return stack_dict_frames(self.output[aname])

    def collect_all(self):
        ocollect = collections.OrderedDict.fromkeys(self.pipeline.keys())

        for k in ocollect.keys():
            ocollect[k] = self.collect(k)

        return ocollect


class FrameAnnotator:
    name = 'base'

    def __init__(self):
        self.cache = {}

    def clear(self):
        self.cache = {}

    def start(self, input):
        pass

    def annotate(self, batch):
        return [batch.start]

    def collect(self, output):
        return output


class FrameInput:

    def __init__(self, input_path, bsize=256):
        self.video_cap = cv2.VideoCapture(input_path)
        self.bsize = bsize
        self.fcount = 0
        self.input_path = input_path
        self.vname = os.path.basename(input_path)
        self.continue_read = True
        self.start = 0
        self.end = 0
        self.meta = self.metadata()
        self._img = np.zeros((bsize * 2, self.meta['height'],
                              self.meta['width'], 3), dtype=np.uint8)
        self._fill_bandwidth() # fill the buffer with the first batch

    def metadata(self):
        return {'type': 'video',
                'fps': self.video_cap.get(cv2.CAP_PROP_FPS),
                'frames': int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'height': int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'width': int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}

    def next_batch(self):
        # shift window over by one bandwidth
        self._img[:self.bsize, :, :, :] = self._img[self.bsize:, :, :, :]

        # fill up the next bandwidth
        self._fill_bandwidth()

        frame_start = self.fcount
        self.start = self.end
        self.end = self.video_cap.get(cv2.CAP_PROP_POS_MSEC)
        self.fcount = self.fcount + self.bsize

        return FrameBatch(vname=self.vname, start=self.start, end=self.end,
                          continue_read=self.continue_read, img = self._img,
                          frame=frame_start)

    def _fill_bandwidth(self):
        for idx in range(self.bsize):
            self.continue_read, frame = self.video_cap.read()
            if self.continue_read:
                rgb_id = cv2.COLOR_BGR2RGB
                self._img[idx + self.bsize, :, :, :] = cv2.cvtColor(frame,
                                                                    rgb_id)
            else:
                self._img[idx + self.bsize, :, :, :] = 0


class FrameBatch:

    def __init__(self, vname, start, end, continue_read, img, frame):
        self.vname = vname
        self.img = img
        self.bsize = img.shape[0] // 2
        self.frame = frame
        self.start = start
        self.end = end
        self.continue_read = continue_read

    def get_frames(self):
        return self.img

    def get_batch(self):
        return self.img[:self.bsize, :, :, :]

    def get_frame_nums(self):
        start = int(self.frame)
        end = int(self.frame + self.bsize)
        return list(range(start, end))

