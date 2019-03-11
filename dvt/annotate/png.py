# -*- coding: utf-8 -*-
"""This module illustrates something.
"""

import os

import cv2

from .core import FrameAnnotator


class PngAnnotator(FrameAnnotator):
    """Here"""

    name = 'png'

    def __init__(self, output_dir, freq=1):
        self.freq = freq
        self.output_dir = os.path.expanduser(output_dir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        super().__init__()

    def annotate(self, batch):
        """Here

        :param batch:

        """
        for fnum in range(0, batch.bsize, self.freq):
            img = cv2.cvtColor(batch.img[fnum, :, :, :], cv2.COLOR_RGB2BGR)
            frame = batch.get_frame_nums()[fnum]
            opath = "frame-{0:06d}.png".format(frame)
            opath = os.path.join(self.output_dir, opath)
            cv2.imwrite(filename=opath, img=img)

        return []
