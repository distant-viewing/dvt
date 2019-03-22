# -*- coding: utf-8 -*-
"""Extract frame images from input.

This module supplies an annotator that saves individual frames to some
location specified on the local machine. It is only useful for its side effects
as no information is returned to the FrameProcessor.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of the PngAnnotator over two batches of the input.

    >>> fp = FrameProcessor()
    >>> fp.load_annotator(PngAnnotator(output_dir="my-frames"))
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    INFO:root:processed batch 00:00:00,000 to 00:00:17,083 with annotator: 'png'
    INFO:root:processed batch 00:00:17,083 to 00:00:25,625 with annotator: 'png'

    There will now be 512 (256 * 2) frames stored in the directory "my-frames".
    They images named according following the format "frame-000255.png".
"""

import os

import cv2

from .core import FrameAnnotator


class PngAnnotator(FrameAnnotator):
    """Annotator for saving PNG still images from an input.

    The annotate method of this annotator does not return any data. It is
    useful only for its side effects.

    Attributes:
        output_dir (str): location where output frames should be saved. Will be
            created if the location does not yet exist.
        freq (int): How often to save the image. For example, setting
            the frequency to 2 will save every other frame in the batch.
    """

    name = "png"

    def __init__(self, output_dir, freq=1):
        self.freq = freq
        self.output_dir = os.path.expanduser(output_dir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the PNG annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            Returns an empty list.
        """
        for fnum in range(0, batch.bsize, self.freq):
            img = cv2.cvtColor(batch.img[fnum, :, :, :], cv2.COLOR_RGB2BGR)
            frame = batch.get_frame_names()[fnum]

            if isinstance(frame, int):
                opath = "frame-{0:06d}.png".format(frame)
            else:
                opath = os.path.basename(frame)
                opath = os.path.splitext(opath)[0] + ".png"

            opath = os.path.join(self.output_dir, opath)
            cv2.imwrite(filename=opath, img=img)

        return []
