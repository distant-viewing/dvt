# -*- coding: utf-8 -*-
"""Extract frame images from input.

This module supplies an annotator that saves individual frames to some
location specified on the local machine. It is only useful for its side effects
as no information is returned to the FrameProcessor.
"""

from os.path import join

from cv2 import cvtColor, imwrite, resize, COLOR_RGB2BGR

from ..abstract import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames, _check_out_dir


class PngAnnotator(FrameAnnotator):
    """Annotator for saving PNG still images from an input.

    The annotate method of this annotator does not return any data. It is
    useful only for its side effects.

    Attributes:
        output_dir (str): location where output frames should be saved. Will be
            created if the location does not yet exist.
        freq (int): How often to save the image. For example, setting
            the frequency to 2 will save every other frame in the batch.
        size (int): What should the size of the output images be? Set to
            None, the default, to preserve the size as given in the input file.
            Given as the desired height; the width will be scaled to keep the
            aspect ratio.
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "png"

    def __init__(self, **kwargs):
        self.output_dir = _check_out_dir(kwargs['output_dir'])
        self.freq = kwargs.get('freq', 1)
        self.size = kwargs.get('size', None)
        self.frames = _proc_frame_list(kwargs.get('frames', None))

        super().__init__(**kwargs)

    def annotate(self, batch):
        """Annotate the batch of frames with the PNG annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            Returns an empty list.
        """
        for fnum in _which_frames(batch, self.freq, self.frames):
            img = cvtColor(batch.img[fnum, :, :, :], COLOR_RGB2BGR)
            frame = batch.get_frame_names()[fnum]

            opath = join(self.output_dir, "frame-{0:06d}.png".format(frame))
            if self.size is not None:
                scale = batch.img.shape[1] / self.size
                new_size = (int(batch.img.shape[2] // scale), int(self.size))
                img_resize = resize(img, new_size)
                imwrite(filename=opath, img=img_resize)
            else:
                imwrite(filename=opath, img=img)
