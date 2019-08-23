# -*- coding: utf-8 -*-
"""Annotators for extracting high-level metadata about the images in the input.

The annotator implemented here finds simple metadata, such as high and width,
of the input images. Particularly useful when using the ImageInput annotator.
"""

from numpy import array

from ..abstract import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames


class ImgAnnotator(FrameAnnotator):
    """Annotator for grabbing metadata about the images in the batch.
    """

    name = "img"

    def __init__(self, **kwargs):
        self.freq = _proc_frame_list(kwargs.get('freq', 1))
        self.frames = _proc_frame_list(kwargs.get('frames', None))

        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the difference annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A dictionaries containing the height and width of the input images.
        """
        # determine which frames to work on
        frames = _which_frames(batch, self.freq, self.frames)
        if not frames:
            return None

        output = {'img_height': [], 'img_width': []}
        for fnum in frames:
            img = batch.get_frames()[fnum]
            output['img_height'].append(img.shape[0])
            output['img_width'].append(img.shape[1])

        # Add frame metadata
        output["frame"] = array(batch.get_frame_names())[list(frames)]

        return output
