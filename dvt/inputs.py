# -*- coding: utf-8 -*-
"""Input objects
"""

from glob import glob
from itertools import chain

from cv2 import (
  COLOR_BGR2RGB,
  cvtColor,
  imread,
  VideoCapture,
  CAP_PROP_FPS,
  CAP_PROP_FRAME_COUNT,
  CAP_PROP_FRAME_HEIGHT,
  CAP_PROP_FRAME_WIDTH,
  CAP_PROP_POS_MSEC
)
from numpy import zeros, uint8, stack, zeros_like

from .core import VisualInput, FrameBatch
from .utils import _expand_path


class FrameInput(VisualInput):
    """An input object for extracting batches of images from an input video.

    Once initialized, subsequent calls to the next_batch method should be
    called to cycle through batches of frames. The continue_read flag will be
    turn false when all of data from the source has been returned within a
    batch. Note that this does not include the look-ahead region. The final
    batch will include padding by zeros (black) in this region.

    Attributes:
        bsize (int): Number of frames in a batch.
        fcount (int): Frame counter for the first frame in the current batch.
        vname (str): Name of the video file.
        continue_read (bool): Indicates whether there more frames to read from
            the input.
        start (float): Time code at the start of the current batch.
        end (float): Time code at the end of the current batch.
        meta (dict): A dictionary containing additional metadata about the
            video file.
    """

    def __init__(self, **kwargs):
        """Construct a new input from a video file.

        Args:
            input_path (str): Path to the video file. Can be any file readable
                by the OpenCV function VideoCapture.
            bsize (int): Number of frames to include in a batch. Defaults to
                256.
        """
        self.input_path = _expand_path(kwargs["input_path"])[0]
        self.bsize = kwargs.get("bsize", 256)
        self.meta = None
        self.fcount = 0
        self.continue_read = True
        self.start = 0
        self.end = 0
        self._video_cap = None
        self._img = None
        self._continue = True

        super().__init__()

    def open_input(self):
        """Open connection to the video file.
        """
        # start settings to
        self.fcount = 0
        self.continue_read = True
        self.start = 0
        self.end = 0
        self._video_cap = VideoCapture(self.input_path)
        self.meta = self._metadata()

        self._img = zeros(
            (self.bsize * 2, self.meta["height"], self.meta["width"], 3),
            dtype=uint8,
        )
        self._fill_bandwidth()  # fill the buffer with the first batch
        self._continue = True   # is there any more input left in the video

    def next_batch(self):
        """Move forward one batch and return the current FrameBatch object.

        Returns:
            A FrameBatch object that contains the next set of frames.
        """

        assert self.continue_read, "No more input to read."

        # shift window over by one bandwidth
        self._img[:self.bsize, :, :, :] = self._img[self.bsize:, :, :, :]

        # fill up the bandwidth; with zeros at and of video input
        if self._continue:
            self._fill_bandwidth()
        else:
            self.continue_read = self._continue
            self._img[self.bsize:, :, :, :] = 0

        # update counters
        frame_start = self.fcount
        self.start = self.end
        self.end = self._video_cap.get(CAP_PROP_POS_MSEC)
        self.fcount = self.fcount + self.bsize

        # get frame names
        fnames = list(range(int(frame_start), int(frame_start + self.bsize)))

        # return batch of frames.
        return FrameBatch(
            img=self._img,
            start=self.start,
            end=self.end,
            continue_read=self.continue_read,
            fnames=fnames,
            bnum=(frame_start // self.bsize),
        )

    def get_metadata(self):
        return self.meta

    def _metadata(self):
        """Fill metadata attribute using metadata from the video source.
        """
        path, bname, filename, file_extension = _expand_path(self.input_path)
        return {
            "type": "video",
            "fps": self._video_cap.get(CAP_PROP_FPS),
            "frames": int(self._video_cap.get(CAP_PROP_FRAME_COUNT)),
            "height": int(self._video_cap.get(CAP_PROP_FRAME_HEIGHT)),
            "width": int(self._video_cap.get(CAP_PROP_FRAME_WIDTH)),
            "input_path": path,
            "input_bname": bname,
            "input_filename": filename,
            "input_file_extension": file_extension,
        }

    def _fill_bandwidth(self):
        """Read in the next set of frames from disk and store results.

        This should not be called directly, but only through the next_batch
        method. Otherwise the internal counters will become inconsistent.
        """
        for idx in range(self.bsize):
            self._continue, frame = self._video_cap.read()
            if self._continue:
                rgb_id = COLOR_BGR2RGB
                self._img[idx + self.bsize, :, :, :] = cvtColor(
                    frame, rgb_id
                )
            else:
                self._img[idx + self.bsize, :, :, :] = 0


class ImageInput(VisualInput):
    """An input object for create batches of images from input images.

    Once initialized, subsequent calls to the next_batch method should be
    called to cycle through batches of frames. The continue_read flag will be
    turn false when all of data from the sources has been returned. Note that
    the batch will always be of size 1 and include a look-ahead region of all
    black pixels. This is needed because not all images will be the same size.

    Attributes:
        bsize (int): Number of frames in a batch. Always 1.
        vname (str): Name of the video file.
        continue_read (bool): Indicates whether there more frames to read from
            the input.
        fcount (int): Pointer to the next image to return.
        meta (dict): A dictionary containing additional metadata about the
            input images.
    """

    def __init__(self, **kwargs):
        """Construct a new input from a set of paths.

        Args:
            input_paths (str or list): Paths the images. Will use glob
                expansion on the elements.
        """
        self.continue_read = True
        self.fcount = 0

        # find input paths
        input_paths = kwargs.get("input_paths")
        if not isinstance(input_paths, list):
            input_paths = [input_paths]

        input_paths = [glob(x, recursive=True) for x in input_paths]
        self.paths = list(chain.from_iterable(input_paths))

        # fill in attribute defaults
        self.meta = None
        self.fcount = 0
        self.continue_read = True
        self.start = 0
        self.end = 0
        self._video_cap = None

        # create metadata
        self.meta = {"type": "image", "paths": self.paths}

        super().__init__()

    def open_input(self):
        self.fcount = 0
        self.continue_read = True

    def next_batch(self):
        """Move forward one batch and return the current FrameBatch object.

        Returns:
            A FrameBatch object that contains the next set of frames.
        """

        assert self.continue_read, "No more input to read."

        this_index = self.fcount

        # read the next image and create buffer
        img = imread(self.paths[this_index])
        img = cvtColor(img, COLOR_BGR2RGB)
        img = stack([img, zeros_like(img)])

        # is this the last image?
        self.fcount += 1
        if self.fcount >= len(self.paths):
            self.continue_read = False

        # return batch of frames.
        return FrameBatch(
            img=img,
            start=float(this_index),
            end=float(this_index),
            continue_read=self.continue_read,
            fnames=[self.paths[this_index]],
            bnum=this_index
        )

    def get_metadata(self):
        return self.meta
