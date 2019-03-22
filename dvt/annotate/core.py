# -*- coding: utf-8 -*-
"""Core objects for processing video files in the Distant Viewing Toolkit.

This module provides the four main classes for processing a raw video file.
These are (i) FrameProcessor, (ii) FrameAnnotator, and (iii) FrameInput,
and (iv) FrameBatch. The basic pipeline is shown in the following example.

Extending the toolkit requires creating new subclasses of the FrameAnnotator
object, specifically by overriding the annotate method. This method in turn
requires working with a FrameBatch object. Most users should not need to modify
or follow the internal structure of the FrameProcessor and FrameInput objects.

Example:
    Start by constructing a new input source by pointing to an existing video
    file on disk.

    >>> fin = FrameInput("input.mp4")

    Then, construct a pipeline of annotators by starting with an empty
    FrameProcessor and adding any number of FrameAnnotator objects.

    >>> fp = FrameProcessor()
    >>> fp.load_annotator(FrameAnnotator())

    Next, process batches of the data from the input source. If logging is
    turned on, this will generate verbose information as each annotator is
    called on a particular batch.

    >>> fp.process(fin)

    Finally, collect the annotated data as an ordered dictionary of DictFrame
    objects.

    >>> output = fp.collect_all()

    Following the above sequence will not return any data because we only
    loaded the default FrameAnnotator (it returns no data). The annotators
    contained in the other sub-modules should be used to extract interesting
    results from the source.
"""

import collections
import glob
import itertools
import logging
import os

import cv2
import numpy as np

from ..utils import _format_time, stack_dict_frames


class FrameProcessor:
    """Run a pipeline of annotators over batches of frames.

    Attributes:
        pipeline (OrderedDict): FrameAnnotator objects to run over the inputs.
        output (OrderedDict): DictFrame objects containing the annotations.
    """

    def __init__(self, pipeline=None):
        """Construct a new pipeline of annotators.

        You can pass a collection of annotators to the constructor, or create
        an empty processor and fill it with the load_annotator method.

        Args:
            pipeline (optional): FrameAnnotator objects to run over the inputs.
        """

        if not pipeline:
            pipeline = {}

        self.pipeline = collections.OrderedDict(pipeline)
        self.output = collections.OrderedDict()
        for anno in pipeline.values():
            self.load_annotator(anno)

    def load_annotator(self, annotator):
        """Add an annotator to the end of the pipeline.

        Args:
            annotator (FrameAnnotator): annotator to add into the pipeline.
        """
        assert issubclass(type(annotator), FrameAnnotator)
        self.pipeline.update({annotator.name: annotator})
        self.output.update({annotator.name: []})

    def process(self, input_obj, max_batch=None):
        """Run annotators over an input object.

        Args:
            input_obj (FrameInput): The input source for batches of data.
            max_batch (int): The maximum number of batches to process from the
                input. The default value (None) sets no limit on the total
                number of batches.
        """
        assert input_obj.fcount == 0  # make sure there is a fresh input

        # clear and start each annotator
        for anno in self.pipeline.values():
            anno.clear()
            anno.start(input_obj)

        # cycle through batches and process the file
        while input_obj.continue_read:
            batch = input_obj.next_batch()
            for anno in self.pipeline.values():
                next_values = anno.annotate(batch)
                if next_values is not None:
                    self.output[anno.name] += next_values
                if isinstance(batch.fnames[0], int):
                    msg = "processed {0:s} to {1:s} with annotator: '{2:s}'"
                    logging.info(
                        msg.format(
                            _format_time(batch.start),
                            _format_time(batch.end),
                            anno.name,
                        )
                    )
                else:
                    msg = "processed {0:s} with annotator: '{1:s}'"
                    logging.info(msg.format(batch.fnames[0], anno.name))

            if max_batch is not None:
                if batch.bnum >= max_batch - 1:
                    return

    def clear(self):
        """Clear the pipeline of annotators and remove all output.

        Runs the clear method of each annotator and resets the pipeline and
        output attributes. Useful for recovering memory on the GPU when running
        a collection of large models.
        """
        for annotator in self.pipeline.values():
            annotator.clear()

        self.pipeline = collections.OrderedDict()
        self.output = collections.OrderedDict()

    def collect(self, aname):
        """Collect output from a specific annotator.

        Args:
            aname (str): Name of the annotator from which to collect the data.

        Returns:
            A DictFrame object with the results.
        """
        return stack_dict_frames(self.output[aname])

    def collect_all(self):
        """Collect output from all available annotators.


        Returns:
            An ordered dictionary with elements that are DictFrame objects. The
            names of the entries are given by the annotator names in the
            pipeline.
        """
        ocollect = collections.OrderedDict.fromkeys(self.pipeline.keys())

        for k in ocollect.keys():
            ocollect[k] = self.collect(k)

        return ocollect


class FrameAnnotator:
    """Base class for annotating a batch of frames.

    Attributes:
        name (str): A description of the annotator, used as a key in the output
            returned by a FrameProcessor.
        cache (dict): Holds internal state of the annotator. Used for passing
            data between batches, which should be avoided whenever possible.
    """

    name = "base"

    def __init__(self):
        """Create a new empty FrameAnnotator.
        """
        self.cache = {}

    def clear(self):
        """Clear the internal state of an annotator.
        """
        self.cache = {}

    def start(self, ival):
        """Initialize internal state using metadata from the input.

        Some annotators may need to perform an expensive sequence of set up
        algorithms before processing data. Often the set up requires knowledge
        about the input data. It is useful to do this just once, which can be
        accomplished by putting the code in this method. It gets called once
        when calling the process method of a FrameProcessor.

        Args:
            ival: A FrameInput object.
        """
        pass

    def annotate(self, batch):
        """Annotate a batch of frames and return the resulting annotations.

        This method contains the core functionality of an annotator. It takes
        a batch of frames and returns the annotated output as a list object.
        Lists from batch to batch will be appended together and collected by
        calling stack_dict_frames.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            The method should return a list of item(s) that can be combined
            into a DictFrame. Specifically, the items should be dictionaries
            with a consistent set of keys where all items have the same length
            (or first shape value, in the case of numpy array). Can also return
            None, in which case nothing is added to the current output.
        """
        return None


class FrameInput:
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

    def __init__(self, input_path, bsize=256):
        """Construct a new input from a video file.

        Args:
            input_path (str): Path to the video file. Can be any file readable
                by the OpenCV function VideoCapture.
            bsize (int): Number of frames to include in a batch. Defaults to
                256.
        """
        self.bsize = bsize
        self.fcount = 0
        self.vname = os.path.basename(input_path)
        self.continue_read = True
        self.start = 0
        self.end = 0
        self._video_cap = cv2.VideoCapture(input_path)
        self.meta = self._metadata()

        self._img = np.zeros(
            (bsize * 2, self.meta["height"], self.meta["width"], 3), dtype=np.uint8
        )
        self._fill_bandwidth()  # fill the buffer with the first batch
        self._continue = True  # is there any more input left in the video

    def next_batch(self):
        """Move forward one batch and return the current FrameBatch object.

        Returns:
            A FrameBatch object that contains the next set of frames.
        """

        assert self.continue_read, "No more input to read."

        # shift window over by one bandwidth
        self._img[: self.bsize, :, :, :] = self._img[self.bsize :, :, :, :]

        # fill up the bandwidth; with zeros as and of video input
        if self._continue:
            self._fill_bandwidth()
        else:
            self.continue_read = self._continue
            self._img[self.bsize :, :, :, :] = 0

        # update counters
        frame_start = self.fcount
        self.start = self.end
        self.end = self._video_cap.get(cv2.CAP_PROP_POS_MSEC)
        self.fcount = self.fcount + self.bsize

        # get frame names
        fnames = list(range(int(frame_start), int(frame_start + self.bsize)))

        # return batch of frames.
        return FrameBatch(
            img=self._img,
            vname=self.vname,
            start=self.start,
            end=self.end,
            continue_read=self.continue_read,
            fnames=fnames,
            bnum=(frame_start // self.bsize),
        )

    def _metadata(self):
        """Fill metadata attribute using metadata from the video source.
        """
        return {
            "type": "video",
            "fps": self._video_cap.get(cv2.CAP_PROP_FPS),
            "frames": int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "height": int(self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "width": int(self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        }

    def _fill_bandwidth(self):
        """Read in the next set of frames from disk and store results.

        This should not be called directly, but only through the next_batch
        method. Otherwise the internal counters will become inconsistent.
        """
        for idx in range(self.bsize):
            self._continue, frame = self._video_cap.read()
            if self._continue:
                rgb_id = cv2.COLOR_BGR2RGB
                self._img[idx + self.bsize, :, :, :] = cv2.cvtColor(frame, rgb_id)
            else:
                self._img[idx + self.bsize, :, :, :] = 0


class ImageInput:
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

    def __init__(self, input_paths, vname=""):
        """Construct a new input from a set of paths.

        Args:
            input_paths (str or list): Paths the images. Will use glob
                expansion on the elements.
            vname (str): Name of the input. Defaults to an empty string.
        """
        self.bsize = 1
        self.vname = vname
        self.continue_read = True
        self.fcount = 0
        self.start = 0
        self.meta = {"type": "image", "height": -1, "width": -1}

        # find input paths
        if not isinstance(input_paths, list):
            input_paths = [input_paths]

        input_paths = [glob.glob(x, recursive=True) for x in input_paths]
        self.paths = list(itertools.chain.from_iterable(input_paths))

    def next_batch(self):
        """Move forward one batch and return the current FrameBatch object.

        Returns:
            A FrameBatch object that contains the next set of frames.
        """

        assert self.continue_read, "No more input to read."

        this_index = self.fcount

        # read the next image and create buffer
        img = cv2.imread(self.paths[this_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.stack([img, np.zeros_like(img)])

        # is this the last image?
        self.fcount += 1
        if self.fcount >= len(self.paths):
            self.continue_read = False

        # return batch of frames.
        return FrameBatch(
            img=img,
            vname=self.vname,
            start=float(this_index),
            end=float(this_index),
            continue_read=self.continue_read,
            fnames=[self.paths[this_index]],
            bnum=this_index,
        )


class FrameBatch:
    """A collection of frames and associated metadata.

    The batch contains an array of size (bsize * 2, width, height, 3). At the
    start and end of the video file, the array is padded with zeros (an all
    black frame). The batch includes twice as many frames as given in the
    batch size, but an annotator should only return results from the first
    half of the data (the "batch"). The other data is included for annotators
    that need to look ahead of the current, such as the cut detectors.

    Attributes:
        img (np.array): A four-dimensional array containing pixels from the
            next 2*bsize of images.
        vname (str): Name of the video file.
        start (float): Time code at the start of the current batch.
        end (float): Time code at the end of the current batch.
        continue_read (bool): Indicates whether there more frames to read from
            the input.
        fnames (list): Names of frames in the batch.
        bsize (int): Number of frames in a batch.
        bnum (int): The batch number.

    """

    def __init__(self, img, vname, start, end, continue_read, fnames, bnum):
        self.img = img
        self.vname = vname
        self.start = start
        self.end = end
        self.continue_read = continue_read
        self.fnames = fnames
        self.bsize = img.shape[0] // 2
        self.bnum = bnum

    def get_frames(self):
        """Return the entire image dataset for the batch.

        Use this method if you need to look ahead at the following batch for
        an annotator to work. Images are given in RGB space.

        Returns:
            A four-dimensional array containing pixels from the current and
            next batches of data.
        """
        return self.img

    def get_batch(self):
        """Return image data for just the current batch.

        Use this method unless you have a specific need to look ahead at new
        values in the data. Images are given in RGB space.

        Returns:
            A four-dimensional array containing pixels from the current batch
            of images.
        """
        return self.img[: self.bsize, :, :, :]

    def get_frame_names(self):
        """Return frame names for the current batch of data.

        Returns:
            A list of names of length equal to the batch size.
        """
        return self.fnames
