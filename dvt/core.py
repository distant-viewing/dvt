# -*- coding: utf-8 -*-
"""Core objects for working with the distant viewing toolkit.

The objects VisualInput, FrameAnnotator, and Aggregator are abstract classes.
They correspond to the method for extracting visual inputs from digitized
files, methods for extracting metadata from visual images, and methods for
aggregating the data across a collection. The toolkit provides many ready-made
implementions of these classes in order to solve common tasks.

A DataExtraction class is constructed for each input object. Annotators and
aggregators can be iteratively passed to the object to extract metadata. The
FrameBatch object serves as the primary internal structure for storing visual
information. Users who construct their own FrameAnnotator subclasses will need
to interface with these objects and their methods for grabbing visual data
inputs. See the example in DataExtraction for the basic usage of the classes.
"""

from collections import OrderedDict
from glob import glob
from itertools import chain
from math import ceil

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
from pandas import concat, DataFrame
from progress.bar import Bar
from scipy.io.wavfile import read

from .abstract import VisualInput
from .utils import (
    process_output_values,
    _data_to_json,
    _expand_path,
    _subtitle_data
)


class DataExtraction:
    """The core class of the toolkit. Used to pass algorithms to visual data.

    Each instance of a data extraction is tied to a particular input object.
    Collections of annotators or individual aggregators can be passed to the
    relevant methods to extract metadata from the associated input.

    Attributes:
        vinput (VisualInput): The input object associated with the dataset.
        ainput (str): Path to audio input as wav file. Optional.
        sinput (str): Path to subtitle input as srt file. Optional.
        data (OrderedDict): Extracted metadata.

    Example:
        Assuming we have an input named "input.mp4", the following example
        shows a straightforward usage of DataExtraction. We create an
        extraction object, pass through a difference annotator, and then
        aggregate use the cut detector. Finally, the output from the cut
        aggregator is obtained by calling the get_data method and grabbing
        the relevant key ("cut").

        >>> from dvt.core import DataExtraction, FrameInput
        >>> from dvt.annotate.diff import DiffAnnotator
        >>> from dvt.aggregate.cut import CutAggregator

        >>> dextra = DataExtraction(FrameInput(input_path="input.mp4"))
        >>> dextra.run_annotators([DiffAnnotator(quantiles=[40])])
        >>> dextra.run_aggregator(CutAggregator(cut_vals={'q40': 3}))

        >>> dextra.get_data()['cut']

        Using the input file in the Distant Viewing Toolkit test directory
        yields the following output:

        >>> dextra.get_data()['cut']
           frame_start  frame_end
        0            0         74
        1           75        154
        2          155        299
        3          300        511

    """

    def __init__(self, vinput, ainput=None, sinput=None):
        self.vinput = vinput
        self.ainput = ainput
        self.sinput = sinput
        self.data = OrderedDict({
            "meta": process_output_values(self.vinput.get_metadata())
        })

    def run_annotators(self, annotators, max_batch=None, msg="Progress: "):
        """Run a collection of annotators over the input material.

        Batches of inputs are grabbed from vinput and passed to the annotators.
        Output is collapsed into one DataFrame per annotator, and stored as
        keys in the data attribute. An additional key is included ("meta")
        that contains metadata about the collection.

        Args:
            annotators (list): A list of annotator objects.
            max_batch (int): The maximum number of batches to process. Useful
                for testing and debugging. Set to None (default) to process
                all available frames.
            progress (bool): Should a progress bar be shown over each batch?
            msg (str): Message to display in the progress bar, if used. Set
                to None to supress the message
        """
        self.vinput.open_input()

        pipeline = {anno.name: anno for anno in annotators}
        output = OrderedDict({key: [] for key in pipeline.keys()})
        output["meta"] = process_output_values(self.vinput.get_metadata())

        # setup progress bar
        mbatch = self.vinput.max_batch
        if msg is not None:
            progress_bar = Bar(msg, max=(
                mbatch if max_batch is None else min(mbatch, max_batch)
            ))

        # cycle through batches and process the file
        cread = self.vinput.continue_read
        while cread:
            batch = self.vinput.next_batch()
            for anno in pipeline.values():
                next_values = anno.annotate(batch)
                if next_values is not None:
                    output[anno.name].extend(process_output_values(
                        next_values
                    ))

            cread = self.vinput.continue_read
            if max_batch is not None:
                if batch.bnum >= max_batch - 1:
                    cread = False

            if msg is not None:
                progress_bar.next()

        for key, value in output.items():
            if value:
                self.data[key] = concat(value, ignore_index=True)
            else:
                self.data[key] = DataFrame()

        if msg is not None:
            progress_bar.finish()

    def run_audio_annotator(self):
        """Run the audio annotator.

        After running this method, two new annotations are given: 'audio' and
        'audiometa'. They contain all of the sound data as a DataFrame
        objects.
        """

        # check sound exists and read
        assert self.ainput is not None
        rate, data_in = read(self.ainput)

        output = {}
        if len(data_in.shape) == 2:
            output['data'] = (data_in[:, 0] + data_in[:, 1]) // 2
            output['data_left'] = data_in[:, 0]
            output['data_right'] = data_in[:, 1]
        else:
            output['data'] = data_in

        self.data["audio"] = process_output_values(output)[0]
        self.data["audiometa"] = process_output_values({'rate': rate})[0]

    def run_subtitle_annotator(self):
        """Run the subtitle annotator.

        After running this method, a new annotation called 'subtitle' will be
        added to the DataExtraction object. Requires that the attribue sinput
        is set to a valid path.
        """

        # open video input to get the video metadata
        self.vinput.open_input()
        self.data['meta'] = process_output_values(
            self.vinput.get_metadata()
        )[0]

        # check sound exists and get the annotation
        assert self.sinput is not None
        self.data["subtitle"] = _subtitle_data(self.sinput, self.data['meta'])

    def run_aggregator(self, aggregator):
        """Run an aggregator over the extracted annotations.

        Args:
            aggregator (Aggregator): Aggregator object use for processing the
                input data.
        """

        value = process_output_values(aggregator.aggregate(self.data))
        if value:
            self.data[aggregator.name] = concat(value, ignore_index=True)
        else:
            self.data[aggregator.name] = DataFrame()

    def get_data(self):
        """Get dataset from the object.

        Returns:
            An ordered dictionary where each key corresponds to an annotator
            or aggregator and the values are all pandas DataFrame objects.
        """

        return self.data

    def get_json(self, path=None, exclude_set=None, exclude_key=None):
        """Get dataset as a JSON object.

        Args:
            path_or_buf: Location to store the output. If set to None, return
                as a string.
            exclude_set: Set of dataset names to ignore when creating the
                output. None, the default, includes all data in the output.
            exclude_key: Set of column names to ignore when creating the
                output. None, the default, includes all keys in the output.
        """

        return _data_to_json(self.data, path, exclude_set, exclude_key)


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
        start (float): Time code at the start of the current batch.
        end (float): Time code at the end of the current batch.
        continue_read (bool): Indicates whether there more frames to read from
            the input.
        fnames (list): Names of frames in the batch.
        bnum (int): The batch number.
        bsize (int): Number of frames in a batch.
    """

    def __init__(self, **kwargs):
        self.img = kwargs.get("img")
        self.start = kwargs.get("start")
        self.end = kwargs.get("end")
        self.continue_read = kwargs.get("continue_read")
        self.fnames = kwargs.get("fnames")
        self.bnum = kwargs.get("bnum")
        self.bsize = self.img.shape[0] // 2

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
        return self.img[:self.bsize, :, :, :]

    def get_frame_names(self):
        """Return frame names for the current batch of data.

        Returns:
            A list of names of length equal to the batch size.
        """
        return self.fnames


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
        self.max_batch = 0
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
        self.max_batch = ceil(self.get_metadata()['frames'] / self.bsize)

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
        self.max_batch = len(self.paths)

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
            fnames=[int(this_index)],
            bnum=this_index
        )

    def get_metadata(self):
        return self.meta
