# -*- coding: utf-8 -*-
"""Core objects.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict

from pandas import concat, DataFrame

from .utils import process_output_values

# Abstract Classes that must be implemented


class VisualInput(ABC):     # pragma: no cover
    """Base class for producing batches of visual input.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Create a new input object.
        """
        return

    @abstractmethod
    def open_input(self):
        """Restart and initialize the input feed.
        """
        return

    @abstractmethod
    def next_batch(self):
        """Move forward one batch and return the current FrameBatch object.
        """
        return

    @abstractmethod
    def get_metadata(self):
        """Provide metadata from the input connection.
        """
        return


class FrameAnnotator(ABC):   # pragma: no cover
    """Base class for annotating a batch of frames.

    Attributes:
        name (str): A description of the annotator, used as a key in the output
            returned by a FrameProcessor.
        cache (dict): Holds internal state of the annotator. Used for passing
            data between batches, which should be avoided whenever possible.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Create a new empty FrameAnnotator.
        """
        return

    @abstractmethod
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
        return


class Aggregator(ABC):    # pragma: no cover
    """Base class for aggregating the output from a pipeline of processors.

    Attributes:
        name (str): A description of the aggregator.
    """

    @abstractmethod
    def __init__(self, **kargs):
        """Create a new empty Aggregator.
        """
        return

    @abstractmethod
    def aggregate(self, ldframe, **kwargs):
        """Aggregate annotations.

        Args:
            ldframe (dict): A dictionary of data pased from a DataExtraction
                object.

        Returns:
            An object that can be processed by process_output_values.
        """
        return


# Core Objects that have a single implementation


class DataExtraction:
    """Contains all the metadata extracted from an input object.
    """

    def __init__(self, vinput):
        self.vinput = vinput
        self.data = OrderedDict()

    def run_annotators(self, annotators, max_batch=None):
        """Run the extraction.
        """
        self.vinput.open_input()

        pipeline = {anno.name: anno for anno in annotators}
        output = OrderedDict({key: [] for key in pipeline.keys()})
        output['meta'] = process_output_values(self.vinput.get_metadata())

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

        for key, value in output.items():
            if value:
                self.data[key] = concat(value, ignore_index=True)
            else:
                self.data[key] = DataFrame()

    def run_aggregator(self, aggregator):
        """Run an aggregator.
        """

        value = process_output_values(aggregator.aggregate(self.data))
        if value:
            self.data[aggregator.name] = concat(value, ignore_index=True)
        else:
            self.data[aggregator.name] = DataFrame()

    def get_data(self):
        """Get dataset from the object.
        """

        return self.data


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
        bsize (int): Number of frames in a batch.
        bnum (int): The batch number.

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
        return self.img[: self.bsize, :, :, :]

    def get_frame_names(self):
        """Return frame names for the current batch of data.

        Returns:
            A list of names of length equal to the batch size.
        """
        return self.fnames
