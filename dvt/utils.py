# -*- coding: utf-8 -*-
"""Utility functions used across the Distant Viewing Toolkit.
"""

import collections
import itertools

import cv2
import numpy as np


class DictFrame(collections.OrderedDict):
    """Object for storing information from annotations.

    A dictionary frame (DictFrame) is an extension of an ordered dictionary.
    It has an additional shape attribute that provides the tuple (nrow, ncol).
    The second attribute gives the number of items in the dictionary and is
    equivalent to calling the length of the dictionary. The first describes
    the size of the values in the dictionary. Every value should either be a
    list of length nrow or a numpy array with a first dimension of nrow. Set
    the check flag to True (the default) on construction to verify that these
    parameters are valid.

    If an new dictionary frame is constructed with an input value containing a
    data type that is neither an array nor a list, the function attempts to
    wrap the object in a list of length one. This will only produce a valid
    output if all of the inputs are of length one. The behavior described here
    is useful when stacking together individual observations, allowing users
    to supply data of (for example) integers, strings, and floats.

    When directly modifying this object, note that it is possible to break the
    special structure of the dictionary. It is best to create new dictionary
    frames using the constructor or the stack_dict_frames rather than modifying
    the original data.

    The design of a dictionary frame is inspired by R data frame, which allow
    for columns containing matrices, and is simultaneously an extension and
    stripped down version of a pandas data frame.

    Attributes:
        data (dict): A dictionary containing elements that are either lists or
            numpy arrays. All have the same length / first dimension.
        shape (tuple): The number of rows and columns in the dataset.
    """

    def __init__(self, data=None, check=True):
        if data is None:
            data = {}

        super().__init__(_process_df_input(data))
        self.shape = self._compute_shape()
        if check:
            self.verify()

    def __add__(self, dframe):
        return stack_dict_frames([self, dframe])

    def todf(self):
        """Convert to pandas data frame.

        Numpy arrays are expanded to conform with the structure of a pandas
        data frame. Useful for saving as a CSV file and creating simple plots.

        Returns:
            A pandas data frame.
        """
        import pandas as pd

        dobj = dict(self.items())
        marray = []

        base_keys = set(dobj.keys())
        for k in base_keys:
            if isinstance(dobj[k], np.ndarray):
                if len(dobj[k].shape) >= 2:
                    dfo = pd.DataFrame(dobj.pop(k))
                    dfo.columns = [k + "-" + str(x) for x in dfo.columns]
                    marray.append(dfo)

        return pd.concat([pd.DataFrame(dobj)] + marray, axis=1)

    def verify(self):
        """Verify that the shapes and types of the objects.

        Checks that all of the values in the dictionary are either a list or
        a numpy array. Also verifies the length (list) or first dimension
        (numpy array) are equal to the number of columns describing the object.
        Throws and assertion error if any of these are untrue.
        """

        for key, value in self.items():
            assert isinstance(value, (list, np.ndarray)), (
                "Error in data type of '" + key + "'."
            )
            assert len(value) == self.shape[0], "Error in length of '" + key + "'."

    def _compute_shape(self):
        """ """

        ncol = len(self.items())
        if ncol:
            nrow = len(next(iter(self.values())))
        else:
            nrow = 0

        return (nrow, ncol)


def stack_dict_frames(ilist, check=True):
    """Combine a list of objects into a dictionary frame.

    The input list must consist of dictionary frames (or items that can
    converted to dictionary frames) all containing the same set of keys and
    corresponding data types.

    Args:
        ilist (list): A list of dictionary frames or items that can be
            converted to a dictionary frame.
        check (bool): Should checks be run on the dictionary frames. True by
            default.

    Returns:
        A single dictionary frame object.
    """

    # Convert all elements to DictFrame objects; remove None items
    ilist = [DictFrame(x, check=check) for x in ilist if x is not None]

    # If empty, return an empty DictFrame object
    if not ilist:
        return DictFrame({})

    # What keys will be in the output and what are their types? Take these
    # from the first element
    base_keys = set(ilist[0].keys())
    is_key_np = [isinstance(ilist[0][k], np.ndarray) for k in base_keys]

    # Unless 'check' is switched off (dangerous), check that each element
    # has the same set of keys and same data types
    if check:
        for item in ilist:
            assert set(item.keys()) == base_keys
            for k, is_np in zip(base_keys, is_key_np):
                if is_np:
                    assert isinstance(item[k], np.ndarray)

    # Create a new empty dictionary from the base keys
    out = collections.OrderedDict.fromkeys(base_keys)

    # Fill in values for the dictionary by combining types (ndarray or list)
    for k in out.keys():
        if isinstance(ilist[0][k], np.ndarray):
            out[k] = np.concatenate([x[k] for x in ilist], axis=0)
        else:
            out[k] = list(itertools.chain.from_iterable([x[k] for x in ilist]))

    return DictFrame(out)


def pd_to_dict_frame(pdf, use_array=True):
    """Convert a pandas data frame into a dictionary frame.

    Useful for reading a csv file and reconverting into a dictionary frame.

    Args:
        use_array (bool): Should the function attempt to recreate the array
            structure. Default to True. Currently unimplemented.
    """
    out = DictFrame(pdf.to_dict(orient="list"))

    return out


def get_batch(input_obj, batch_num=0):
    """Manually extract a batch object from an input.

    This function is mostly useful for testing new annotators. Generally,
    batches should be sent to annotators by using the FrameProcesser.

    Args:
        input_obj: A FrameInput object.
        batch_num: What batch number to take as the output. Default is 0, the
            first batch.

    Returns:
        The selected batch object.
    """

    bnum = 0
    while input_obj.continue_read:
        batch = input_obj.next_batch()
        if bnum == batch_num:
            break
        bnum += 1

    return batch


def sub_image(img, top, right, bottom, left, fct=1, output_shape=None):
    """Take a subset of an input image and return a (resized) subimage.

    Args:
        img (numpy array): Image stored as a three-dimensional image.
        top (int): Top coordinate of the new image.
        right (int): Right coordinate of the new image.
        bottom (int): Bottom coordinate of the new image.
        left (int): Left coordinate of the new image.
        fct (float): Percentage to expand the bounding box by. Defaults to
            1, using the input coordinates as given.
        output_shape (tuple): Size to scale the output image, in pixels. Set
            to None (default) to keep the native resolution.

    Returns:
        A three-dimensional numpy array describing the new image.
    """

    # convert to center, height and width:
    center = [int((top + bottom) / 2), int((left + right) / 2)]
    height = int((bottom - top) / 2 * fct)
    width = int((right - left) / 2 * fct)
    box = [center[0] - height, center[0] + height, center[1] - width, center[1] + width]

    # crop the image as an array
    box[0] = max(0, box[0])
    box[2] = max(0, box[2])
    box[1] = min(img.shape[1], box[1])
    box[3] = min(img.shape[0], box[3])
    crop_img = img[box[0] : box[1], box[2] : box[3], :]

    if output_shape:
        img_scaled = cv2.resize(crop_img, output_shape)
    else:
        img_scaled = crop_img

    return np.uint8(img_scaled)


def _process_df_input(data):
    """Helper function to construct a dictionary frame.

    Args:
        data (dict): Input dataset.

    Returns:
        An ordered dictionary containing values that are only either arrays or
        lists.
    """

    for key, value in data.items():
        if not isinstance(value, (list, np.ndarray)):
            data[key] = [value]

    return collections.OrderedDict(data)


def _format_time(msec):
    """Takes a millisecond and produces a ISO-8601 formatted string.

    Args:
        msec (int): Time in milliseconds.

    Returns:
        String of the data in ISO-8601 format.
    """
    msec = int(msec)
    hour = msec // (1000 * 60 * 60)
    minute = (msec % (1000 * 60 * 60)) // (1000 * 60)
    second = (msec % (1000 * 60)) // (1000)
    remainder = msec % 1000

    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(
        hour, minute, second, int(remainder)
    )


def _trim_bbox(css, image_shape):
    """Given a bounding box and image size, returns a new trimmed bounding box.

    Some algorithms produce bounding boxes that extend over the edges of the
    source image. This function takes such a box and returns a new bounding
    box trimmed to the source.

    Args:
        css (array): An array of dimension four.
        image_shape (array): An array of dimension two.

    Returns:
        An updated bounding box trimmed to the extend of the image.
    """
    return (
        max(css[0], 0),
        min(css[1], image_shape[1]),
        min(css[2], image_shape[0]),
        max(css[3], 0),
    )
