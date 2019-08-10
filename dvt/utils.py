# -*- coding: utf-8 -*-
"""Core objects.
"""

from os.path import abspath, expanduser, isdir, basename, splitext
from os import makedirs

from pandas import DataFrame
from numpy import ndarray, array, isin, flatnonzero, uint8, vstack
from cv2 import resize


def process_output_values(values):
    """Take input and create pandas data frame.

    Input can be a DataFrame (just return) or a dictionary.
    """
    if isinstance(values, DataFrame):
        return [values]

    if isinstance(values, list) or values == None:
        return values

    assert isinstance(values, dict)

    # convert numpy array into a list of arrays for pandas
    for key in values.keys():
        if isinstance(values[key], ndarray) and len(values[key].shape) > 1:
            values[key] = [x for x in values[key]]

    try:
        df = DataFrame(values)
    except ValueError as ve:
        df = DataFrame(values, index=[0])

    return [df]


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
    box = [
        center[0] - height,
        center[0] + height,
        center[1] - width,
        center[1] + width,
    ]

    # crop the image as an array
    box[0] = max(0, box[0])
    box[2] = max(0, box[2])
    box[1] = min(img.shape[0], box[1])
    box[3] = min(img.shape[1], box[3])
    crop_img = img[box[0]:box[1], box[2]:box[3], :]

    if output_shape:
        img_scaled = resize(crop_img, output_shape)
    else:
        img_scaled = crop_img

    return uint8(img_scaled)


def setup_tensorflow():
    """Setup options for TensorFlow.

    These options should allow most users to run TensorFlow with either a
    GPU or CPU.
    """
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    import os

    # supress warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # ensure that keras does not use all of the available memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

    # fix a common local bug
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pd_col_asarray(pdf, column):
    return vstack(pdf[column].to_list())


def _proc_frame_list(frames):
    if frames is not None:
        frames = array(frames)
    return frames


def _which_frames(batch, freq, frames):
    """Determine which frame numbers should be used.
    """
    if frames is None:
        return list(range(0, batch.bsize, freq))

    return flatnonzero(isin(batch.fnames, frames)).tolist()


def _check_out_dir(output_dir, should_exist=False):
    if output_dir is not None:
        output_dir = abspath(expanduser(output_dir))
        if should_exist:
            assert isdir(output_dir)
        elif not isdir(output_dir):
            makedirs(output_dir)

    return output_dir


def _expand_path(path):
    path = abspath(expanduser(path))
    bname = basename(path)
    filename, file_extension = splitext(bname)
    return path, bname, filename, file_extension


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
