# -*- coding: utf-8 -*-
"""This module illustrates something.
"""

import collections
import itertools

import numpy as np


class DictFrame(collections.OrderedDict):
    """Here"""

    def __init__(self, data=None, check=True):
        if data is None:
            data = {}

        super().__init__(_process_df_input(data))
        self.shape = self._compute_shape()
        if check:
            self._check_data()

    def __add__(self, dframe):
        return stack_dict_frames([self, dframe])

    def todf(self):
        """Here"""
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

    def _check_data(self):
        """ """

        for _, value in self.items():
            assert isinstance(value, (list, np.ndarray))
            assert len(value) == self.shape[0]

    def _compute_shape(self):
        """ """

        ncol = len(self.items())
        if ncol:
            nrow = len(next(iter(self.values())))
        else:
            nrow = 0

        return (nrow, ncol)


def stack_dict_frames(ilist, check=True):
    """Here

    :param ilist: 
    :param check:  (Default value = True)

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
        elif isinstance(ilist[0][k], list):
            out[k] = list(itertools.chain.from_iterable([x[k] for x in ilist]))
        else:
            raise TypeError()

    return DictFrame(out)


def pd_to_dict_frame(pdf, use_array=True):
    """Here

    :param pdf: 
    :param use_array:  (Default value = True)

    """
    out = DictFrame(pdf.to_dict(orient='list'))
    if use_array:
        pass

    return out


def get_batch(input_obj, batch_num=0):
    """Here

    :param input_obj: 
    :param batch_num:  (Default value = 0)

    """

    bnum = 0
    while input_obj.continue_read:
        batch = input_obj.next_batch()
        if bnum == batch_num:
            break
        bnum += 1

    return batch


def sub_image(img, top, right, bottom, left, fct=1, output_shape=None):
    """Here

    :param img: 
    :param top: 
    :param right: 
    :param bottom: 
    :param left: 
    :param fct:  (Default value = 1)
    :param output_shape:  (Default value = None)

    """
    from skimage.transform import resize

    # convert to center, height and width:
    center = [int((top + bottom) / 2), int((left + right) / 2)]
    height = int((bottom - top) / 2 * fct)
    width = int((right - left) / 2 * fct)
    box = [center[1] - height, center[1] + height,
           center[0] - width, center[0] + width]

    # crop the image as an array
    box[0] = max(0, box[0])
    box[2] = max(0, box[2])
    box[1] = min(img.shape[1], box[1])
    box[3] = min(img.shape[0], box[3])
    crop_img = img[box[2]:box[3], box[0]:box[1], :]

    if output_shape:
        img_scaled = resize(crop_img, output_shape, mode='constant',
                            preserve_range=True, anti_aliasing=True)

    return np.uint8(img_scaled)


def dict_to_dataframe(dobj):
    """Here

    :param dobj: 

    """
    import pandas as pd
    marray = []

    for k in dobj.keys():
        if isinstance(dobj[k], np.ndarray):
            if len(dobj[k].shape) >= 2:
                pdf = pd.DataFrame(dobj.pop(k))
                pdf.columns = [k + str(x) for x in pdf.columns]
                marray.append(pdf)

    return pd.concat([pd.DataFrame(dobj)] + marray, axis=1)


def _process_df_input(data):
    """

    :param data: 

    """

    for key, value in data.items():
        if not isinstance(value, (list, np.ndarray)):
            data[key] = [value]

    return collections.OrderedDict(data)


def _format_time(msec):
    """Here

    :param msec: 

    """
    msec = int(msec)
    hour = msec // (1000 * 60 * 60)
    minute = (msec % (1000 * 60 * 60)) // (1000 * 60)
    second = (msec % (1000 * 60)) // (1000)
    remainder = msec % 1000

    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(hour, minute, second,
                                                    int(remainder))


def _trim_bounds(css, image_shape):
    """Here

    :param css: 
    :param image_shape: 

    """
    return max(css[0], 0), min(css[1], image_shape[1]), \
            min(css[2], image_shape[0]), max(css[3], 0)
