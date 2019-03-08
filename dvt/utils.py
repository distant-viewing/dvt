# -*- coding: utf-8 -*-

import collections
import itertools
import numpy as np


class DictFrame(collections.OrderedDict):

    def __init__(self, data=None, check=True):
        if data is None:
            data = {}

        super().__init__(self._process_input(data))
        self.shape = self._compute_shape()
        if check:
            self._check_data()

    def __add__(self, x):
        return stack_dict_frames([self, x])

    def todf(self):
        import pandas as pd

        d = dict(self.items())
        marray = []

        base_keys = set(d.keys())
        for k in base_keys:
            if isinstance(d[k], np.ndarray):
                if len(d[k].shape) >= 2:
                    df = pd.DataFrame(d.pop(k))
                    df.columns = [k + "-" + str(x) for x in df.columns]
                    marray.append(df)

        return pd.concat([pd.DataFrame(d)] + marray, axis=1)

    def _check_data(self):

        for key, value in self.items():
            assert isinstance(value, (list, np.ndarray))
            assert len(value) == self.shape[0]

    def _compute_shape(self):

        ncol = len(self.items())
        if ncol:
            nrow = len(next(iter(self.values())))
        else:
            nrow = 0

        return (nrow, ncol)

    def _process_input(self, data):

        for key, value in data.items():
            if not isinstance(value, (list, np.ndarray)):
                data[key] = [value]

        return collections.OrderedDict(data)


def stack_dict_frames(ilist, check=True):

    # Convert all elements to DictFrame objects; remove None items
    ilist = [DictFrame(x, check=check) for x in ilist if x is not None]

    # If empty, return an empty DictFrame object
    if not len(ilist):
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


def pd_to_dict_frame(df, use_array=True):
    out = DictFrame(df.to_dict(orient='list'))
    if use_array:
        pass

    return out


def get_batch(input_obj, batch_num=0):

    bnum = 0
    while input_obj.continue_read:
        batch = input_obj.next_batch()
        if bnum == batch_num:
            return batch
        else:
            bnum += 1

    return batch


def sub_image(img, top, right, bottom, left, fct=1, output_shape=None):
    from skimage.transform import resize

    # convert to center, height and width:
    c = [int((top + bottom) / 2), int((left + right) / 2)]
    h = int((bottom - top) / 2 * fct)
    w = int((right - left) / 2 * fct)
    d = [c[1] - h, c[1] + h, c[0] - w, c[0] + w]

    # crop the image as an array
    d[0] = max(0, d[0])
    d[2] = max(0, d[2])
    d[1] = min(img.shape[1], d[1])
    d[3] = min(img.shape[0], d[3])
    crop_img = img[d[2]:d[3], d[0]:d[1], :]

    if output_shape:
        img_scaled = resize(crop_img, output_shape, mode='constant',
                            anti_aliasing=True)

    return img_scaled


def dict_to_dataframe(d):
    import pandas as pd
    marray = []

    for k in d.keys():
        if isinstance(d[k], np.ndarray):
            if len(d[k].shape) >= 2:
                df = pd.DataFrame(d.pop(k))
                df.columns = [k + str(x) for x in df.columns]
                marray.append(df)

    return pd.concat([pd.DataFrame(d)] + marray, axis=1)


def _format_time(msec):
    msec = int(msec)
    hour = msec // (1000 * 60 * 60)
    minute = (msec % (1000 * 60 * 60)) // (1000 * 60)
    second = (msec % (1000 * 60)) // (1000)
    remainder = msec % 1000

    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(hour, minute, second,
                                                    int(remainder))


def _trim_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), \
           min(css[2], image_shape[0]), max(css[3], 0)
