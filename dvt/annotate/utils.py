# -*- coding: utf-8 -*-

import numpy as np

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


def combine_list_dicts(ilist):
    import itertools
    import collections

    if not ilist:
        return None

    keys = ilist[0].keys()
    d = collections.OrderedDict.fromkeys(keys)

    for k in keys:
        if isinstance(ilist[0][k], np.ndarray):
            d[k] = np.concatenate([x[k] for x in ilist], axis=0)
        elif isinstance(ilist[0][k], list):
            d[k] = list(itertools.chain.from_iterable([x[k] for x in ilist]))
        else:
            d[k] = [x[k] for x in ilist]

    return d
