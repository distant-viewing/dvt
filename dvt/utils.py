# -*- coding: utf-8 -*-

import collections

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


def _format_time(msec):
    msec = int(msec)
    hour = msec // (1000 * 60 * 60)
    minute = (msec % (1000 * 60 * 60)) // (1000 * 60)
    second = (msec % (1000 * 60)) // (1000)
    remainder = msec % 1000

    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(hour, minute, second,
                                                    int(remainder))

def combine_list_dicts(ilist):
    d = collections.OrderedDict()
    for k in ilist[0].keys():
        d[k] = []

    for item in ilist:
        for k in item.keys():
            if isinstance(item[k], np.ndarray):
                d[k].append(item[k])
            else:
                d[k] += item[k]

    for k in ilist[0].keys():
        if isinstance(ilist[0][k], np.ndarray):
            d[k] = np.hstack(d[k])

    return d





