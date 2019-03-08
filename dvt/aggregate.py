# -*- coding: utf-8 -*-

from .utils import stack_dict_frames

class Aggregator:

    def __init__(self):
        pass

    def aggregate(self, ldframe):
        pass


class CutAggregator(Aggregator):

    def __init__(self, min_len=1, ignore_vals=None, cut_vals=None):
        if ignore_vals is None:
            ignore_vals = {}

        if cut_vals is None:
            cut_vals = {}

        self.min_len = min_len
        self.ignore_vals = ignore_vals
        self.cut_vals = cut_vals

    def aggregate(self, ldframe):

        # grab the data, initialize counters, and create output `cuts`
        op = ldframe['diff']
        this_video = ""
        ignore_this_frame = True
        current_cut_start = 0
        cuts = []

        # cycle through frames and collection shots; assumes that the data is
        # grouped by video and ordered by frame
        mlen = len(op['frame'])
        for ind in range(mlen):
            this_frame = op['frame'][ind]

            # if this a new video, restart the frame numbering
            if this_video != op['video'][ind]:
                this_video = op['video'][ind]
                ignore_last_frame = False
                current_cut_start = this_frame

            # check to see if we should ignore the next frame; by default we
            # ignore the phantom frame at the end of the video at time T+1.
            ignore_next_frame = False
            if (ind + 1) >= mlen:
                ignore_next_frame = True
            else:
                for key, coff in self.ignore_vals.items():
                    if op[key][ind + 1] < coff:
                        ignore_next_frame = True
                        break

            # check if there should be a cut; note: this is defined such that
            # the this_frame is the *last* frame in the current cut, not the
            # first in the next cut
            if (this_frame - current_cut_start + 1) >= self.min_len and not ignore_next_frame:
                cut_detect = True
                for key, coff in self.cut_vals.items():
                    if op[key][ind] < coff:
                         cut_detect = False
                         break
            else:
                cut_detect = False

            if ignore_next_frame and not ignore_this_frame:
                cut_detect = True

            # if `cut_detect` at this point, then we want to finish the active
            # cut with the current frame
            if cut_detect:
                cuts.append({'video': this_video,
                             'frame_start': current_cut_start,
                             'frame_end': this_frame})

            if cut_detect or ignore_next_frame:
                current_cut_start = this_frame + 1

            # push forward the ignore flag
            ignore_this_frame = ignore_next_frame

        return stack_dict_frames(cuts)
