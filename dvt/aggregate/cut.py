# -*- coding: utf-8 -*-
"""Aggregate frame level information to detect cuts.

The aggregator functions here take local information about frames and estimates
where cuts in the video occur.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of a CutAggregator over two batches of the input. First
    we need to collect differences between successive frames:

    >>> fp = FrameProcessor()
    >>> fp.load_annotator(DiffAnnotator(quantiles=[40]))
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    >>> obj = fp.collect_all()

    Then, construct a cut aggregator with a cut-off score for both histograms
    (0.2) and pixels ()

    >>> ca = CutAggregator(cut_vals={'h40': 0.2, 'q40': 3})
    >>> ca.aggregate(obj).todf()
           video  frame_start  frame_end
    0  input.mp4            0         92
    1  input.mp4           93        176
    2  input.mp4          177        212
    3  input.mp4          213        247
    4  input.mp4          248        293
    5  input.mp4          294        402
    6  input.mp4          403        511

    Additional options allow for ignoring frames that are too dark, have a cut
    that is too close to another, or include additional cut-off values.
"""

from ..core import Aggregator


class CutAggregator(Aggregator):
    """Uses difference between successive frames to detect cuts.

    This aggregator uses information from the difference annotator to detect
    the cuts.

    Attributes:
        min_len (int): minimum allowed length of a cut.
        ignore_vals (dict): Dictionary of cutoffs that cause a frame to be
            ignored for the purpose of detecting cuts. Keys indicate the
            variables in the differences output and values show the minimum
            value allowed for a frame to be considered for a cut. Typically
            used to ignore frames that are too dark (i.e., during fades). Set
            to None (default) if no ignored values are needed.
        cut_vals (dict): Dictionary of cutoffs that cause a frame to be
            considered a cut. Keys indicate the variables in the differences
            output and values are the cutoffs. Setting to None (default) will
            return no cuts.
    """

    name = "cut"

    def __init__(self, **kargs):

        self.ignore_vals = kargs.get("ignore_vals", {})
        self.cut_vals = kargs.get("cut_vals", None)
        self.min_len = kargs.get("min_len", 1)

        super().__init__()

    def aggregate(self, ldframe, **kwargs):
        """Aggregate difference annotator.

        Args:
            ldframe (dict): A dictionary of DictFrames from a FrameAnnotator.
                Must contain an entry with the key 'diff', which is used in the
                annotation.

        Returns:
            A dictionary frame giving the detected cuts.
        """

        # grab the data, initialize counters, and create output `cuts`
        ops = ldframe["diff"]
        ignore_this_frame = True
        current_cut_start = 0
        cuts = {'frame_start': [], 'frame_end': []}

        # cycle through frames and collection shots; assumes that the data is
        # ordered by frame
        mlen = len(ops["frame"])
        for ind in range(mlen):
            this_frame = ops["frame"][ind]

            # check to see if we should ignore the next frame; by default we
            # ignore the phantom frame at the end of the video at time T+1.
            ignore_next_frame = False
            if (ind + 1) >= mlen:
                ignore_next_frame = True
            else:
                for key, coff in self.ignore_vals.items():
                    if ops[key][ind + 1] < coff:   # pragma: no cover
                        ignore_next_frame = True
                        break

            # check if there should be a cut; note: this is defined such that
            # the this_frame is the *last* frame in the current cut, not the
            # first in the next cut
            long_flag = (this_frame - current_cut_start + 1) >= self.min_len
            if long_flag and not ignore_next_frame:
                cut_detect = True
                for key, coff in self.cut_vals.items():
                    if ops[key][ind] < coff:
                        cut_detect = False
                        break
            else:
                cut_detect = False

            if ignore_next_frame and not ignore_this_frame:
                cut_detect = True

            # if `cut_detect` at this point, then we want to finish the active
            # cut with the current frame
            if cut_detect:
                cuts["frame_start"].append(current_cut_start)
                cuts["frame_end"].append(this_frame)

            if cut_detect or ignore_next_frame:
                current_cut_start = this_frame + 1

            # push forward the ignore flag
            ignore_this_frame = ignore_next_frame

        return cuts
