# -*- coding: utf-8 -*-

def _simple_cuts(op, l40, h16, av, min_len):
    cuts = []
    first_frame = op['frame'][0]
    dark_flag = True
    for ind in range(len(op['frame'])):
        this_frame = op['frame'][ind]

        # determine if the shot is (mostly) black, or not; determine next
        # step accordingly
        if op['avg_value'][ind] < 3:
            if not dark_flag:
                break_cut = True
            else:
                break_cut = False
                first_frame = this_frame + 1

            dark_flag = True

        else:
            dark_flag = False
            break_cut = (op['vals_l40'][ind] > l40 and \
                         op['vals_h16'][ind] > h16 and \
                         (this_frame - first_frame) > min_len)

        # always break at the end of the video
        if ind == (len(op['frame']) - 1):
            break_cut = True

        # if there is a break, add it to the list
        if break_cut:
            cuts.append({'frame_start': first_frame,
                         'frame_end': this_frame})
            first_frame = this_frame + 1

    return cuts
