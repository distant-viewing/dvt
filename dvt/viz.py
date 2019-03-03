# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

from .utils import combine_list_dicts, dict_to_dataframe


def plot_annotations(input_dir, output_dir, obj=None, face=None, cut=None):
    # make output directory if not already exists
    if not os.path.isdir("temp-frames"):
        os.mkdir("temp-frames")

    # what frames to look at?
    frames = []
    if obj is not None:
        frames += list(obj['frame'].values)
    if face is not None:
        frames += list(face['frame'].values)
    if cut is not None:
        frames += list(cut['frame'].values)

    frames = list(set(frames))
    frames.sort()

    # process files
    for fnum in list(set(frames)):
        _process_frame(input_dir, output_dir, fnum, obj, face, cut)


def make_video(input_dir, oname="temp.mp4", mp3=None, ffmpeg="ffmpeg"):
    import subprocess

    cmd = [ffmpeg, "-r", "29.97", "-f", "image2", "-s", "710x480", "-i",
           os.path.join(input_dir, "frame-%06d.png")]

    if mp3 is not None:
        cmd += ["-i", mp3]

    cmd += ["-vcodec", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", oname]

    subprocess.run(cmd)


def _process_frame(input_dir, output_dir, fnum, obj, face, cut):
    # define colours
    box_color = (255, 165, 0)
    face_color = (22, 75, 203)
    white_color = (255, 255, 255)

    fname = 'frame-{0:06d}.png'.format(fnum)
    img = cv2.imread(os.path.join(input_dir, fname))
    img = np.vstack([img, np.zeros((150, img.shape[1], 3))])

    if obj is not None:
        img = add_bbox(img, fnum, obj, box_color, 2)
        img = add_box_text(img, fnum, obj, 'class',
                           color=white_color, bg=box_color, s=0.5)

    if face is not None:
        img = add_bbox(img, fnum, face, face_color, 1)

    if cut is not None:
        img = add_ts(img, fnum, cut, 'vals_l40', 1, white_color, s=5)
        img = add_ts(img, fnum, cut, 'vals_h16', 100, face_color, s=2)
        img = add_ts_line(img)

    x = cv2.imwrite('temp-frames/frame-{0:06d}.png'.format(fnum), img)


def add_bbox(img, frame, df, color=(255, 255, 255), thickness=2):
    for ind in np.argwhere(df['frame'].values == frame):
        # grab values from data
        top = df['top'].values[ind]
        right = df['right'].values[ind]
        bottom = df['bottom'].values[ind]
        left = df['left'].values[ind]

        # plot the rectangle
        img = cv2.rectangle(img, (left, top), (right, bottom),
                            color, thickness)

    return img


def add_box_text(img, frame, df, lvar, color=(0, 0, 0), bg=None, s=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ind in np.argwhere(df['frame'].values == frame):
        # grab values from data
        top = df['top'].values[ind]
        right = df['right'].values[ind]
        bottom = df['bottom'].values[ind]
        left = df['left'].values[ind]
        msg = list(df[lvar].values[ind])[0]

        if bg:
            # make a text box with background color bg
            (text_width, text_height) = cv2.getTextSize(msg, font,
                                                        fontScale=s,
                                                        thickness=1)[0]
            text_offset_x = left
            text_offset_y = bottom
            box_coords = ((text_offset_x, text_offset_y + 1),
                          (text_offset_x + text_width + 5,
                           text_offset_y - text_height - 10))
            img = cv2.rectangle(img, box_coords[0], box_coords[1],
                                bg, cv2.FILLED)

        # plot text and text box
        img = cv2.putText(img, msg, (left + 5, bottom - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, s, color, 1, cv2.LINE_AA)

    return img


def add_ts(img, frame, df, lvar, fct=1, color=(0, 0, 0), s=3):
    for k in range(-15, 15):
        fnum = frame + k
        for ind in np.argwhere(df['frame'].values == (fnum - 1)):
            val = int(df[lvar][ind] * fct)
            img = cv2.circle(img, (img.shape[1]//2 + k*25,
                                   img.shape[0] - val - 10),
                             s, color, -1)

    return img


def add_ts_line(img):
    img = cv2.line(img, (img.shape[1]//2, img.shape[0]),
                   (img.shape[1]//2, img.shape[0] - 150), (255,255,255), 2)
    return img
