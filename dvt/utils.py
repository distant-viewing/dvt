# -*- coding: utf-8 -*-
"""Annotators for extracting high-level metadata about the images in the input.
"""

import os.path
import re
import sys

import cv2
import numpy as np
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse


def load_video(video_path: str, height: int, width: int):
    """Load a video file as a numpy array.

    Args:
        video_path (str): path to the video file.
        height (int): desired height of the output frames.
        width (int): desired width of the output frames.

    Returns:
        A numpy array of size N x H x W x C, where N is the
        number of frames, H the height, W the width, and C 
        the number of color channels.
    """
    vinput = cv2.VideoCapture(_expand_path(video_path))
    N = int(vinput.get(cv2.CAP_PROP_FRAME_COUNT))
    video = np.zeros((N, height, width, 3), dtype=np.uint8)

    inum = 0
    while True:
        status, img = vinput.read()
        if not status:
            break
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        video[inum] = frame
        inum += 1

    return video


def yield_video(video_path: str):
    """Generator to cycle through frames of a video file.

    Args:
        video_path (str): path to the video file.

    Returns:
        A generator object that yields a tuple giving the
        image (as a numpy array), the frame count number (int),
        and the time at the start of the frame in seconds (float)
    """
    vinput = cv2.VideoCapture(video_path)

    while True:
        status, img = vinput.read()
        if not status:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = int(vinput.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        msec = frames / vinput.get(cv2.CAP_PROP_FPS)
        yield img, frames, msec


def video_info(video_path: str) -> dict:
    """Return metadata about a video file.

    Args:
        video_path (str): path to the video file.

    Returns:
        A dictionary of a metadata about the video file.
    """
    vinput = cv2.VideoCapture(_expand_path(video_path))
    output = {
        "frame_count": vinput.get(cv2.CAP_PROP_FRAME_COUNT),
        "height": vinput.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "width": vinput.get(cv2.CAP_PROP_FRAME_WIDTH),
        "fps": vinput.get(cv2.CAP_PROP_FPS),
    }
    return output


def load_image(image_path: str) -> np.ndarray:
    """Return metadata about a video file.

    Args:
        image_path (str): path to an image file.

    Returns:
        A numpy array of the image as an RGB array.
    """
    img = cv2.imread(_expand_path(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image_path: str, img: np.ndarray) -> None:
    """Return metadata about a video file.

    Args:
        image_path (str): path to the output image file.
        img (np.ndarray): image file to save, in RGB format.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(_expand_path(image_path), img)


def _download_file(
    url: str, basename="https://github.com/distant-viewing/dvt/releases/download/0.0.1/"
) -> str:

    if basename:
        url = os.path.join(basename, url)

    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
        hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)

    return cached_file


def _expand_path(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    return path
