# -*- coding: utf-8 -*-
"""General utilities used across the toolkit
"""

import datetime
import enum
import os

from six.moves.urllib.request import urlretrieve
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen

AnnotatorStatus = enum.Enum('AnnotatorStatus', 'NEXT_ANNOTATOR NEXT_FRAME')


def iso8601():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def model_data_path():
    return os.path.join(os.path.expanduser('~'), '.dvt', 'models')


def get_file(fname, origin):
    """Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded into the
    location `~/.dvt/models`
    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
    # Returns
        Path to the downloaded file
    """
    datadir = model_data_path()
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fpath = os.path.join(datadir, fname)

    download = False
    if not os.path.exists(fpath):
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    return fpath
