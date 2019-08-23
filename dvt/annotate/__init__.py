# -*- coding: utf-8 -*-
"""Annotators for processing the visual input.

Annotators work with small subsets of the input visual materials and produce
metadata, usually at the frame-level.
"""

from __future__ import absolute_import

from . import diff
from . import embed
from . import png
from . import cielab
from . import opticalflow
from . import hofm
from . import img
from . import obj
from . import face
