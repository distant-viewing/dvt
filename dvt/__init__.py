# -*- coding: utf-8 -*-
"""Distant Viewing Toolkit for the Analysis of Visual Culture

The Distant TV Toolkit is a Python package designed to facilitate the
computational analysis of visual culture. It contains low-level architecture
for applying state-of-the-art computer vision algorithms to still and moving
images. The higher-level functionality of the toolkit allows users to quickly
extract semantic metadata from digitized collections. Extracted information
can be visualized for search and discovery or aggregated and analyzed to find
patterns across a corpus.
"""

from __future__ import absolute_import

from . import abstract
from . import aggregate
from . import annotate
from . import core
from . import utils

__version__ = "0.3.0"
