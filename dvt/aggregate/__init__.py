# -*- coding: utf-8 -*-
"""Aggregators collected extracted features and produce high-level metadata.

While annotators work individually to process the input digitized materials,
aggregators take extracted metadata and produce higher order semantic
information. Unlike the annotators, aggregators may make use of the metadata
from the entire media and from multiple annotators at once.
"""

from __future__ import absolute_import

from . import audio
from . import cut
from . import length
from . import display
from . import people
