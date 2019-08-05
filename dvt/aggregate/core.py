# -*- coding: utf-8 -*-
"""Core objects for aggregating data from frame and audio processor pipelines.

The objects here can be further extended with logic for specific aggregation
tasks. See the implemented tasks in this model for examples.
"""

from abc import ABC, abstractmethod

class Aggregator(ABC):    # pragma nocov
    """Base class for aggregating the output from a pipeline of processors.

    Attributes:
        name (str): A description of the aggregator.
    """

    name = "base"

    @abstractmethod
    def __init__(self):
        """Create a new empty Aggregator.
        """
        return

    @abstractmethod
    def aggregate(self, ldframe, **kwargs):
        """Aggregate annotations.

        Args:
            ldframe (dict): A dictionary with values equal to DictFrames.

        Returns:
            While not strictly enforced, subclasses should return a DictFrame
            or dictionary of DictFrames from the aggregate method.
        """
        return
