# -*- coding: utf-8 -*-
"""Core objects for aggregating data from frame and audio processor pipelines.

The objects here can be further extended with logic for specific aggregation
tasks. See the implemented tasks in this model for examples.
"""


class Aggregator:
    """Base class for aggregating the output from a pipeline of processors.

    Attributes:
        name (str): A description of the aggregator.
    """

    name = "base"

    def __init__(self):
        """Create a new empty Aggregator.
        """
        pass

    def aggregate(self, ldframe):
        """Aggregate annotations.

        Args:
            ldframe (dict): A dictionary with values equal to DictFrames.

        Returns:
            While not strictly enforced, subclasses should return a DictFrame
            or dictionary of DictFrames from the aggregate method.
        """
        pass
