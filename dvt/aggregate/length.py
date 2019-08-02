# -*- coding: utf-8 -*-
"""Aggregate frame level information to estimate shot length.

The aggregator functions here takes detected faces and objects to estimate the
shot length. It also provides aggregated information about the detected faces
and objects for each frame.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of a ShotLengthAggregator over two batches of the input.
    First we need to collect the face embeddings:

    >>> fp = FrameProcessor()
    >>> fp.load_annotator(MetaAnnotator())
    >>> fp.load_annotator(
    ...     FaceAnnotator(detector=FaceDetectDlib(), freq=128)
    ... )
    >>> fp.load_annotator(
    ...     ObjectAnnotator(detector=ObjectDetectRetinaNet(), freq=128)
    ... )
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    >>> obj = fp.collect_all()

    Then, construct a ShotLengthAggregator:

    >>> sla = ShotLengthAggregator()
    >>> sla.aggregate(obj).todf()

       frame  num_faces  num_people  ...  largest_body  \
    0      0          2           2  ...      0.870833
    1    128          4           5  ...      0.668750
    2    256          2           1  ...      0.814583

       shot_length       objects
    0        5-MCU  oven; person
    1        3-MLS        person
    2        5-MCU  oven; person

    [3 rows x 7 columns]

    In this example, we see that the first and last frames contains two people
    in a  medium close-up (MCU) and the middle frame contains five people
    (there are actually 6, but one person is too obscured) in a medium long
    shot (MLS).
"""

from ..utils import DictFrame
from .core import Aggregator

import numpy as np


class ShotLengthAggregator(Aggregator):
    """Uses detected faces and objects to estimate shot length.

    You can change the cut-offs and names of the face types.

    Attributes:
        min_obj_score (float): minimum confidence score to include a detected
            object in the computation
        min_face_score (float): minimum confidence score to include a detected
            face in the computation
        shot_names (list): a list of shot names, from the longest shot to the
            tightest. Set to None to use the default settings.
        shot_sizes (list): as list of shot size cut-offs given as a proportion
            (vertical) of face size to the entire shot. Should be an increasing
            list starting at zero and the same length as shot_names. Set to
            None to use the default settings.
    """

    def __init__(
        self,
        min_obj_score=0.7,
        min_face_score=0.7,
        shot_names=None,
        shot_sizes=None,
    ):

        if shot_names is None:
            shot_names = [
                "1-VLS",
                "2-LS",
                "3-MLS",
                "4-MS",
                "5-MCU",
                "6-CU",
                "7-BCU",
            ]

        if shot_sizes is None:
            shot_sizes = [0, 0.05, 0.15, 0.2, 0.3, 0.5, 0.7]

        assert len(shot_sizes) == len(shot_names)

        self.min_obj_score = min_obj_score
        self.min_face_score = min_face_score
        self.shot_sizes = np.array(shot_sizes)
        self.shot_names = shot_names

        super().__init__()

    def aggregate(self, ldframe, frames=None):
        """Determine shot lengths using detected faces and objects.

        Args:
            ldframe (dict): A dictionary of DictFrames from a FrameAnnotator.
                Must contain an entry with the keys 'meta', 'face' and 'object',
                which are used in the annotation.
            frames (list): An optional list of frames. Otherwise, will annotate
                any frame with a detected face or object.

        Returns:
            A dictionary frame giving the detected people, with one row per
            frame in the original input.
        """

        # grab the data sets
        face = ldframe["face"]
        objs = ldframe["object"]
        meta_height = ldframe["meta"]["height"]

        # compute data using vectorized numpy arrays, where possible
        face_frames = np.array(face["frame"])
        objs_frames = np.array(objs["frame"])
        face_height = (
            np.array(face["bottom"]) - np.array(face["top"])
        ) / meta_height
        objs_height = (
            np.array(objs["bottom"]) - np.array(objs["top"])
        ) / meta_height
        face_scores = np.array(face["confidence"])
        objs_scores = np.array(objs["score"])
        objs_object = np.array(objs["class"])

        # what frames to include?
        if frames is None:
            frames = set(face_frames).union(objs_frames)
        frames = sorted(frames)

        # create the output
        output = DictFrame(
            {
                "frame": frames,
                "num_faces": [0] * len(frames),
                "num_people": [0] * len(frames),
                "largest_face": [0.0] * len(frames),
                "largest_body": [0.0] * len(frames),
                "shot_length": [""] * len(frames),
                "objects": [""] * len(frames),
            }
        )

        for fnum, frame in enumerate(frames):
            face_ids = np.nonzero(
                (face_frames == frame) & (face_scores > self.min_face_score)
            )[0]
            objs_ids = np.nonzero(
                (objs_frames == frame)
                & (objs_object == "person")
                & (objs_scores > self.min_obj_score)
            )[0]
            aobj_ids = np.nonzero(
                (objs_frames == frame) & (objs_scores > self.min_obj_score)
            )[0]

            output["num_faces"][fnum] = len(face_ids)
            output["num_people"][fnum] = len(objs_ids)
            output["largest_face"][fnum] = np.max(
                face_height[face_ids], initial=0
            )
            output["largest_body"][fnum] = np.max(
                objs_height[objs_ids], initial=0
            )
            output["objects"][fnum] = "; ".join(
                sorted(set(objs_object[aobj_ids]))
            )

            output["shot_length"][fnum] = self.shot_names[
                np.argmax(self.shot_sizes >= output["largest_face"][fnum])
            ]

        return DictFrame(output)
