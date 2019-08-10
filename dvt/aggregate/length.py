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
       frame  num_faces  num_people  ...  shot_length  objects
    0      0          2           2  ...        5-MCU  oven; person
    1    128          4           5  ...        3-MLS        person
    2    256          2           1  ...        5-MCU  oven; person

    [3 rows x 7 columns]

    In this example, we see that the first and last frames contains two people
    in a  medium close-up (MCU) and the middle frame contains five people
    (there are actually 6, but one person is too obscured) in a medium long
    shot (MLS).
"""

import numpy as np

from ..core import Aggregator


class ShotLengthAggregator(Aggregator):
    """Uses detected faces and objects to estimate shot length.

    You can change the cut-offs and names of the face types.

    Attributes:
        min_obj_score (float): minimum confidence score to include a detected
            object in the computation
        min_face_score (float): minimum confidence score to include a detected
            face in the computation
        max_person_dist (float): maximum distance to a face to categorize as
            a known person.
        shot_names (list): a list of shot names, from the longest shot to the
            tightest. Set to None to use the default settings.
        shot_sizes (list): as list of shot size cut-offs given as a proportion
            (vertical) of face size to the entire shot. Should be an increasing
            list starting at zero and the same length as shot_names. Set to
            None to use the default settings.
    """

    name = "length"

    def __init__(self, **kwargs):

        self.min_obj_score = kwargs.get("min_obj_score", 0.7)
        self.min_face_score = kwargs.get("min_face_score", 0.7)
        self.max_person_dist = kwargs.get("max_person_dist", 100)
        self.shot_sizes = np.array(kwargs.get("shot_sizes",
            [0, 0.05, 0.15, 0.2, 0.3, 0.5, 0.7]
        ))
        self.shot_names = kwargs.get("shot_names", [
            "1-VLS",
            "2-LS",
            "3-MLS",
            "4-MS",
            "5-MCU",
            "6-CU",
            "7-BCU",
        ])
        self.frames = kwargs.get('frames', None)

        assert len(self.shot_sizes) == len(self.shot_names)

    def aggregate(self, ldframe):
        """Determine shot lengths using detected faces and objects.

        Args:
            ldframe (dict): A dictionary of DictFrames from a FrameAnnotator.
                Must contain an entry with the keys 'meta', 'face' and
                'obj', which are used in the annotation.
            frames (list): An optional list of frames. Otherwise, will
                annotate any frame with a detected face or object.

        Returns:
            A dictionary frame giving the detected people, with one row per
            frame in the original input.
        """

        # grab the data sets
        face = ldframe["face"]
        objs = ldframe["obj"]
        meta_height = ldframe["meta"].height.values[0]

        # compute data using vectorized numpy arrays, where possible
        face_height = (
            face.bottom.values - face.top.values
        ) / meta_height
        objs_height = (
            objs.bottom.values - objs.top.values
        ) / meta_height
        if "person" not in face:
            face['person'] = ""
        if "person_dist" not in face:
            face['person_dist'] = 0

        # what frames to include?
        if self.frames is None:
            self.frames = set(face.frame.values).union(objs.frame.values)
        frames = sorted(self.frames)

        # create the output
        output = {
                "frame": frames,
                "num_faces": [0] * len(frames),
                "num_people": [0] * len(frames),
                "largest_face": [0.0] * len(frames),
                "largest_body": [0.0] * len(frames),
                "shot_length": [""] * len(frames),
                "objects": [""] * len(frames),
                "people": [""] * len(frames)
        }

        for fnum, frame in enumerate(frames):
            face_ids = np.nonzero(
                (face.frame.values == frame) &
                (face.confidence.values > self.min_face_score)
            )[0]
            face_person_ids = np.nonzero(
                (face.frame.values == frame)
                & (face.confidence.values > self.min_face_score)
                & (face.person_dist.values < self.max_person_dist)
            )[0]
            objs_ids = np.nonzero(
                (objs.frame.values == frame)
                & (objs.category.values == "person")
                & (objs.score.values > self.min_obj_score)
            )[0]
            aobj_ids = np.nonzero(objs.frame.values == frame)[0]

            output["num_faces"][fnum] = len(face_ids)
            output["num_people"][fnum] = len(objs_ids)
            output["largest_face"][fnum] = np.max(
                face_height[face_ids], initial=0
            )
            output["largest_body"][fnum] = np.max(
                objs_height[objs_ids], initial=0
            )
            output["objects"][fnum] = "; ".join(
                sorted(set(objs.category.values[aobj_ids]))
            )
            output["people"][fnum] = "; ".join(
                sorted(set(face.person.values[face_ids]))
            )

            output["shot_length"][fnum] = self.shot_names[
                np.argmax(self.shot_sizes >= output["largest_face"][fnum])
            ]

        return output
