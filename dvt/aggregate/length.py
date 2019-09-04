# -*- coding: utf-8 -*-
"""Aggregate frame level information to estimate shot length.

The aggregator functions here takes detected faces and objects to estimate the
shot length. It also provides aggregated information about the detected faces
and objects for each frame.
"""

from numpy import argmax, array, max as npmax, nonzero

from ..abstract import Aggregator
from ..utils import _check_data_exists


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
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "length"

    def __init__(self, **kwargs):

        self.min_obj_score = kwargs.get("min_obj_score", 0.7)
        self.min_face_score = kwargs.get("min_face_score", 0.7)
        self.max_person_dist = kwargs.get("max_person_dist", 100)
        self.shot_sizes = array(
            kwargs.get("shot_sizes", [0, 0.05, 0.15, 0.2, 0.3, 0.5, 0.7])
        )
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

        super().__init__(**kwargs)

    def aggregate(self, ldframe, **kwargs):
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
        # make sure annotators have been run
        _check_data_exists(ldframe, ["face", "obj", "meta"])

        # grab the data sets
        face = ldframe["face"]
        objs = ldframe["obj"]

        # get heights; different depending on input type (video or images)
        if "height" in ldframe["meta"].keys():
            face = face.assign(
                img_height=ldframe["meta"].height.values[0],
                img_width=ldframe["meta"].width.values[0]
            )
            objs = objs.assign(
                img_height=ldframe["meta"].height.values[0],
                img_width=ldframe["meta"].width.values[0]
            )
        else:
            face = face.merge(ldframe["img"], on="frame")
            objs = objs.merge(ldframe["img"], on="frame")

        # compute data using vectorized numpy arrays, where possible
        face_height = (
            face.bottom.values - face.top.values
        ) / face.img_height.values
        objs_height = (
            objs.bottom.values - objs.top.values
        ) / objs.img_height.values
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
            face_ids = nonzero(
                (face.frame.values == frame) &
                (face.confidence.values > self.min_face_score)
            )[0]
            face_person_ids = nonzero(
                (face.frame.values == frame)
                & (face.confidence.values > self.min_face_score)
                & (face.person_dist.values < self.max_person_dist)
            )[0]
            objs_ids = nonzero(
                (objs.frame.values == frame)
                & (objs.category.values == "person")
                & (objs.score.values > self.min_obj_score)
            )[0]
            aobj_ids = nonzero(objs.frame.values == frame)[0]

            output["num_faces"][fnum] = len(face_ids)
            output["num_people"][fnum] = len(objs_ids)
            output["largest_face"][fnum] = npmax(
                face_height[face_ids], initial=0
            )
            output["largest_body"][fnum] = npmax(
                objs_height[objs_ids], initial=0
            )
            output["objects"][fnum] = "; ".join(
                sorted(set(objs.category.values[aobj_ids]))
            )
            output["people"][fnum] = "; ".join(
                sorted(set(face.person.values[face_person_ids]))
            )

            output["shot_length"][fnum] = self.shot_names[
                argmax(self.shot_sizes >= output["largest_face"][fnum])
            ]

        return output
