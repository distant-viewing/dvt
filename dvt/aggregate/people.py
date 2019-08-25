# -*- coding: utf-8 -*-
"""Aggregate frame level information to detect people in shots.

The aggregator functions here take face embeddings and tries to predict the
identity of people within each shot.
"""

from os.path import join, basename, splitext

from numpy import argmin, min as npmin, vstack
from numpy.linalg import norm

from ..abstract import Aggregator
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..core import DataExtraction, ImageInput
from ..utils import _check_data_exists


class PeopleAggregator(Aggregator):
    """Uses face embeddings to identify the identity of people in the frame.

    You will need to provide baseline faces for the annotator to compare to.
    Note that the annotator returns the nearest faces along with the distance
    to each face.

    Attributes:
        face_names (list): List of names associated with each face in the set
            of predefined faces
        fprint (numpy array): A numpy array giving the embedding vectors for
            the predefined faces. Each row should correspond with one face id
            and the number of columns should match the number of columns in
            your embedding.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "people"

    def __init__(self, **kwargs):

        self.face_names = kwargs.get("face_names")
        self.fprint = kwargs.get("fprint")

        assert self.fprint.shape[0] == len(self.face_names)

        super().__init__(**kwargs)

    def aggregate(self, ldframe, **kwargs):
        """Aggregate faces.

        Args:
            ldframe (dict): A dictionary of DictFrames from a FrameAnnotator.
                Must contain an entry with the key 'face', which is used in the
                annotation.

        Returns:
            A dictionary frame giving the detected people, with one row per
            detected face.
        """
        # make sure annotators have been run
        _check_data_exists(ldframe, ["face"])

        # grab the data and create new output
        ops = ldframe["face"]

        output = {
                "frame": ops.frame.values.copy(),
                "top": ops.top.values.copy(),
                "left": ops.left.values.copy(),
                "bottom": ops.bottom.values.copy(),
                "right": ops.right.values.copy(),
                "confidence": ops.confidence.values.copy(),
                "person": [""] * len(ops.frame.values),
                "person-dist": [""] * len(ops.frame.values),
        }

        # cycle through frames and detect closest face; let the user filter as
        # needed
        for fid, face in enumerate(ops.embed.values):
            dists = norm(face - self.fprint, axis=1)
            output["person"][fid] = self.face_names[argmin(dists)]
            output["person-dist"][fid] = npmin(dists)

        return output


def make_fprint_from_images(dinput):
    """Create face fingerprints from a directory of faces.

    This function takes as an input a directory containing image files, with
    each image given the name of a person or character. The function returns
    the 'fingerprints' (sterotypical embedding) of the faces in a format that
    can be passed to the PeopleAggregator.

    Args:
        face_names (list): List of names associated with each face in the set
            of predefined faces

    Returns:
        A tuple giving the number array of embedding vectors and a list of the
        names of the people in the images.
    """
    dextra = DataExtraction(ImageInput(input_paths=join(dinput, "", "*")))
    dextra.run_annotators([FaceAnnotator(
        detector=FaceDetectMtcnn(),
        embedding=FaceEmbedVgg2()
    )])

    faces = dextra.get_data()['face']
    face_names = [
        splitext(basename(x))[0] for x in dextra.get_data()["meta"]["paths"]
    ]

    return vstack(faces.embed), face_names
