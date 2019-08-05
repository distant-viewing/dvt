# -*- coding: utf-8 -*-
"""Aggregate frame level information to detect people in shots.

The aggregator functions here take face embeddings and tries to predict the
identity of people within each shot.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of a PeopleAggregator over two batches of the input.
    First we need to collect the face embeddings:

    >>> anno = FaceAnnotator(detector=FaceDetectDlib(),
    ...                      embedding=FaceEmbedVgg2(), freq=4)
    >>> fp = FrameProcessor()
    >>> fp.load_annotator(anno)
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    >>> obj = fp.collect_all()

    Then, construct a PeopleAggregator using the first two detected faces as
    default faces:

    >>> pa = PeopleAggregator(face_names=['person 1', 'person 2'],
    ...                       fprint=obj['face']['embed'][[0, 1]])
    >>> pa.aggregate(obj).todf()
                video  frame  top  left  ...  confidence    person person-dist
    0  video-clip.mp4      0  101   451  ...    1.048362  person 1    0.000000
    1  video-clip.mp4      0  105   136  ...    1.014278  person 2    0.000000
    2  video-clip.mp4      4  101   441  ...    1.051648  person 1   20.011372
    3  video-clip.mp4      4  105   136  ...    1.024614  person 2   46.463203
    4  video-clip.mp4      8  101   441  ...    1.066064  person 1   21.747568
    5  video-clip.mp4      8  105   136  ...    1.024315  person 2   49.413498
    6  video-clip.mp4     12  101   441  ...    1.063129  person 1   23.384321
    7  video-clip.mp4     12  105   136  ...    1.020311  person 2   53.846676

    In this part of the clip, every shot contains two characters that are
    sitting next to one another.
"""

from os.path import join, basename, splitext

import numpy as np

from ..annotate.face import FaceAnnotator, FaceDetectMtcnn, FaceEmbedVgg2
from ..utils import DictFrame
from .core import Aggregator
from ..annotate.core import ImageInput, FrameProcessor

class PeopleAggregator(Aggregator):
    """Uses face embeddings to identify the identity of people in the frame.

    You will need to provide baseline faces for the annotator to compare to.
    Note that the annotator returns the nearest faces along with the distance
    to each face.

    Attributes:
        face_names (list): List of names associated with each face in the set
            of predefined faces
        fprint (np.array): A numpy array giving the embedding vectors for the
            predefined faces. Each row should correspond with one face id and
            the number of columns should match the number of columns in your
            embedding.
    """

    def __init__(self, face_names, fprint):

        assert fprint.shape[0] == len(face_names)

        self.face_names = face_names
        self.fprint = fprint

        super().__init__()

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

        # grab the data and create new output
        ops = ldframe["face"]

        output = DictFrame(
            {
                "video": ops["video"].copy(),
                "frame": ops["frame"].copy(),
                "top": ops["top"].copy(),
                "left": ops["left"].copy(),
                "bottom": ops["bottom"].copy(),
                "right": ops["right"].copy(),
                "confidence": ops["confidence"].copy(),
                "person": [""] * len(ops["frame"]),
                "person-dist": [""] * len(ops["frame"]),
            }
        )

        # cycle through frames and detect closest face; let the user filter as
        # needed
        for fid, face in enumerate(ops["embed"]):
            dists = np.linalg.norm(face - self.fprint, axis=1)
            output["person"][fid] = self.face_names[np.argmin(dists)]
            output["person-dist"][fid] = np.min(dists)

        return DictFrame(output)


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
    fpobj = FrameProcessor()
    fpobj.load_annotator(FaceAnnotator(
        detector=FaceDetectMtcnn(),
        embedding=FaceEmbedVgg2()
    ))

    fpobj.process(ImageInput(input_paths=join(dinput, "", "*")))

    faces = fpobj.collect('face')
    names = [splitext(basename(x))[0] for x in faces['frame']]

    return faces['embed'], names
