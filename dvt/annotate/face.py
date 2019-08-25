# -*- coding: utf-8 -*-
"""Annotators to detect and identify faces.

Identifying individuals in an image generally requires two distinct steps. The
first is detecting bounding boxes for faces in the image and the second is
identifying the faces themselves. Currently the most common method for doing
the second step is to project a detected face into a high-dimensional space
designed such that different images of the same person will be close together
and images of different people will be farther apart. This module is built
around this paradigm, allowing for the specification of custom detectors and
embeddings into the model.
"""

from importlib import import_module

from numpy import float32, expand_dims

from ..abstract import FrameAnnotator
from ..utils import (
    process_output_values,
    sub_image,
    _proc_frame_list,
    _which_frames,
    _trim_bbox
)


class FaceAnnotator(FrameAnnotator):
    """Annotator for detecting faces and embedding them as a face vector.

    The annotator will return a list with one DictList item for every frame
    with a detected face. If an embedding is supplied, the DictList items
    will contain a numpy array with the face embeddings.

    Attributes:
        detector: An object with a method called detect that takes an image
            and returns a set of detect faces. Can be set to None (default) as
            a pass-through option for testing.
        embedding: An object with a method embed that takes an image along with
            a set of bounding boxed and returns embeddings of the faces as a
            numpy array. Set to None (default) to only run the face detector.
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will embed every other frame in the batch.
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "face"

    def __init__(self, **kwargs):
        self.freq = kwargs.get("freq", 1)
        self.detector = kwargs.get("detector")
        self.embedding = kwargs.get("embedding", None)
        self.frames = _proc_frame_list(kwargs.get("frames", None))

        super().__init__(**kwargs)

    def annotate(self, batch):
        """Annotate the batch of frames with the face annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, bounding
            box coordinates (top, bottom, left, and right). If an embedding is
            included, the result will also contain a numpy array of the
            embedding for each face.
        """

        f_faces = []
        for fnum in _which_frames(batch, self.freq, self.frames):
            img = batch.img[fnum, :, :, :]
            t_faces = self.detector.detect(img)
            for face in t_faces:
                face['frame'] = batch.get_frame_names()[fnum]
                if self.embedding is not None:
                    face["embed"] = [self.embedding.embed(img, face)]
                f_faces.extend(process_output_values(face))

        return f_faces


class FaceDetectMtcnn:
    """Detect faces using the Multi-task Cascaded CNN model.

    Attributes:
        cutoff (float): A cutoff value for which faces to include in the final
            output. Set to zero (default) to include all faces.
    """

    def __init__(self, cutoff=0):
        self.mtcnn = import_module("mtcnn.mtcnn")
        self.cutoff = cutoff
        self._mt = self.mtcnn.MTCNN(min_face_size=20)

    def detect(self, img):
        """Detect faces in an image.

        Args:
            img (numpy array): A single image stored as a three-dimensional
                numpy array.

        Returns:
            A list of dictionaries where each dictionary represents a detected
            face. Keys include the bounding box (top, left, bottom, right) as
            well as a confidence score.
        """
        dets = self._mt.detect_faces(img)

        faces = []
        for det in dets:
            if det["confidence"] >= self.cutoff:
                bbox = _trim_bbox(
                    (
                        det["box"][1],
                        det["box"][0] + det["box"][2],
                        det["box"][1] + det["box"][3],
                        det["box"][0],
                    ),
                    img.shape,
                )
                faces += [
                    {
                        "top": bbox[0],
                        "right": bbox[1],
                        "bottom": bbox[2],
                        "left": bbox[3],
                        "confidence": [det["confidence"]],
                    }
                ]

        return faces


class FaceEmbedVgg2:
    """Embed faces using the VGGFace2 model.

    A face embedding with state-of-the-art results, particularly suitable when
    there are small or non-forward-facing examples in the dataset.
    """

    def __init__(self):
        from keras.models import load_model
        from keras.utils import get_file
        from keras import backend as K

        mloc = get_file(
            "vggface2-resnet50.h5",
            origin="https://github.com/distant-viewing/dvt/"
            "releases/download/0.0.1/"
            "vggface2-resnet50.h5",
        )
        self._model = load_model(mloc)
        self._iformat = K.image_data_format()

    def embed(self, img, face):
        """Embed detected faces in an image.

        Args:
            img (numpy array): A single image stored as a three-dimensional
                numpy array.
            faces (dict): Location of detected faces in the image.

        Returns:
            A numpy array with one row for each input face and 2048 columns.
        """

        iscale = self._proc_image(
            sub_image(
                img=img,
                top=face["top"],
                right=face["right"],
                bottom=face["bottom"],
                left=face["left"],
                fct=1.3,
                output_shape=(224, 224),
            )
        )

        return self._model.predict(iscale)[0, 0, 0, :]

    def _proc_image(self, iscale):
        iscale = float32(iscale)
        iscale = expand_dims(iscale, axis=0)

        if self._iformat == "channels_first":    # pragma: no cover
            iscale = iscale[:, ::-1, ...]
            iscale[:, 0, :, :] -= 91.4953
            iscale[:, 1, :, :] -= 103.8827
            iscale[:, 2, :, :] -= 131.0912
        else:
            iscale = iscale[..., ::-1]
            iscale[..., 0] -= 91.4953
            iscale[..., 1] -= 103.8827
            iscale[..., 2] -= 131.0912

        return iscale
