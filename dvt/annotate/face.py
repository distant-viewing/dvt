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

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the sample usage of FaceAnnotator over two batches of the input. A CNN
    model from dlib is used to detect the faces and the VGGFace2 algorithm is
    used to embed the faces into a 2048-dimensional space. The algorithm is
    applied to every 128 frames.

    >>> detector = FaceDetectDlib()
    >>> embedding = FaceEmbedVgg2()
    >>>
    >>> fp = FrameProcessor()
    >>> fp.load_annotator(FaceAnnotator(freq=128, detector=detector,
    ...                   embedding=embedding))
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)
    INFO:root:processed batch 00:00:00,000 to 00:00:17,083 with annotator: 'face'
    INFO:root:processed batch 00:00:17,083 to 00:00:25,625 with annotator: 'face'

    Then, collect the output from the annotator and display as a pandas data
    frame.

    >>> fp.collect("face").todf()
       frame  bottom      video     ...      embed-2045  embed-2046  embed-2047
    0    128     171  input.mp4     ...        7.355998    0.000000    0.000000
    1    384     209  input.mp4     ...        0.128695    0.640979    0.207890
    2    384     220  input.mp4     ...        0.187535    0.754207    0.705644

    [3 rows x 2055 columns]

    The detector was run on four frames (0, 128, 256, and 384). It found one
    face at frame 128 and two faces at frame 384.
"""

import dlib
import numpy as np
from keras.utils import get_file
from keras import backend as K

from .core import FrameAnnotator
from ..utils import stack_dict_frames, sub_image, _trim_bbox


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
    """

    name = "face"

    def __init__(self, detector, embedding=None, freq=1):
        self.freq = freq
        self.detector = detector
        self.embedding = embedding
        super().__init__()

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
        for fnum in range(0, batch.bsize, self.freq):
            img = batch.img[fnum, :, :, :]
            t_faces = stack_dict_frames(self.detector.detect(img))
            if t_faces:
                frame = batch.get_frame_names()[fnum]
                t_faces["video"] = [batch.vname] * len(t_faces["top"])
                t_faces["frame"] = [frame] * len(t_faces["top"])
                if self.embedding is not None:
                    t_faces["embed"] = self.embedding.embed(img, t_faces)
                f_faces.append(t_faces)

        return f_faces


class FaceDetectDlib:
    """Detect faces using the dlib CNN model.

    A face detector that balances speed and accuracy.

    Attributes:
        cutoff (float): A cutoff value for which faces to include in the final
            output. Set to zero (default) to include all faces.
    """

    def __init__(self, cutoff=0):
        mloc = get_file(
            "mmod_human_face_detector.dat",
            origin="https://github.com/distant-viewing/dvt/"
            "releases/download/0.0.1/"
            "mmod_human_face_detector.dat",
        )
        self.cutoff = cutoff
        self._cfd = dlib.cnn_face_detection_model_v1(mloc)

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
        dets = self._cfd(img, 1)

        faces = []
        for det in dets:
            if det.confidence >= self.cutoff:
                bbox = _trim_bbox(
                    (
                        det.rect.top(),
                        det.rect.right(),
                        det.rect.bottom(),
                        det.rect.left(),
                    ),
                    img.shape,
                )
                faces += [
                    {
                        "top": bbox[0],
                        "right": bbox[1],
                        "bottom": bbox[2],
                        "left": bbox[3],
                        "confidence": det.confidence,
                    }
                ]

        return faces


class FaceEmbedDlib:
    """Embed faces using the dlib CNN model.

    A face embedding that balances ease of use with accuracy.
    """

    def __init__(self):
        mloc = get_file(
            "dlib_face_recognition_resnet_model_v1.dat",
            origin="https://github.com/distant-viewing/dvt/"
            "releases/download/0.0.1/"
            "dlib_face_recognition_resnet_model_v1.dat",
        )
        self.encode = dlib.face_recognition_model_v1(mloc)

        mloc = get_file(
            "shape_predictor_5_face_landmarks.dat",
            origin="https://github.com/distant-viewing/dvt/"
            "releases/download/0.0.1/"
            "shape_predictor_5_face_landmarks.dat",
        )
        self.pose = dlib.shape_predictor(mloc)

    def embed(self, img, faces):
        """Embed detected faces in an image.

        Args:
            img (numpy array): A single image stored as a three-dimensional
                numpy array.
            faces (DictList): A DictList giving the location of detected faces
                in the image.

        Returns:
            A numpy array with one row for each input face and 128 columns.
        """
        embed_mat = []
        for ind in range(len(faces["top"])):
            # detect pose
            rec = dlib.rectangle(
                left=faces["left"][ind],
                top=faces["top"][ind],
                right=faces["right"][ind],
                bottom=faces["bottom"][ind],
            )
            rls = self.pose(img, rec)

            # compute the embedding and add to our list of output
            emat = self.encode.compute_face_descriptor(img, rls, 1)
            embed_mat.append(np.array(emat))

        return np.vstack(embed_mat)


class FaceEmbedVgg2:
    """Embed faces using the VGGFace2 model.

    A face embedding with state-of-the-art results, particularly suitable when
    there are small or non-forward-facing examples in the dataset.
    """

    def __init__(self):
        from keras.models import load_model

        mloc = get_file(
            "vggface2-resnet50.h5",
            origin="https://github.com/distant-viewing/dvt/"
            "releases/download/0.0.1/"
            "vggface2-resnet50.h5",
        )
        self._model = load_model(mloc)
        self._iformat = K.image_data_format()

    def embed(self, img, faces):
        """Embed detected faces in an image.

        Args:
            img (numpy array): A single image stored as a three-dimensional
                numpy array.
            faces (DictList): A DictList giving the location of detected faces
                in the image.

        Returns:
            A numpy array with one row for each input face and 2048 columns.
        """

        embed_mat = []
        for ind in range(len(faces["top"])):
            iscale = self._proc_image(
                sub_image(
                    img=img,
                    top=faces["top"][ind],
                    right=faces["right"][ind],
                    bottom=faces["bottom"][ind],
                    left=faces["left"][ind],
                    fct=1.3,
                    output_shape=(224, 224),
                )
            )

            embed = self._model.predict(iscale)
            embed_mat.append(embed[0, 0, 0, :])

        return np.vstack(embed_mat)

    def _proc_image(self, iscale):
        iscale = np.float32(iscale)
        iscale = np.expand_dims(iscale, axis=0)

        if self._iformat == "channels_first":
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
