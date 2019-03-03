# -*- coding: utf-8 -*-

import numpy as np

from .core import FrameAnnotator
from ..utils import combine_list_dicts, sub_image, _trim_bounds


class FaceAnnotator(FrameAnnotator):
    name = 'face'

    def __init__(self, detector=None, embedding=None, freq=1):
        self.freq = freq
        self.detector = detector
        self.embedding = embedding
        super().__init__()

    def annotate(self, batch):

        f_faces = []
        for fnum in range(0, batch.bsize, self.freq):
            img = batch.img[fnum, :, :, :]
            t_faces = combine_list_dicts(self.detector.detect(img))
            if t_faces:
                frame = batch.get_frame_nums()[fnum]
                t_faces['video'] = [batch.vname] * len(t_faces['top'])
                t_faces['frame'] = [frame] * len(t_faces['top'])
                if self.embedding is not None:
                    t_faces['embed'] = self.embedding.embed(img, t_faces)
                f_faces.append(t_faces)

        return f_faces


class FaceDetectDlib():

    def __init__(self, cutoff=0):
        import dlib
        import face_recognition_models as frm

        mloc = frm.cnn_face_detector_model_location()
        self.cfd = dlib.cnn_face_detection_model_v1(mloc)
        self.cutoff = cutoff

    def detect(self, img):
        dets = self.cfd(img, 1)

        faces = []
        for det in dets:
            if det.confidence >= self.cutoff:
                bbox = _trim_bounds((det.rect.top(), det.rect.right(),
                                     det.rect.bottom(), det.rect.left()),
                                     img.shape)
                faces += [{'top': bbox[0], 'right': bbox[1],
                           'bottom': bbox[2], 'left': bbox[3],
                           'confidence': det.confidence}]

        return faces


class FaceEmbedDlib():

    def __init__(self):
        import dlib
        import face_recognition as fr
        self.encode = fr.face_encodings

    def embed(self, img, faces):

        embed_mat = []
        for ind in range(len(faces['frame'])):
            bbox = (faces['top'][ind], faces['bottom'][ind],
                    faces['left'][ind], faces['right'][ind])
            embed_mat.append(self.encode(img, [bbox])[0])

        return np.vstack(embed_mat)


class FaceEmbedVgg2():

    def __init__(self, cutoff=0.8):
        from keras_vggface.vggface import VGGFace
        from keras_vggface.utils import preprocess_input

        self.model = VGGFace(model='resnet50', include_top=False)
        self.pi = preprocess_input
        self.cutoff = preprocess_input

    def embed(self, img, faces):

        embed_mat = []
        for ind in range(len(faces['frame'])):
            iscale = sub_image(img=img,
                               top=faces['top'][ind],
                               right=faces['right'][ind],
                               bottom=faces['bottom'][ind],
                               left=faces['left'][ind],
                               fct=1.3,
                               output_shape=(224, 224))
            iscale = np.expand_dims(iscale, axis=0)
            iscale = self.pi(iscale, version=2)
            embed = self.model.predict(iscale)
            embed_mat.append(embed[0, 0, 0, :])

        return np.vstack(embed_mat)
