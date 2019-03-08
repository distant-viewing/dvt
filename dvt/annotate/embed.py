# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pandas as pd

from .core import FrameAnnotator
from ..utils import stack_dict_frames


class EmbedAnnotator(FrameAnnotator):
    name = 'embed'

    def __init__(self, embedding, freq=1):
        self.freq = freq
        self.embedding = embedding
        super().__init__()

    def annotate(self, batch):

        # what frames do we annotate?
        fnum = range(0, batch.bsize, self.freq)

        # run the embedding and add video and frame metadata
        obj = self.embedding.embed(batch.img[fnum, :, :, :])
        obj['video'] = [batch.vname] * len(fnum)
        obj['frame'] = np.array(batch.get_frame_nums())[list(fnum)]

        return [obj]


class EmbedFrameKeras(EmbedAnnotator):

    def __init__(self, model, preprocess_input=None, outlayer=None):
        from keras.models import Model

        if outlayer is not None:
            model = Model(inputs=model.input,
                          outputs=model.get_layer(outlayer).output)


        self.input_shape = list(model.input_shape[1:])
        self.model = model
        self.preprocess_input = preprocess_input

    def embed(self, img):
        from skimage.transform import resize

        # resize the images
        rimg = np.zeros([img.shape[0]] + self.input_shape)

        for i in range(img.shape[0]):
            rimg[i, :, :, :] = resize(img[i, :, :, :], self.input_shape,
                                      mode='constant', anti_aliasing=True)

        # process the inputs image
        if self.preprocess_input:
            rimg = self.preprocess_input(rimg)

        # produce embeddings
        embed = self.model.predict(rimg)

        return {'embed': embed}


class EmbedFrameKerasResNet50(EmbedFrameKeras):

    def __init__(self):
        import keras.applications.resnet50

        super().__init__(keras.applications.resnet50.ResNet50(weights='imagenet'),
                         keras.applications.resnet50.preprocess_input,
                         outlayer="avg_pool")
