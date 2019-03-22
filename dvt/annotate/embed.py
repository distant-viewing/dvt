# -*- coding: utf-8 -*-
"""Annotator to embedding a set of frame using a neural network.

Given a convolutional neural network trained on a supervised learning task,
embedding into the penultimate layer (or some other internal layer) gives a
useful embedding that can be used similar to word vectors. This module returns
an embedding over a (possible subset) of the frames in an input. The module
can also be used when the embedding corresponds to a concrete supervised task.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of the EmbedFrameKerasResNet50 over two batches of the
    input. The embedding is applied to every 128 frames.

    >>> embed = EmbedFrameKerasResNet50()
    >>> fp = FrameProcessor()
    >>> fp.load_annotator(EmbedAnnotator(freq=128, embedding=embed))
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)

    Then, collect the output from the annotator and display as a pandas data
    frame.

    >>> fp.collect("embed").todf()
           video  frame   embed-0     ...      embed-2045  embed-2046  embed-2047
    0  input.mp4      0  0.000000     ...        0.000000    0.004976    0.000000
    1  input.mp4    128  0.534926     ...        0.100585    0.379687    0.016144
    2  input.mp4    256  0.259463     ...        0.663053    0.002361    0.168496
    3  input.mp4    384  0.079264     ...        0.351160    0.025871    0.189005

    [4 rows x 2050 columns]

    Notice that there are 4 rows because we embedded once every 128 frames and
    ran two batches, each with 256 frames.
"""

import cv2
import numpy as np

from .core import FrameAnnotator


class EmbedAnnotator(FrameAnnotator):
    """Annotator for embedding frames into an ambient space.

    The annotator will return a numpy array, with one row per processed frame.
    Control how frequently the annotator runs by setting the frequency
    attribute to a number higher than 1. Note that frequency should be able to
    divide the batch size.

    Attributes:
        embedding (EmbedFrameKeras): Object to perform the embedding.
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will embed every other frame in the batch.
    """

    name = "embed"

    def __init__(self, embedding, freq=1):
        self.freq = freq
        self.embedding = embedding
        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the embedding annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, and a
            numpy array of the embedding.
        """

        # what frames do we annotate?
        fnum = range(0, batch.bsize, self.freq)

        # run the embedding and add video and frame metadata
        obj = self.embedding.embed(batch.img[fnum, :, :, :])
        obj["video"] = [batch.vname] * len(fnum)
        obj["frame"] = np.array(batch.get_frame_names())[list(fnum)]

        return [obj]


class EmbedFrameKeras:
    """A generic class for applying an embedding to frames.

    Applies a keras model to a batch of frames. The input of the model is
    assumed to be an image with three channels. The class automatically
    handles resizing the images to the required input shape.

    Attributes:
        model: A keras model to apply to the frames.
        preprocess_input: An optional function to preprocess the images. Set to
            None (the default) to not apply any preprocessing.
        outlayer: Name of the output layer. Set to None (the default) to use
            the final layer predictions as the embedding.
    """

    def __init__(self, model, preprocess_input=None, outlayer=None):
        from keras.models import Model

        if outlayer is not None:
            model = Model(inputs=model.input, outputs=model.get_layer(outlayer).output)

        self.input_shape = (model.input_shape[1], model.input_shape[2])
        self.model = model
        self.preprocess_input = preprocess_input
        super().__init__()

    def embed(self, img):
        """Embed a batch of images.

        Args:
            img: A four dimensional numpy array to embed using the keras model.

        Returns:
            A numpy array, with a first dimension matching the first dimension
            of the input image.
        """

        # resize the images
        rimg = np.zeros([img.shape[0]] + list(self.input_shape) + [3])
        for i in range(img.shape[0]):
            rimg[i, :, :, :] = cv2.resize(img[i, :, :, :], self.input_shape)

        # process the inputs image
        if self.preprocess_input:
            rimg = self.preprocess_input(rimg)

        # produce embeddings
        embed = self.model.predict(rimg)

        return {"embed": embed}


class EmbedFrameKerasResNet50(EmbedFrameKeras):
    """Example embedding using ResNet50.

    Provides an example of how to use an embedding annotator and provides
    easy access to one of the most popular models for computing image
    similarity metrics in an embedding space. See the (very minimal) source
    code for how to extend this function to other pre-built keras models.

    Attributes:
        model: The ResNet-50 model, tuned to produce the penultimate layer as
            an output.
        preprocess_input: Default processing function for an image provided as
            an array in RGB format.
    """

    def __init__(self):
        import keras.applications.resnet50

        model = keras.applications.resnet50.ResNet50(weights="imagenet")
        ppobj = keras.applications.resnet50.preprocess_input

        super().__init__(model, ppobj, outlayer="avg_pool")
