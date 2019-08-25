# -*- coding: utf-8 -*-
"""Annotator to embedding a set of frame using a neural network.

Given a convolutional neural network trained on a supervised learning task,
embedding into the penultimate layer (or some other internal layer) gives a
useful embedding that can be used similar to word vectors. This module returns
an embedding over a (possible subset) of the frames in an input. The module
can also be used when the embedding corresponds to a concrete supervised task.
"""

from cv2 import resize
from numpy import array, zeros

from ..abstract import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames


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
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set to something other than None, the freq input is ignored.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "embed"

    def __init__(self, **kwargs):
        self.embedding = kwargs.get("embedding")
        self.freq = kwargs.get("freq", 1)
        self.frames = _proc_frame_list(kwargs.get("frames", None))
        super().__init__(**kwargs)

    def annotate(self, batch):
        """Annotate the batch of frames with the embedding annotator.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, and a
            numpy array of the embedding.
        """

        # what frames do we annotate?
        fnum = _which_frames(batch, self.freq, self.frames)
        if not fnum:
            return None

        # run the embedding and add video and frame metadata
        obj = self.embedding.embed(batch.img[fnum, :, :, :])
        obj["frame"] = array(batch.get_frame_names())[list(fnum)]

        return obj


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
            model = Model(
                inputs=model.input, outputs=model.get_layer(outlayer).output
            )

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
        rimg = zeros([img.shape[0]] + list(self.input_shape) + [3])
        for i in range(img.shape[0]):
            rimg[i, :, :, :] = resize(img[i, :, :, :], self.input_shape)

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
