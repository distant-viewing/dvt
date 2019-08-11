# -*- coding: utf-8 -*-
"""Annotator to detect objects.

Detecting objects in an image is an import task for the analysis of both still
and moving images. This modules provides the generic annotator ObjectAnnotator
to which a specific object detector can be inserted. The module also supplies
a direct wrapper to the RetinaNet algorithm, which includes 80 classes of
common objects.

Example:
    Assuming we have an input named "input.mp4", the following example shows
    the a sample usage of the ObjectDetectRetinaNet over two batches of the
    input. The embedding is applied to every 128 frames.

    >>> detector = ObjectDetectRetinaNet()
    >>> fp = FrameProcessor()
    >>> fp.load_annotator(ObjectAnnotator(freq=128, detector=detector))
    >>> fp.process(FrameInput("input.mp4"), max_batch=2)

    Then, collect the output from the annotator and display as a pandas data
    frame.

    >>> fp.collect("obj").todf()
        frame     score      video  left  right  bottom  top   class
    0     128  0.984119  input.mp4   181    466     474   75  person
    1     256  0.934686  input.mp4     0    304     480  136  person
    2     256  0.750618  input.mp4   367    469     306  186  person
    3     256  0.726257  input.mp4   540    708     420  224  person
    4     256  0.683815  input.mp4   112    298     368   49  person
    5     256  0.643473  input.mp4   517    666     355  207  person
    6     256  0.618098  input.mp4   496    618     314  180  person
    7     256  0.503872  input.mp4   143    275     279   11  person
    8     384  0.911823  input.mp4     0    326     479   89  person
    9     384  0.837799  input.mp4   345    708     476   56  person
    10    384  0.511585  input.mp4   141    708     470  183   couch
    11    384  0.500984  input.mp4    52    687     480   91  person

    The detector was run on four frames (0, 128, 256, and 384). Here, it found
    a total of 11 people and one couch.
"""

from warnings import catch_warnings, simplefilter

from numpy import expand_dims

from ..core import FrameAnnotator
from ..utils import _proc_frame_list, _which_frames, process_output_values


class ObjectAnnotator(FrameAnnotator):
    """Annotator for detecting objects in frames or images.

    The annotator will return a list with one DictList item for every frame
    with a detected face.

    Attributes:
        detector: An object with a method called detect that takes an image
            and returns a set of detect faces.
        freq (int): How often to perform the embedding. For example, setting
            the frequency to 2 will embed every other frame in the batch.
        frames (array of ints): An optional list of frames to process. This
            should be a list of integers or a 1D numpy array of integers. If
            set set to something other than None, the freq input is ignored.
    """

    name = "obj"

    def __init__(self, **kwargs):
        self.freq = kwargs.get("freq", 1)
        self.detector = kwargs.get("detector")
        self.frames = _proc_frame_list(kwargs.get("frames", None))

        super().__init__()

    def annotate(self, batch):
        """Annotate the batch of frames with the object detector.

        Args:
            batch (FrameBatch): A batch of images to annotate.

        Returns:
            A list of dictionaries containing the video name, frame, and any
            additional information (i.e., bounding boxes or object names)
            supplied by the detector.
        """

        f_obj = []
        for fnum in _which_frames(batch, self.freq, self.frames):
            img = batch.img[fnum, :, :, :]
            t_obj = self.detector.detect(img)
            for obj in t_obj:
                obj['frame'] = batch.get_frame_names()[fnum]
                f_obj.extend(process_output_values(obj))

        return f_obj


class ObjectDetectRetinaNet:
    """Detect objects using RetinaNet.

    An object detector that locates 80 object types.

    Attributes:
        cutoff (float): A cutoff value for which objects to include in the
            final output. Set to zero (default) to include all object. The
            default is 0.5.
    """

    def __init__(self, cutoff=0.5):
        from keras_retinanet import models
        from keras_retinanet.utils.image import preprocess_image, resize_image
        from keras.utils import get_file

        mloc = get_file(
            "resnet50_coco_best_v2.1.0.h5",
            origin="https://github.com/distant-viewing/dvt/"
            "releases/download/0.0.1/"
            "resnet50_coco_best_v2.1.0.h5",
        )

        self.preprocess_image = preprocess_image
        self.resize_image = resize_image
        self.cutoff = cutoff
        with catch_warnings():
            simplefilter("ignore")
            self.model = models.load_model(mloc, backbone_name="resnet50")

        self.lcodes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
            -1: "unknown",
        }

    def detect(self, img):
        """Detect objects in an image.

        Args:
            img (numpy array): A single image stored as a three-dimensional
                numpy array.

        Returns:
            A list of dictionaries where each dictionary represents a detected
            face. Keys include the bounding box (top, left, bottom, right), a
            confidence score, and the class of the object.
        """

        # process the input image
        img = self.preprocess_image(img)
        img, scale = self.resize_image(img)
        img = expand_dims(img, axis=0)

        # make predictions and scale back to original
        boxes, scores, labels = self.model.predict_on_batch(img)
        boxes /= scale

        # arrange output as list of dictionaries for processing
        objs = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score > self.cutoff:
                objs += [
                    {
                        "top": int(box[1]),
                        "right": int(box[2]),
                        "bottom": int(box[3]),
                        "left": int(box[0]),
                        "score": score,
                        "category": self.lcodes[label],
                    }
                ]

        return objs
