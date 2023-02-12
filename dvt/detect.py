# -*- coding: utf-8 -*-
"""Annotators for detecting image segmentation.
"""

import numpy as np
import torch
from torchvision import transforms as T
from torch.hub import load

from .utils import _download_file

CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class AnnoDetect:
    """Detect objects and people within an input image.

    Uses the FasterRCNN, ResNet50, FPN, V2 weights trained on MS COCO, as
    provided by the PyTorch model zoo.

    Attributes:
        model_path (str): location of the saved pytorch weights; set to
            None to download from the DVT repository.
    """
    def __init__(self, model_path=None):
        if not model_path:
            model_path = _download_file("dvt_detect.pt", version="1.0.1")
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def run(self, img: np.ndarray, visualize=False) -> dict:
        """Run the annotator.

        Args:
            img (np.ndarray): image file, as returned by dvt.load_image,
                or dvt.yield_video (for frame from a video file).
            visualize (bool): should a visualization of the output be
                returned as well? Default is False.

        Returns:
            A dictionary giving the annotation.
        """
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img / 255).unsqueeze(0)

        boxes, labels, scores = self.model(img)

        # clean up the labels
        labels = labels.detach().numpy()
        labels = [CATEGORIES[x] for x in labels]

        # unpack bounding boxes
        boxes = boxes.detach().numpy()
        vec_x = np.int32(boxes[:, 0])
        vec_y = np.int32(boxes[:, 1])
        vec_xend = np.int32(boxes[:, 2])
        vec_yend = np.int32(boxes[:, 3])

        return {
            "x": vec_x,
            "xend": vec_xend,
            "y": vec_y,
            "yend": vec_yend,
            "labels": labels,
            "scores": scores.detach().numpy()
        }
