# -*- coding: utf-8 -*-
"""Annotators for detecting image segmentation.
"""

import numpy as np
import torch
from torchvision import transforms as T
from torch.hub import load

from .utils import _download_file



CATEGORIES = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']



class AnnoSegmentation:
    """Detect semantic segmentation of an input image.

    Uses the DeepLabV3 weights trained on MS COCO + VOC, as provided by 
    the PyTorch model zoo.

    Attributes:
        model_path (str): location of the saved pytorch weights; set to
            None to download from the DVT repository.
    """
    def __init__(self, model_path=None):
        if not model_path:
            model_path = _download_file("dvt_segment.pt")
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
        img = self.norm(img)

        output = self.model(img)
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        normalized_masks = normalized_masks.detach().numpy()[0]

        # clean up some of the output
        z = np.argmax(normalized_masks, axis=(0))
        values, counts = np.unique(z.flatten(), return_counts=True)
        counts = {"category": [CATEGORIES[x] for x in values], "count": np.array(counts)}

        return {"counts": counts, "mask": normalized_masks}
