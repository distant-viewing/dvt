# -*- coding: utf-8 -*-
"""Annotators for embedding images.
"""

import numpy as np
import torch
from torchvision import transforms as T
from torch.hub import load_state_dict_from_url

from .utils import _download_file


class AnnoEmbed:
    """Convert an image into an embedding in a lower-dimensional space.

    Uses the EfficientNet weights trained on ImageNet, as provided by 
    the PyTorch model zoo.

    Attributes:
        model_path (str): location of the saved pytorch weights; set to
            None to download from the DVT repository.
    """

    def __init__(self, model_path=None):
        if not model_path:
            model_path = _download_file("dvt_embed.pt")
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def run(self, img: np.ndarray) -> dict:
        """Run the annotator.

        Args:
            img (np.ndarray): image file, as returned by dvt.load_image,
                or dvt.yield_video (for frame from a video file).

        Returns:
            A dictionary giving the annotation.
        """
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img / 255).unsqueeze(0)

        output = self.model(img).detach().numpy()[0]
        return {"embedding": output}
