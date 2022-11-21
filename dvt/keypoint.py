# -*- coding: utf-8 -*-
"""Annotators for detecting keypoints.
"""

import numpy as np
import torch
from torchvision import transforms as T

from .utils import _download_file


class AnnoKeypoints:
    """Detect body keypoints in an input image.

    Uses the KeypointRCNN weights trained on MS COCO, as provided by 
    the PyTorch model zoo.

    Attributes:
        model_path (str): location of the saved pytorch weights; set to
            None to download from the DVT repository.
    """
    def __init__(self, model_path=None):
        if not model_path:
            model_path = _download_file("dvt_detect_keypoint.pt")
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
        img_t = np.transpose(img, (2, 0, 1))
        img_t = torch.Tensor(img_t / 255).unsqueeze(0)

        # Detect the keypoints
        kpnt = self.model(img_t)
        if not len(kpnt):
            return None
        kpnt = [x.detach().numpy() for x in kpnt]

        # Clean up the results
        npeople = kpnt[0].shape[0]
        output = {"person_id": np.repeat(np.arange(npeople), 17),
                  "kpnt_id": np.tile(np.arange(17), npeople),
                  "x": np.int32(kpnt[3][:, :, 0].flatten()),
                  "y": np.int32(kpnt[3][:, :, 1].flatten()),
                  "prob": np.repeat(kpnt[2], 17)}

        # visualize?
        if visualize:
            img_out = img.copy()
            yellow = [250, 189, 47]
            for index in range(npeople):
                for idk in range(17):
                    if kpnt[2][index] > 0.5:
                        img_out[
                            int(kpnt[3][index, idk, 1]),
                            int(kpnt[3][index, idk, 0])
                        ] = yellow
        else:
            img_out = None


        return {"kpnt": output, "img": img_out}
