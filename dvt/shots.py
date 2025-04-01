# -*- coding: utf-8 -*-
"""Annotators for detecting shot boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

import random
import numpy as np

from .utils import _download_file, load_video


class AnnoShotBreaks:
    """Detect shot boundaries in a video file.

    Uses the TransNetV2 algorithm to detect shot boundaries

    Attributes:
        model_path (str): location of the saved pytorch weight for the
            detection algorithm; set to None to download from the DVT repository.
    """
    def __init__(self, model_path=None):
        if not model_path:
            model_path = _download_file("dvt_detect_shots.pt")

        self.model = _TransNetV2()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def run(self, video_path: str, visualize=False) -> dict:
        """Run the annotator.

        Args:
            video_path (str): path to a local video file; must be able
                to be read by the function cv2.VideoCapture.
            visualize (bool): should a visualization of the output be
                returned as well? Default is False.

        Returns:
            A dictionary giving the annotation.
        """
        frames = load_video(video_path, height=27, width=48)
        sfp, afp = self.model.predict_frames(frames)
        scenes = predictions_to_scenes(afp)
        if visualize:
            img = _visualize_predictions(frames, predictions=(sfp, afp), width=8)
        else:
            img = None

        return {
            "scenes": {"start": scenes[:, 0], "end": scenes[:, 1]},
            "frames": {"single_frame_pred": sfp, "average_frame_pred": afp},
            "img": img
        }



#################################################################################
# All of the code below is adapted from the MIT License library, which is 
# covered under the MIT License, Copyright (c) 2020 Tomáš Souček.

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0

    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t

    if t == 0:
        scenes.append([start, i])

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def _visualize_predictions(frames: np.ndarray, predictions, width: int):
    N, ih, iw, ic = frames.shape

    # pad frames so that length of the video is divisible by width
    # pad frames also by len(predictions) pixels in width in order to show predictions
    pad_with = width - N % width if N % width != 0 else 0
    frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions) + 1), (0, 0)])
    predictions = [np.pad(x, (0, pad_with)) for x in predictions]

    # add the predictions on the left of each frame
    for index in range(N):
        pred_range = np.arange(
            ih - round(predictions[0][index] * ih), ih, dtype=np.int32
        )
        frames[index][pred_range, iw, 0] = 214
        frames[index][pred_range, iw, 1] = 93
        frames[index][pred_range, iw, 2] = 14
        pred_range = np.arange(
            ih - round(predictions[1][index] * ih), ih, dtype=np.int32
        )
        frames[index][pred_range, iw + 1, 0] = 152
        frames[index][pred_range, iw + 1, 1] = 151
        frames[index][pred_range, iw + 1, 2] = 26

    height = len(frames) // width
    img = frames.reshape([height, width, ih + 1, iw + len(predictions) + 1, ic])
    img = np.concatenate(
        np.split(np.concatenate(np.split(img, height), axis=2)[0], width), axis=2
    )[0, :-1]

    return img


class _TransNetV2(nn.Module):
    def __init__(self, F=16, L=3, S=2, D=1024):
        super(_TransNetV2, self).__init__()

        self.SDDCNN = nn.ModuleList(
            [
                _StackedDDCNNV2(
                    in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0
                )
            ]
            + [
                _StackedDDCNNV2(
                    in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i
                )
                for i in range(1, L)
            ]
        )

        self.frame_sim_layer = _FrameSimilarity(
            sum([(F * 2**i) * 4 for i in range(L)]),
            lookup_window=101,
            output_dim=128,
            similarity_dim=128,
            use_bias=True,
        )
        self.color_hist_layer = _ColorHistograms(lookup_window=101, output_dim=128)

        self.dropout = nn.Dropout(0.5)

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        output_dim += 128
        output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)

        self.eval()

    def forward(self, inputs):
        assert (
            isinstance(inputs, torch.Tensor)
            and list(inputs.shape[2:]) == [27, 48, 3]
            and inputs.dtype == torch.uint8
        ), "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = torch.cat([self.frame_sim_layer(block_features), x], 2)
        x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot

    def predict_raw(self, frames: np.ndarray):
        with torch.no_grad():
            # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
            # input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
            single_frame_pred, all_frames_pred = self(torch.from_numpy(frames).cpu())
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frames_pred = torch.sigmoid(all_frames_pred["many_hot"]).cpu().numpy()

        single_frame_pred = single_frame_pred.reshape([-1])
        all_frames_pred = all_frames_pred.reshape([-1])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4, "Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = (
                25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
            )  # 25 - 74
            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start
                + [frames]
                + [end_frame] * no_padded_frames_end,
                0,
            )
            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr : ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred[25:75], all_frames_pred[25:75]))

            print(
                "\rProcessing video frames {}/{}".format(
                    min(len(predictions) * 50, len(frames)), len(frames)
                ),
                end="",
            )
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return (
            single_frame_pred[: len(frames)],
            all_frames_pred[: len(frames)],
        )


class _StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters,
        n_blocks,
        filters,
        shortcut=True,
        pool_type="avg",
        stochastic_depth_drop_prob=0.0,
    ):
        super(_StackedDDCNNV2, self).__init__()

        assert pool_type == "max" or pool_type == "avg"

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                _DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        self.pool = (
            nn.MaxPool3d(kernel_size=(1, 2, 2))
            if pool_type == "max"
            else nn.AvgPool3d(kernel_size=(1, 2, 2))
        )
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class _DilatedDCNNV2(nn.Module):
    def __init__(
        self, in_filters, filters, batch_norm=True, activation=None
    ):  # not supported
        super(_DilatedDCNNV2, self).__init__()

        self.Conv3D_1 = _Conv3DConfigurable(
            in_filters, filters, 1, use_bias=not batch_norm
        )
        self.Conv3D_2 = _Conv3DConfigurable(
            in_filters, filters, 2, use_bias=not batch_norm
        )
        self.Conv3D_4 = _Conv3DConfigurable(
            in_filters, filters, 4, use_bias=not batch_norm
        )
        self.Conv3D_8 = _Conv3DConfigurable(
            in_filters, filters, 8, use_bias=not batch_norm
        )

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class _Conv3DConfigurable(nn.Module):
    def __init__(
        self, in_filters, filters, dilation_rate, separable=True, use_bias=True
    ):
        super(_Conv3DConfigurable, self).__init__()

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(
                in_filters,
                2 * filters,
                kernel_size=(1, 3, 3),
                dilation=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            )
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class _FrameSimilarity(nn.Module):
    def __init__(
        self,
        in_filters,
        similarity_dim=128,
        lookup_window=101,
        output_dim=128,
        stop_gradient=False,  # not supported
        use_bias=False,
    ):
        super(_FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplemented(
                "Stop gradient not implemented in Pytorch version of Transnet!"
            )

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(
            x, x.transpose(1, 2)
        )  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(
            similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2]
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class _ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super(_ColorHistograms, self).__init__()

        self.fc = (
            nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        )
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (
            torch.arange(0, batch_size * time_window, device=frames.device) << 9
        ).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(
            batch_size * time_window * 512, dtype=torch.int32, device=frames.device
        )
        histograms.scatter_add_(
            0,
            binned_values,
            torch.ones(len(binned_values), dtype=torch.int32, device=frames.device),
        )

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(
            x, x.transpose(1, 2)
        )  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(
            similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2]
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities
