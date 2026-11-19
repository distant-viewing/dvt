import random
import sys
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from numpy.typing import NDArray

PROGRESS_LINES = 10

Scores = NDArray[np.float32]


class TransNetV2(nn.Module):
    def __init__(self, F: int = 16, L: int = 3, S: int = 2, D: int = 1024) -> None:
        super().__init__()

        self.SDDCNN = nn.ModuleList(
            [_StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)]
            + [
                _StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i)
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

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6
        output_dim += 128
        output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)

        self.eval()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert (
            isinstance(inputs, torch.Tensor)
            and list(inputs.shape[2:]) == [27, 48, 3]
            and inputs.dtype == torch.uint8
        ), "incorrect input type and/or shape"
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features: list[torch.Tensor] = []
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
        return one_hot, {"many_hot": self.cls_layer2(x)}

    def predict_raw(self, frames: NDArray[np.uint8]) -> tuple[Scores, Scores]:
        with torch.no_grad():
            single, many = self(torch.as_tensor(frames).cpu())
            single_frame_pred: Scores = torch.sigmoid(single).cpu().numpy()
            all_frames_pred: Scores = torch.sigmoid(many["many_hot"]).cpu().numpy()

        return single_frame_pred.reshape([-1]), all_frames_pred.reshape([-1])

    def predict_frames(
        self, frames: NDArray[np.uint8] | torch.Tensor, label: str | None = None
    ) -> tuple[Scores, Scores]:
        assert len(frames.shape) == 4, "Input shape must be [frames, height, width, 3]."

        def input_iterator() -> Iterator[NDArray[np.uint8]]:
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
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

        predictions: list[tuple[Scores, Scores]] = []

        interactive = sys.stdout.isatty()
        prefix = f"[{label}:frames] " if label else ""
        every = max(1, len(frames) // PROGRESS_LINES)
        logged = 0

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred[25:75], all_frames_pred[25:75]))

            done = min(len(predictions) * 50, len(frames))
            if interactive:
                print(f"\r{prefix}Processing video frames {done}/{len(frames)}", end="")
            elif done - logged >= every:
                print(f"{prefix}Processing video frames {done}/{len(frames)}", flush=True)
                logged = done

        if interactive:
            print("")
        elif logged < len(frames):
            print(f"{prefix}Processing video frames {len(frames)}/{len(frames)}", flush=True)

        single_frame_pred = np.concatenate([single for single, _ in predictions])
        all_frames_pred = np.concatenate([every for _, every in predictions])

        return (
            single_frame_pred[: len(frames)],
            all_frames_pred[: len(frames)],
        )


class _StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        shortcut: bool = True,
        pool_type: str = "avg",
        stochastic_depth_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        shortcut: torch.Tensor | None = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        # The port this came from tested `self.shortcut is not None`, which a
        # bool always is, so the shortcut is added whatever `shortcut` says.
        # The trained weights were made that way, and it is kept.
        if shortcut is None:
            raise ValueError("a stacked block with no layers has no shortcut")
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
        self,
        in_filters: int,
        filters: int,
        batch_norm: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.Conv3D_1 = _Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = _Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = _Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = _Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        self,
        in_filters: int,
        filters: int,
        dilation_rate: int,
        separable: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if separable:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class _FrameSimilarity(nn.Module):
    def __init__(
        self,
        in_filters: int,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
        stop_gradient: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        if stop_gradient:
            raise NotImplementedError(
                "Stop gradient not implemented in Pytorch version of Transnet!"
            )

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))
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
    def __init__(self, lookup_window: int = 101, output_dim: int | None = None) -> None:
        super().__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.int()

        def get_bin(frames: torch.Tensor) -> torch.Tensor:
            r, g, b = frames[:, :, 0] >> 5, frames[:, :, 1] >> 5, frames[:, :, 2] >> 5
            return (r << 6) + (g << 3) + b

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))
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
