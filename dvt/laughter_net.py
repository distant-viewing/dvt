"""The network the laughter stage reads the speech stem with: a small
convolutional-recurrent net that says, every 20 ms, whether the audience is
laughing and whether someone is speaking. Trained in 2026_laugh
(17_train.py, the checkpoint 19_select.py chose); only laughter is used here.

Checked at the standard level, as transnet.py is: the layers call torch's
nn.Module.__init__, whose arguments torch leaves unannotated."""

from collections.abc import Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

GROUPS = 8
N_MELS = 128
CHUNK = 16384                  # spectrogram frames (10 ms) a piece of the convolutions
MARGIN = 32                    # frames either side of a piece; the convolutions see 8


class LaughterNet(nn.Module):
    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()

        def block(a: int, b: int, pool: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(GROUPS, b), nn.ReLU(), nn.MaxPool2d(pool)
            )

        self.conv = nn.Sequential(
            block(1, 24, (2, 2)),
            block(24, 48, (1, 2)),
            block(48, 64, (1, 2)),
            block(64, 96, (1, 4)),
            nn.Dropout(0.1),
        )
        self.proj = nn.Linear(96 * (n_mels // 32), 128)
        self.gru = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.head = nn.Linear(192, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, frames of 10 ms, mels) in dB; out: (batch, frames / 2, 2)
        logits, laughter then speech."""
        h = self.conv(((x + 50) / 20).unsqueeze(1))
        h = h.permute(0, 2, 1, 3).flatten(2)
        h, _ = self.gru(torch.relu(self.proj(h)))
        return self.head(h)

    def laughter(
        self, spec: NDArray[np.float16], level: float, chunk: int = CHUNK
    ) -> NDArray[np.float32]:
        """The probability of laughter every 20 ms of a spectrogram, (frames of
        10 ms, mels), an even number of frames, less `level` in dB.

        It gives what forward() gives over the whole file, to float rounding,
        in memory that does not grow with the file's length. The convolutions
        run over pieces of `chunk` frames, each with MARGIN frames either side
        beyond what they can see. Each GroupNorm normalises over the whole
        file, so its mean and variance are gathered over every piece first,
        one layer at a time, and only then applied. The GRU reads the whole
        file, 128 numbers a 20 ms frame."""
        n = len(spec)
        if n % 2 or chunk % 2:
            raise ValueError("the spectrogram and the piece need an even number of frames")
        blocks = [b for b in self.conv if isinstance(b, nn.Sequential)]
        with torch.no_grad():
            stats: list[tuple[torch.Tensor, torch.Tensor]] = []
            for depth in range(len(blocks)):
                total = torch.zeros(GROUPS, dtype=torch.float64)
                square = torch.zeros(GROUPS, dtype=torch.float64)
                count = 0
                for core, x in self._pieces(spec, level, chunk):
                    h = self._run(blocks, x, stats, depth)
                    h = h[:, :, core[0] if depth == 0 else core[1]]
                    g = h.reshape(GROUPS, -1).double()
                    total += g.sum(1)
                    square += (g * g).sum(1)
                    count += g.shape[1]
                mean = total / count
                stats.append((mean.float(), (square / count - mean * mean).float()))
            feats: list[torch.Tensor] = []
            for core, x in self._pieces(spec, level, chunk):
                h = self._run(blocks, x, stats, len(blocks))[:, :, core[1]]
                feats.append(torch.relu(self.proj(h.permute(0, 2, 1, 3).flatten(2))))
            h, _ = self.gru(torch.cat(feats, 1))
            p: NDArray[np.float32] = torch.sigmoid(self.head(h)[0, :, 0]).numpy()
        return p

    def _pieces(
        self, spec: NDArray[np.float16], level: float, chunk: int
    ) -> Iterator[tuple[tuple[slice, slice], torch.Tensor]]:
        """Each piece's input, with its margins, and where its own frames sit
        in what the convolutions return, before and after the time pooling."""
        n = len(spec)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            lo, hi = max(0, a - MARGIN), min(n, b + MARGIN)
            x = np.asarray(spec[lo:hi], np.float32) - level
            core = (slice(a - lo, b - lo), slice((a - lo) // 2, (b - lo) // 2))
            yield core, ((torch.from_numpy(x).unsqueeze(0) + 50) / 20).unsqueeze(1)

    @staticmethod
    def _run(
        blocks: list[nn.Sequential],
        h: torch.Tensor,
        stats: list[tuple[torch.Tensor, torch.Tensor]],
        depth: int,
    ) -> torch.Tensor:
        """The blocks before `depth`, normalised with their gathered stats,
        then, short of the last, the convolution of block `depth`."""
        for k, block in enumerate(blocks):
            conv, norm, relu, pool = block
            if not isinstance(norm, nn.GroupNorm):
                raise TypeError(f"block {k} has no GroupNorm second")
            h = conv(h)
            if k == depth:
                return h
            mean, var = stats[k]
            c, t, f = h.shape[1:]
            h = h.reshape(1, GROUPS, c // GROUPS, t, f)
            h = (h - mean.view(1, GROUPS, 1, 1, 1)) / torch.sqrt(
                var.view(1, GROUPS, 1, 1, 1) + norm.eps
            )
            h = h.reshape(1, c, t, f) * norm.weight.view(1, c, 1, 1) + norm.bias.view(1, c, 1, 1)
            h = pool(relu(h))
        return h
