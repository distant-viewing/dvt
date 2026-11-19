"""What is set before any model loads: quiet libraries, thread counts, the device.

Every annotator that loads a model imports this first, and so does run. The
TF32 switches below change what a convolution returns on some cards, so a
table is the same whichever annotators the process happened to load.
"""

import os

from .noise import silence

silence()

import torch  # noqa: E402

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"
    torch.set_num_threads(4)


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TBM_OVERLAP=0 runs every batch on the main thread, for timing against.
OVERLAP = os.environ.get("TBM_OVERLAP", "1") != "0"

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
