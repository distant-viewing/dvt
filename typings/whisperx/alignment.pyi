from collections.abc import Iterable

import numpy as np
import torch
from numpy.typing import NDArray

from .schema import AlignedTranscriptionResult, SingleSegment

DEFAULT_ALIGN_MODELS_TORCH: dict[str, str]
DEFAULT_ALIGN_MODELS_HF: dict[str, str]

def load_align_model(
    language_code: str,
    device: str,
    model_name: str | None = None,
    model_dir: str | None = None,
    model_cache_only: bool = False,
) -> tuple[torch.nn.Module, dict[str, object]]: ...
def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict[str, object],
    audio: str | NDArray[np.float32] | torch.Tensor,
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult: ...
