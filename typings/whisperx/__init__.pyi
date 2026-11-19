import numpy as np
from numpy.typing import NDArray

from .alignment import align as align
from .alignment import load_align_model as load_align_model
from .schema import TranscriptionResult

class FasterWhisperPipeline:
    def transcribe(
        self,
        audio: str | NDArray[np.float32],
        batch_size: int | None = None,
        num_workers: int = 0,
        language: str | None = None,
    ) -> TranscriptionResult: ...

def load_model(
    whisper_arch: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    asr_options: dict[str, object] | None = None,
    language: str | None = None,
) -> FasterWhisperPipeline: ...
def load_audio(file: str, sr: int = 16000) -> NDArray[np.float32]: ...
