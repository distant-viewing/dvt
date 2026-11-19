import torch

def load_silero_vad(onnx: bool = False, opset_version: int = 16) -> torch.jit.ScriptModule: ...

# Times in seconds with return_seconds, in samples without.
def get_speech_timestamps(
    audio: torch.Tensor,
    model: torch.jit.ScriptModule,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = ...,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
) -> list[dict[str, float]]: ...
