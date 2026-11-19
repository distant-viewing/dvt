"""Where each language's wav2vec2 aligner lives."""

from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

from . import config

TORCHAUDIO = "torchaudio"
HUGGINGFACE = "huggingface"


def supported_languages() -> list[str]:
    return sorted({*DEFAULT_ALIGN_MODELS_TORCH, *DEFAULT_ALIGN_MODELS_HF})


def resolve_align_model(language: str, align_model: str | None = None) -> tuple[str, str]:
    if align_model is not None:
        return HUGGINGFACE, align_model
    if language in DEFAULT_ALIGN_MODELS_TORCH:
        return TORCHAUDIO, DEFAULT_ALIGN_MODELS_TORCH[language]
    if language in DEFAULT_ALIGN_MODELS_HF:
        return HUGGINGFACE, DEFAULT_ALIGN_MODELS_HF[language]
    raise ValueError(
        f"whisperx has no default aligner for {language!r}. Supported: "
        f"{', '.join(supported_languages())}. To use another language, pass an "
        "explicit HuggingFace wav2vec2 model via --align-model."
    )


def align_model_dir(language: str, align_model: str | None = None) -> str | None:
    backend, _ = resolve_align_model(language, align_model)
    return None if backend == TORCHAUDIO else str(config.models())
