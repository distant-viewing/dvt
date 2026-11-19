"""Where the models are and where scratch files go. Set once, before any stage runs."""

import os
import tempfile
from pathlib import Path

from .tools.values import StrPath

MODELS_ENV = "DVT_MODELS"
# Where the models go when nothing names a directory.
DEFAULT_MODELS = Path.home() / ".cache" / "dvt"

_models: Path | None = None
_tmp: Path | None = None


def configure(models: StrPath, tmp: StrPath | None = None) -> None:
    """Names the model directory and, if the system's is too small for a
    film's separated audio, a scratch directory. Call it before anything
    loads a model: it points torch hub into the model directory too."""
    global _models, _tmp
    _models = Path(models)
    _tmp = Path(tmp) if tmp is not None else None
    os.environ["TORCH_HOME"] = str(torch_home())


def models() -> Path:
    found = _models
    if found is None:
        found = Path(os.environ.get(MODELS_ENV) or DEFAULT_MODELS)
        configure(found)
    return found


def tmp() -> Path:
    return _tmp if _tmp is not None else Path(tempfile.gettempdir()) / "dvt"


def whisper() -> Path:
    return models() / "whisper-large-v3-ct2"


def nltk_data() -> Path:
    return models() / "nltk_data"


def separator() -> Path:
    return models() / "audio-separator"


def torch_home() -> Path:
    return models() / "torch"


def assert_torch_home_pinned() -> None:
    import torch

    hub_dir = Path(torch.hub.get_dir())
    if models() not in hub_dir.parents:
        raise RuntimeError(
            f"torch hub resolves to {hub_dir}, outside {models()}. "
            "Call config.configure before triggering any torch.hub download."
        )
