import torch

class Wav2Vec2Bundle:
    # The checkpoint's name under torch hub's checkpoints directory.
    _path: str
    def get_model(self, *, dl_kwargs: dict[str, object] | None = None) -> torch.nn.Module: ...
