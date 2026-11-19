from pathlib import Path

from torch import fx

def convert(
    onnx_model_or_path: str | Path,
    save_input_names: bool = False,
    attach_onnx_mapping: bool = False,
) -> fx.GraphModule: ...
