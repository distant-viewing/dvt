import torch

class Separator:
    # None until the constructor has chosen a device.
    torch_device: torch.device | None
    def __init__(
        self,
        log_level: int = ...,
        log_formatter: object = None,
        model_file_dir: str = "/tmp/audio-separator-models/",
        output_dir: str | None = None,
        output_format: str = "WAV",
        output_bitrate: str | None = None,
        normalization_threshold: float = 0.9,
        amplification_threshold: float = 0.0,
        output_single_stem: str | None = None,
        invert_using_spec: bool = False,
        sample_rate: int = 44100,
        use_soundfile: bool = False,
        use_autocast: bool = False,
        use_directml: bool = False,
        chunk_duration: float | None = None,
        mdx_params: dict[str, int | float | bool] = ...,
        vr_params: dict[str, int | float | bool] = ...,
        demucs_params: dict[str, str | int | float | bool] = ...,
        mdxc_params: dict[str, int | float | bool] = ...,
        info_only: bool = False,
    ) -> None: ...
    def load_model(self, model_filename: str = ...) -> None: ...
    def download_model_files(self, model_filename: str) -> tuple[str, str, str, str, str | None]: ...
    def separate(
        self, audio_file_path: str | list[str], custom_output_names: dict[str, str] | None = None
    ) -> list[str]: ...
