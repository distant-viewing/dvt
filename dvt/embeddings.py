"""The embedding stages: DINOv2 and the two SigLIP2 models over the sampled frames."""

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

# The settings every annotator runs under, before torch is imported.
# isort: off
from . import runtime as runtime

# isort: on
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    Siglip2VisionModel,
    SiglipVisionModel,
)

from . import config
from .annotators import stamp
from .log import log_status
from .noise import quiet_partial_load
from .runtime import TORCH_DEVICE
from .sampling import FrameStage

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from .frame_ops import Box
    from .source import Source


def on_device(model: torch.nn.Module) -> torch.nn.Module:
    """The model in eval mode on TORCH_DEVICE. transformers wraps its own
    `to` so that a checker reads it as unbound; through nn.Module it reads
    as what it is."""
    return model.eval().to(TORCH_DEVICE)


class EmbeddingStage(FrameStage):
    needs_rgb = True
    backfill_shots = True
    # Each embedding stage names its table and its log label.
    parquet: str
    name: str

    PROC_KWARGS: ClassVar[dict[str, object]] = {}

    FEATURE_KEYS = ("pixel_values",)

    # The image processor and the model load_model sets up.
    processor: Callable[..., Mapping[str, torch.Tensor]]
    model: torch.nn.Module

    BATCH = int(os.environ.get("TBM_EMBED_BATCH", "32"))

    def __init__(self, od: Path, source: "Source | None" = None) -> None:
        super().__init__(od, source)
        self.path = od / self.parquet
        self.crop: Box | None = None
        self.frame_numbers: list[int] = []
        self.embeddings: list[np.ndarray] = []
        self.pending_fnums: list[int] = []
        self.pending: list[list[torch.Tensor]] = []

    def done(self) -> bool:
        return self.path.exists()

    def start(self, width: int, height: int) -> None:
        self.crop = None
        if not self.wants_crop:
            self.load_model(width, height)
            return
        self.crop = self.resolve_crop(width, height, self.name)
        self.load_model(width, height)

    def load_model(self, width: int, height: int) -> None:
        raise NotImplementedError

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def handle(self, fnum: int, bgr: "MatLike", rgb: "MatLike | None") -> None:
        if rgb is None:
            raise ValueError("an embedding stage is handed the RGB frame, as needs_rgb asks")
        if self.crop is not None:
            t, b, left, r = self.crop
            rgb = rgb[t:b, left:r]
        feats = self.processor(images=rgb, return_tensors="pt", **self.PROC_KWARGS)
        self.pending_fnums.append(fnum)
        self.pending.append([feats[k] for k in self.FEATURE_KEYS])
        if len(self.pending) >= self.BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        batch = [torch.cat(parts).to(TORCH_DEVICE) for parts in zip(*self.pending, strict=True)]
        with torch.no_grad():
            embeddings = F.normalize(self._features(*batch), dim=1)
        out = embeddings.cpu().numpy()
        self.frame_numbers.extend(self.pending_fnums)
        self.embeddings.extend(out[i : i + 1] for i in range(len(out)))
        self.pending_fnums.clear()
        self.pending.clear()

    def finish(self) -> None:
        self.flush()
        src = ["shot" if f in self.backfill else "grid" for f in self.frame_numbers]
        write_embeddings(self.path, self.frame_numbers, self.embeddings, src, self.name)


class DinoStage(EmbeddingStage):
    label = "DINOv2"
    sample_hz = 0.5
    parquet = "dino2.parquet"
    name = "dino2"

    def load_model(self, width: int, height: int) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large", use_fast=True, cache_dir=config.models()
        )
        self.model = (
            AutoModel.from_pretrained("facebook/dinov2-large", cache_dir=config.models())
            .eval()
            .to(TORCH_DEVICE)
        )
        log_status(self.od, "dino2", f"DINOv2 on {TORCH_DEVICE}, batch={self.BATCH}")

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


class SiglipStage(EmbeddingStage):
    label = "SigLIP2"
    sample_hz = 0.5
    parquet = "siglip2.parquet"
    name = "siglip2"

    wants_crop = True

    MODEL = "google/siglip2-so400m-patch16-naflex"
    PROC_KWARGS: ClassVar[dict[str, object]] = {"max_num_patches": 1024}
    FEATURE_KEYS = ("pixel_values", "pixel_attention_mask", "spatial_shapes")

    def load_model(self, width: int, height: int) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL, use_fast=True, cache_dir=config.models()
        )
        with quiet_partial_load():
            self.model = on_device(
                Siglip2VisionModel.from_pretrained(self.MODEL, cache_dir=config.models())
            )
        log_status(
            self.od,
            "siglip2",
            f"SigLIP2 NaFlex on {TORCH_DEVICE}, batch={self.BATCH}, "
            f"max_num_patches={self.PROC_KWARGS['max_num_patches']}",
        )

    def _features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        ).pooler_output


class SiglipBaseStage(EmbeddingStage):
    """The small fixed-resolution SigLIP2, on the same frames and the same
    crop as the NaFlex model. The processor squashes the crop to 224 by 224,
    aspect and all, which is how the model was trained."""

    label = "SigLIP2 base"
    sample_hz = 0.5
    parquet = "siglip2_base.parquet"
    name = "siglip2_base"

    wants_crop = True

    MODEL = "google/siglip2-base-patch16-224"

    def load_model(self, width: int, height: int) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL, use_fast=True, cache_dir=config.models()
        )
        with quiet_partial_load():
            self.model = on_device(
                SiglipVisionModel.from_pretrained(self.MODEL, cache_dir=config.models())
            )
        log_status(
            self.od, "siglip2_base", f"SigLIP2 base 224 on {TORCH_DEVICE}, batch={self.BATCH}"
        )

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


def write_embeddings(
    path: Path,
    frame_numbers: Sequence[int],
    embeddings: Sequence[np.ndarray],
    src: Sequence[str],
    name: str,
) -> None:
    if not embeddings:
        stamp(pl.DataFrame(schema={"frame": pl.Int64, "src": pl.String}), name).write_parquet(path)
        return
    df = pl.DataFrame({"frame": frame_numbers}).with_columns(
        pl.Series(
            "embedding", np.vstack(embeddings), dtype=pl.Array(pl.Float32, embeddings[0].shape[1])
        ),
        pl.Series("src", src, dtype=pl.String),
    )
    stamp(df, name).write_parquet(path)
