"""The yolo stage: the head detector on every frame."""

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

# The settings every annotator runs under, before torch is imported.
# isort: off
from . import runtime as runtime

# isort: on
import cv2
import numpy as np
import polars as pl
import torch

from . import config
from .annotators import stamp
from .log import log_status
from .runtime import TORCH_DEVICE
from .sampling import FrameStage

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from .source import Source


class YoloStage(FrameStage):
    label = "yolo"
    needs_rgb = False
    sample_hz = None
    overlap = True
    imports = ("ultralytics",)

    CONF = 0.05

    IMGSZ = int(os.environ.get("TBM_YOLO_IMGSZ", "640"))
    TRAIN_MAX_SIDE = int(os.environ.get("TBM_YOLO_TIER", "960"))

    BATCH = int(os.environ.get("TBM_YOLO_BATCH", "32"))

    # The scale start sets: the frames are shrunk to the size the model was
    # trained at, and never enlarged.
    scale: float

    SCHEMA: ClassVar[dict[str, type[pl.DataType]]] = {
        "frame": pl.Int64,
        "score": pl.Float32,
        "label": pl.String,
        "top_left_x": pl.Float32,
        "top_left_y": pl.Float32,
        "bottom_right_x": pl.Float32,
        "bottom_right_y": pl.Float32,
    }

    def __init__(self, od: Path, source: "Source | None" = None) -> None:
        super().__init__(od, source)
        self.path = od / "yolo.parquet"
        self.frame_numbers: list[int] = []
        self.all_scores: list[float] = []
        self.top_left_x: list[float] = []
        self.top_left_y: list[float] = []
        self.bottom_right_x: list[float] = []
        self.bottom_right_y: list[float] = []
        self.pending_fnums: list[int] = []
        self.pending_imgs: list[MatLike] = []

    def done(self) -> bool:
        return self.path.exists()

    def start(self, width: int, height: int) -> None:
        # ultralytics is in the GPU environment only (the cu128 group), so it
        # is looked up by name here, where the CPU environment the checks run
        # in does not have to find it.
        YOLO = importlib.import_module("ultralytics").YOLO

        self.model = YOLO(str(config.models() / "yolo26l_heads.pt"))
        self.resize(width, height)

        off_recipe = (
            "  !!! NOT THE TRAINED CONFIGURATION (640/960) -- measurement only, do not bank"
            if (self.IMGSZ, self.TRAIN_MAX_SIDE) != (640, 960)
            else ""
        )
        log_status(
            self.od,
            "yolo",
            f"YOLO26-L on {TORCH_DEVICE}, imgsz={self.IMGSZ}, "
            f"conf={self.CONF}, source {width}x{height} "
            f"-> {int(width * self.scale)}x{int(height * self.scale)}"
            f"{off_recipe}",
        )

        blank = np.zeros(
            (max(1, int(height * self.scale)), max(1, int(width * self.scale)), 3), np.uint8
        )
        for _ in range(3):
            self.model.predict(
                blank,
                imgsz=self.IMGSZ,
                conf=self.CONF,
                verbose=False,
                device=0 if TORCH_DEVICE == "cuda" else "cpu",
            )
        if TORCH_DEVICE == "cuda":
            torch.cuda.synchronize()

    def resize(self, width: int, height: int) -> None:
        self.scale = min(1.0, self.TRAIN_MAX_SIDE / max(width, height))

    def handle(self, fnum: int, bgr: "MatLike", rgb: "MatLike | None") -> None:
        img = (
            bgr
            if self.scale >= 1.0
            else cv2.resize(
                bgr,
                (int(bgr.shape[1] * self.scale), int(bgr.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        )

        self.pending_fnums.append(fnum)
        self.pending_imgs.append(img)
        if len(self.pending_imgs) >= self.BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.pending_imgs:
            return
        fnums, imgs = self.pending_fnums, self.pending_imgs
        self.pending_fnums, self.pending_imgs = [], []
        self.submit(self._detect, fnums, imgs)

    def _detect(self, fnums: list[int], imgs: list["MatLike"]) -> None:
        results = self.model.predict(
            imgs,
            imgsz=self.IMGSZ,
            conf=self.CONF,
            verbose=False,
            device=0 if TORCH_DEVICE == "cuda" else "cpu",
        )

        inv = 1.0 / self.scale if self.scale > 0 else 1.0
        for fnum, res in zip(fnums, results, strict=True):
            boxes = res.boxes
            if boxes is None or not len(boxes):
                continue
            xyxy = boxes.xyxy.cpu().numpy() * inv
            conf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), sc in zip(xyxy, conf, strict=True):
                self.frame_numbers.append(fnum)
                self.all_scores.append(float(sc))
                self.top_left_x.append(float(x1))
                self.top_left_y.append(float(y1))
                self.bottom_right_x.append(float(x2))
                self.bottom_right_y.append(float(y2))

    def finish(self) -> None:
        self.flush()
        self.wait()

        if not self.frame_numbers:
            df = pl.DataFrame(schema=self.SCHEMA)
        else:
            df = pl.DataFrame(
                {
                    "frame": self.frame_numbers,
                    "score": np.array(self.all_scores, dtype=np.float32),
                    "label": ["head"] * len(self.frame_numbers),
                    "top_left_x": np.array(self.top_left_x, dtype=np.float32),
                    "top_left_y": np.array(self.top_left_y, dtype=np.float32),
                    "bottom_right_x": np.array(self.bottom_right_x, dtype=np.float32),
                    "bottom_right_y": np.array(self.bottom_right_y, dtype=np.float32),
                },
                schema=self.SCHEMA,
            )
        stamp(df, "yolo").write_parquet(self.path)
