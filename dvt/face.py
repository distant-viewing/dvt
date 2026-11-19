"""The face stage: SCRFD detection and ArcFace embedding on every frame."""

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
from insightface.model_zoo.scrfd import distance2bbox, distance2kps
from insightface.utils import face_align
from numpy.typing import NDArray

from . import config
from .annotators import stamp
from .log import log_status
from .runtime import TORCH_DEVICE
from .sampling import FrameStage

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from .source import Source

Floats = NDArray[np.float32]
# A face: its box, its score and its five landmarks, on the source frame.
Face = tuple[Floats, np.float32, Floats]


# The embeddings are kept in float16: they are 97% of the table, and the one
# reader, tracks, compares them by cosine, which float16's three digits leave
# as they were. They are computed in float32.
EMBEDDING = pl.Array(pl.Float16, 512)


class FaceStage(FrameStage):
    label = "face"
    sample_hz = None
    overlap = True
    imports = ("onnx2torch",)

    SCHEMA: ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "frame": pl.Int64,
        "face_id": pl.Int64,
        "det_score": pl.Float32,
        "bbox_x1": pl.Float32,
        "bbox_y1": pl.Float32,
        "bbox_x2": pl.Float32,
        "bbox_y2": pl.Float32,
        "kps": pl.Array(pl.Float32, 10),
        "embedding": EMBEDDING,
    }

    DET_SIZE = (640, 640)
    DET_THRESH = 0.5
    NMS_THRESH = 0.4
    INPUT_MEAN, INPUT_STD = 127.5, 128.0
    REC_MEAN, REC_STD = 127.5, 127.5
    FEAT_STRIDES = (8, 16, 32)
    FMC = 3
    NUM_ANCHORS = 2

    BATCH = int(os.environ.get("TBM_FACE_BATCH", "32"))

    REC_BATCH = int(os.environ.get("TBM_REC_BATCH", "256"))

    # What start sets: the models, the size a frame is shrunk to before
    # detection, the anchor centres of each stride, and the batch the frames
    # are staged in.
    det: torch.nn.Module
    rec: torch.nn.Module
    new_w: int
    new_h: int
    det_scale: float
    anchors: dict[int, Floats]
    det_stage: torch.Tensor
    det_np: NDArray[np.uint8]

    def __init__(self, od: Path, source: "Source | None" = None) -> None:
        super().__init__(od, source)
        self.path = od / "face.parquet"
        self.pending_fnums: list[int] = []
        self.pending_frames: list[MatLike] = []
        self.frame_numbers: list[int] = []
        self.face_ids: list[int] = []
        self.det_scores: list[float] = []
        self.bbox_x1: list[float] = []
        self.bbox_y1: list[float] = []
        self.bbox_x2: list[float] = []
        self.bbox_y2: list[float] = []
        self.all_kps: list[Floats] = []
        self.all_embeddings: list[Floats | None] = []

    def done(self) -> bool:
        return self.path.exists()

    def start(self, width: int, height: int) -> None:
        from onnx2torch import convert

        pack = config.models() / "insightface" / "models" / "buffalo_l"
        self.det = convert(str(pack / "det_10g.onnx")).eval().to(TORCH_DEVICE)
        self.rec = convert(str(pack / "w600k_r50.onnx")).eval().to(TORCH_DEVICE)

        dw, dh = self.DET_SIZE
        self.anchors = {}
        for stride in self.FEAT_STRIDES:
            h, w = dh // stride, dw // stride
            centers = np.stack(list(np.mgrid[:h, :w][::-1]), axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            self.anchors[stride] = np.stack([centers] * self.NUM_ANCHORS, axis=1).reshape((-1, 2))

        self.det_stage = torch.zeros((self.BATCH, dh, dw, 3), dtype=torch.uint8)
        if TORCH_DEVICE == "cuda":
            self.det_stage = self.det_stage.pin_memory()
        self.det_np = self.det_stage.numpy()
        self.resize(width, height)

        log_status(
            self.od,
            "face",
            f"SCRFD det_10g + w600k_r50 (torch) on {TORCH_DEVICE}, "
            f"det_size={dw}x{dh}, batch={self.BATCH}, "
            f"source {width}x{height} -> {self.new_w}x{self.new_h} padded",
        )

    def resize(self, width: int, height: int) -> None:
        dw, dh = self.DET_SIZE
        im_ratio, model_ratio = height / width, dh / dw
        if im_ratio > model_ratio:
            self.new_h, self.new_w = dh, int(dh / im_ratio)
        else:
            self.new_w, self.new_h = dw, int(dw * im_ratio)
        self.det_scale = self.new_h / height
        # A frame fills the top left of its slot and the rest stays black,
        # so what the last size left outside the new one is cleared.
        self.det_stage.zero_()

    def _hw(self, stride: int) -> int:
        dw, dh = self.DET_SIZE
        return (dh // stride) * (dw // stride)

    def _scores_for(self, out: Floats, stride: int, n: int, i: int) -> Floats:
        return out.reshape(self._hw(stride), n, self.NUM_ANCHORS)[:, i].ravel()

    def _rows_for(self, out: Floats, stride: int, n: int, i: int, pos: NDArray[np.intp]) -> Floats:
        a = self.NUM_ANCHORS
        return out[((pos // a) * n + i) * a + (pos % a)]

    def handle(self, fnum: int, bgr: "MatLike", rgb: "MatLike | None") -> None:
        self.pending_fnums.append(fnum)
        self.pending_frames.append(bgr)
        if len(self.pending_frames) >= self.BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.pending_frames:
            return
        fnums, frames = self.pending_fnums, self.pending_frames
        self.pending_fnums, self.pending_frames = [], []
        self.submit(self._detect, fnums, frames)

    def _detect(self, fnums: list[int], frames: list["MatLike"]) -> None:
        n = len(frames)

        for i, f in enumerate(frames):
            cv2.resize(f, (self.new_w, self.new_h), dst=self.det_np[i, : self.new_h, : self.new_w])

        staged = self.det_stage[:n].to(TORCH_DEVICE, non_blocking=True)
        blob = (staged.permute(0, 3, 1, 2).flip(1).float() - self.INPUT_MEAN) * (
            1.0 / self.INPUT_STD
        )

        with torch.no_grad():
            outs = self.det(blob)
        outs = [o.cpu().numpy() for o in outs]

        crops: list[MatLike] = []
        owners: list[int] = []
        for i in range(n):
            scores_list: list[Floats] = []
            bboxes_list: list[Floats] = []
            kpss_list: list[Floats] = []
            for idx, stride in enumerate(self.FEAT_STRIDES):
                scores = self._scores_for(outs[idx], stride, n, i)
                pos = np.where(scores >= self.DET_THRESH)[0]
                if not pos.size:
                    continue
                centers = self.anchors[stride][pos]
                bbox_preds = self._rows_for(outs[idx + self.FMC], stride, n, i, pos) * stride
                kps_preds = self._rows_for(outs[idx + self.FMC * 2], stride, n, i, pos) * stride

                scores_list.append(scores[pos])
                bboxes_list.append(distance2bbox(centers, bbox_preds))
                kpss = distance2kps(centers, kps_preds)
                kpss_list.append(kpss.reshape((kpss.shape[0], -1, 2)))

            faces = self._nms_one(scores_list, bboxes_list, kpss_list)
            for fid, (bbox, score, kps) in enumerate(faces):
                self.frame_numbers.append(fnums[i])
                self.face_ids.append(fid)
                self.det_scores.append(float(score))
                self.bbox_x1.append(float(bbox[0]))
                self.bbox_y1.append(float(bbox[1]))
                self.bbox_x2.append(float(bbox[2]))
                self.bbox_y2.append(float(bbox[3]))
                self.all_kps.append(kps.flatten().astype(np.float32))
                crops.append(face_align.norm_crop(frames[i], landmark=kps, image_size=112))
                owners.append(len(self.all_embeddings))
                self.all_embeddings.append(None)

        for i in range(0, len(crops), self.REC_BATCH):
            chunk = crops[i : i + self.REC_BATCH]
            rblob = cv2.dnn.blobFromImages(
                chunk, 1.0 / self.REC_STD, (112, 112), (self.REC_MEAN,) * 3, swapRB=True
            )
            with torch.no_grad():
                feats = self.rec(torch.as_tensor(rblob).to(TORCH_DEVICE)).cpu().numpy()
            for slot, feat in zip(owners[i : i + self.REC_BATCH], feats, strict=True):
                self.all_embeddings[slot] = feat.flatten().astype(np.float32)

    def _nms_one(
        self, scores_list: list[Floats], bboxes_list: list[Floats], kpss_list: list[Floats]
    ) -> list[Face]:
        if not scores_list:
            return []
        scores = np.concatenate(scores_list)
        bboxes = np.vstack(bboxes_list) / self.det_scale
        kpss = np.vstack(kpss_list) / self.det_scale

        for _ in range(2):
            order = scores.argsort()[::-1]
            bboxes, kpss, scores = bboxes[order], kpss[order], scores[order]

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        rest = scores.argsort()[::-1]
        keep: list[np.intp] = []
        while rest.size > 0:
            i = rest[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[rest[1:]])
            yy1 = np.maximum(y1[i], y1[rest[1:]])
            xx2 = np.minimum(x2[i], x2[rest[1:]])
            yy2 = np.minimum(y2[i], y2[rest[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[rest[1:]] - inter)
            rest = rest[np.where(ovr <= self.NMS_THRESH)[0] + 1]

        return [(bboxes[i], scores[i], kpss[i]) for i in keep]

    def finish(self) -> None:
        self.flush()
        self.wait()

        if not self.frame_numbers:
            df = pl.DataFrame(schema=self.SCHEMA)
        else:
            df = pl.DataFrame(
                {
                    "frame": self.frame_numbers,
                    "face_id": self.face_ids,
                    "det_score": np.array(self.det_scores, dtype=np.float32),
                    "bbox_x1": np.array(self.bbox_x1, dtype=np.float32),
                    "bbox_y1": np.array(self.bbox_y1, dtype=np.float32),
                    "bbox_x2": np.array(self.bbox_x2, dtype=np.float32),
                    "bbox_y2": np.array(self.bbox_y2, dtype=np.float32),
                }
            ).with_columns(
                pl.Series("kps", self.all_kps, dtype=pl.Array(pl.Float32, 10)),
                pl.Series("embedding", self.all_embeddings, dtype=pl.Array(pl.Float32, 512)).cast(
                    EMBEDDING
                ),
            )
        stamp(df, "face").write_parquet(self.path)
