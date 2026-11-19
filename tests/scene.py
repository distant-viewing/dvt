"""A made-up scene for the tracks stage: shots, YOLO heads and InsightFace
faces of a few people moving across the frame, written as the stages that
come before tracks write them."""

from pathlib import Path

import numpy as np
import polars as pl

from dvt import face

EMB_DIM, KPS_DIM = 512, 10


def write_scene(od: Path, seed: int = 0, people: int = 2, n_frames: int = 120,
                cut: int = 60, width: int = 640, height: int = 360) -> None:  # fmt: skip
    """Each person has an embedding of their own, a head that drifts across
    the frame, and a face inside it that is now and then missed; heads are
    missed now and then too, and a stray face turns up with no head."""
    rng = np.random.default_rng(seed)
    od.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"sid": [0, 1], "f0": [0, cut], "f1": [cut - 1, n_frames - 1]}).write_parquet(
        od / "shots.parquet")  # fmt: skip
    pl.DataFrame({"fps": [24.0], "img_width": [width], "img_height": [height]}).write_parquet(
        od / "metadata.parquet")  # fmt: skip
    ids = rng.normal(0, 1, (people + 1, EMB_DIM))
    ids /= np.linalg.norm(ids, axis=1, keepdims=True)
    start = rng.uniform(40, width - 160, (people, 2)) * [1, 0.5]
    step = rng.normal(0, 1.5, (people, 2))
    heads: list[dict[str, object]] = []
    faces: list[dict[str, object]] = []
    for f in range(n_frames):
        face_id = 0
        for p in range(people):
            x, y = start[p] + step[p] * f + rng.normal(0, 0.8, 2)
            w, h = 80 + 5 * p, 100 + 5 * p
            if rng.random() > 0.08:
                heads.append({"frame": f, "score": float(rng.uniform(0.5, 0.95)), "label": "head",
                              "top_left_x": x, "top_left_y": y, "bottom_right_x": x + w,
                              "bottom_right_y": y + h})  # fmt: skip
            if rng.random() > 0.15:
                # The second person turns into the third halfway through the
                # second shot: an identity switch on one track.
                who = p if not (p == 1 and f > cut + 30) else people
                emb = ids[who] * rng.uniform(18, 26) + rng.normal(0, 0.15, EMB_DIM)
                fx, fy = x + 15, y + 20
                kps = [fx + 15, fy + 25, fx + 35, fy + 25, fx + 25 + rng.normal(0, 3), fy + 40,
                       fx + 18, fy + 52, fx + 32, fy + 52]  # fmt: skip
                faces.append({"frame": f, "face_id": face_id,
                              "det_score": float(rng.uniform(0.6, 0.99)),
                              "bbox_x1": fx, "bbox_y1": fy, "bbox_x2": fx + 50, "bbox_y2": fy + 60,
                              "kps": kps, "embedding": emb.tolist()})  # fmt: skip
                face_id += 1
        if f % 37 == 5:
            faces.append({"frame": f, "face_id": face_id, "det_score": 0.7,
                          "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 40.0, "bbox_y2": 45.0,
                          "kps": [20.0] * KPS_DIM,
                          "embedding": (ids[0] * 5).tolist()})  # fmt: skip
    pl.DataFrame(heads, schema={"frame": pl.Int64, "score": pl.Float32, "label": pl.String,
                                "top_left_x": pl.Float32, "top_left_y": pl.Float32,
                                "bottom_right_x": pl.Float32,
                                "bottom_right_y": pl.Float32}).write_parquet(
        od / "yolo.parquet")  # fmt: skip
    pl.DataFrame(faces, schema={"frame": pl.Int64, "face_id": pl.Int64, "det_score": pl.Float32,
                                "bbox_x1": pl.Float32, "bbox_y1": pl.Float32,
                                "bbox_x2": pl.Float32, "bbox_y2": pl.Float32,
                                "kps": pl.Array(pl.Float32, KPS_DIM),
                                "embedding": face.EMBEDDING}).write_parquet(
        od / "face.parquet")  # fmt: skip
