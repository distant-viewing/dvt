"""The metadata stage: the frame size, the letterbox crop, the aspect, the
language, and the bytes of both sources."""

import os
import time

import cv2
import polars as pl

from . import aspect, audio_clock, language, letterbox
from .annotators import stamp
from .log import log_finished, log_status
from .source import Source
from .tools import ffprobe, filehash
from .tools.values import StrPath


def detect_crop(ip: StrPath, width: int, height: int) -> letterbox.Crop:
    rows, _ = letterbox.sampled(ip)
    n_sampled = len(rows) if rows else 0
    profiles = letterbox.profiles_from_rows(rows) if rows else (None, None)

    if profiles[0] is None:
        return letterbox.crop_from_profiles([], [], width, height, n_sampled)

    return letterbox.crop_from_profiles(*profiles, width, height, n_sampled)


def create_meta(source: Source) -> None:
    ip, od = source.video, source.out

    meta_path = od / "metadata.parquet"
    if meta_path.exists():
        log_status(od, "metadata", "Already exists")
        return

    m4a_path = source.sound
    if m4a_path is None:
        raise FileNotFoundError(
            f"{od.name}: no audio recorded at {source.audio} and no "
            f"{filehash.NOAUDIO_SUFFIX} marker beside it. metadata.parquet "
            f"records both sources or neither, so nothing can be annotated for "
            f"this film until its sound has been taken out; see "
            f"dvt.tools.audio.extract_audio."
        )

    t0 = time.monotonic()
    vinput = cv2.VideoCapture(str(ip))
    n_frames = int(vinput.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vinput.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vinput.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vinput.get(cv2.CAP_PROP_FPS)
    vinput.release()

    crop = letterbox.crop_from_profiles([], [], width, height)
    if n_frames > 0:
        crop = detect_crop(ip, width, height)

    top, bottom = crop["crop_top"], crop["crop_bottom"]
    left, right = crop["crop_left"], crop["crop_right"]
    kept = (height - top - bottom) * (width - left - right)

    ar = aspect.aspect_fields(crop["content_ar"], width, height, source.container)

    df = pl.DataFrame(
        {
            "n_frames": n_frames,
            "fps": fps,
            "img_height": height,
            "img_width": width,
            **crop,
            **ar,
            "crop_frac": 1.0 - kept / (height * width) if height * width else 0.0,
        }
    )

    mp4_bytes, mp4_hash = filehash.size_and_hash(ip)
    m4a_bytes, m4a_hash = filehash.size_and_hash(m4a_path)
    for kind, path, digest in (("mp4", ip, mp4_hash), ("m4a", m4a_path, m4a_hash)):
        if digest is None:
            raise OSError(
                f"{od.name}: cannot read the {kind} at {path} to "
                f"hash it, so this row would record no source on "
                f"that side. Nothing was written."
            )
    df = df.with_columns(
        pl.lit(mp4_bytes, pl.Int64).alias("mp4_bytes"),
        pl.lit(mp4_hash, pl.String).alias("mp4_hash"),
        pl.lit(m4a_bytes, pl.Int64).alias("m4a_bytes"),
        pl.lit(m4a_hash, pl.String).alias("m4a_hash"),
    )

    lang = language.resolve(source)
    df = df.with_columns(
        pl.lit(lang.code, pl.String).alias("language_code"),
        pl.lit(lang.declared, pl.String).alias("language_declared"),
    )

    # Where the first frame and the first sample of the m4a sit on the mp4's
    # timeline. The audio tables are written with the latter added, so they
    # are on the mp4's clock too; see audio_clock.
    video_start = ffprobe.start(ip, "video")
    audio_start = audio_clock.start(ip, m4a_path) if m4a_path.suffix == ".m4a" else None
    df = df.with_columns(
        pl.lit(video_start, pl.Float64).alias("video_start"),
        pl.lit(audio_start, pl.Float64).alias("audio_start"),
    )

    tmp_path = meta_path.with_suffix(".parquet.tmp")
    stamp(df, "metadata").write_parquet(tmp_path)
    os.replace(tmp_path, meta_path)
    marks = [k[5:] for k in ("crop_union", "crop_grey", "crop_override") if crop[k]]
    if not crop["crop_usable"]:
        marks.append("NOT MEASURED")
    elif not crop["crop_stable"]:
        marks.append(f"drift {crop['crop_drift_px']:.0f}px")
    log_status(
        od,
        "metadata",
        f"crop t/b/l/r={top}/{bottom}/{left}/{right} "
        f"({100 * df['crop_frac'][0]:.1f}% bar"
        f"{', ' + ', '.join(marks) if marks else ''})",
    )

    ar_marks: list[str] = []
    if not ar["ar_known"]:
        ar_marks.append("square pixels assumed")
    elif ar["sar_ratio"] != 1.0:
        ar_marks.append(f"sar {ar['sar_ratio']:.4f}")
    if ar["ar_dist"] > aspect.AR_MAX:
        ar_marks.append(f"NO STANDARD RATIO within {aspect.AR_MAX}")
    if ar["ar_break"]:
        ar_marks.append("CONTAINER INCONSISTENT")
    if ar["ar_note"]:
        ar_marks.append(ar["ar_note"])
    log_status(
        od,
        "metadata",
        f"aspect {ar['display_ar']:.3f} "
        f"(nearest {ar['ar_nearest']:g}, {ar['ar_dist']:.3f} away"
        f"{'; ' + '; '.join(ar_marks) if ar_marks else ''})",
    )
    log_status(od, "metadata", f"language {lang.note}")
    log_finished(od, "metadata", t0)
