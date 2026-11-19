"""Small videos made with ffmpeg, for the annotators that read one."""

import shutil
import subprocess
from pathlib import Path

import pytest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def make_video(
    path: Path,
    seconds: float = 4,
    rate: int = 10,
    size: tuple[int, int] = (160, 120),
    bars: tuple[int, int, int, int] = (0, 0, 0, 0),
    gray: bool = False,
    audio: bool = True,
    lang: str | None = "eng",
    chapters: list[tuple[float, float, str]] | None = None,
    subtitles: list[tuple[float, float, str]] | None = None,
    sub_lang: str = "fra",
) -> Path:
    """A test picture of `size` inside black bars of top, bottom, left and
    right pixels, at `rate` frames a second, with a silent AAC track tagged
    `lang`, and the chapters and a text subtitle track asked for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    top, bottom, left, right = bars
    vf = f"pad={w + left + right}:{h + top + bottom}:{left}:{top}:black"
    if gray:
        vf += ",hue=s=0"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-t",
        str(seconds),
        "-i",
        f"testsrc2=size={w}x{h}:rate={rate}",
    ]
    inputs = 1
    if audio:
        cmd += ["-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=48000:cl=stereo"]
        inputs += 1
    meta = None
    chapter_input = sub_input = 0
    if chapters:
        meta = path.with_suffix(".chapters.txt")
        lines = [";FFMETADATA1"]
        for t0, t1, title in chapters:
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={int(t0 * 1000)}",
                f"END={int(t1 * 1000)}",
                f"title={title}",
            ]
        meta.write_text("\n".join(lines) + "\n")
        cmd += ["-i", str(meta)]
        chapter_input = inputs
        inputs += 1
    srt = None
    if subtitles:
        srt = path.with_suffix(".srt")
        srt.write_text(
            "".join(
                f"{i + 1}\n{stamp(t0)} --> {stamp(t1)}\n{text}\n\n"
                for i, (t0, t1, text) in enumerate(subtitles)
            )
        )
        cmd += ["-i", str(srt)]
        sub_input = inputs
        inputs += 1
    cmd += ["-map", "0:v"]
    if audio:
        cmd += ["-map", "1:a"]
    if subtitles:
        cmd += [
            "-map",
            f"{sub_input}:s",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"language={sub_lang}",
        ]
    if chapters:
        cmd += ["-map_chapters", str(chapter_input)]
    cmd += ["-vf", vf, "-c:v", "libx264", "-g", str(rate), "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-metadata:s:a:0", f"language={lang or ''}"]
    cmd += [str(path)]  # fmt: skip
    subprocess.run(cmd, check=True)
    for extra in (meta, srt):
        if extra is not None:
            extra.unlink()
    return path


def stamp(t: float) -> str:
    """t seconds as an srt time."""
    ms = round(t * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model directory holding a TransNetV2 with seeded random weights. Its
    shot scores mean nothing, but they are the same on every run."""
    import numpy as np
    import torch

    from dvt.transnet import TransNetV2

    root = tmp_path_factory.mktemp("models")
    net = TransNetV2()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.as_tensor(rng.normal(0.0, 0.05, tuple(p.shape))))
    torch.save(net.state_dict(), root / "detect_shots.pt")
    return root


@pytest.fixture
def models(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from dvt import config

    monkeypatch.setattr(config, "_models", model_dir)
    return model_dir
