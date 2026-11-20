"""The annotators: what each writes, what it reads, and which kind of machine it runs on."""

from dataclasses import dataclass

import polars as pl

# The four kinds of run an annotator belongs to. The cpu and gpu sides read
# the video; gpuaudio separates the sound into stems, and cpuaudio reads
# the speech stem.
CPU, CPU_AUDIO, GPU, GPU_AUDIO = "cpu", "cpuaudio", "gpu", "gpuaudio"
SIDES = (CPU, CPU_AUDIO, GPU, GPU_AUDIO)

# The size the frames_r720 annotator saves its frames at, and the name of
# the directory they are kept under.
FRAMES_SIZE = "r720"


@dataclass(frozen=True)
class Annotator:
    outputs: tuple[str, ...]
    source: str
    side: str
    requires: tuple[str, ...] = ()
    version: int = 1


ANNOTATORS = {
    "metadata": Annotator(("metadata.parquet",), "mp4", CPU),
    "letterbox": Annotator(("letterbox.parquet",), "mp4", CPU),
    "chroma": Annotator(("chroma.parquet",), "mp4", CPU),
    "chapters": Annotator(("chapters.parquet",), "mp4", CPU),
    "subtitles": Annotator(("subtitles.parquet",), "mp4", CPU),
    "frames": Annotator(("frames.parquet",), "mp4", CPU, ("metadata",)),
    "shots": Annotator(("shots.parquet",), "mp4", CPU, ("frames",)),
    "frames_r720": Annotator(("frames/r720",), "mp4", CPU, ("metadata", "shots")),
    "dino2": Annotator(("dino2.parquet",), "mp4", GPU, ("shots",)),
    "siglip2": Annotator(("siglip2.parquet",), "mp4", GPU, ("metadata", "shots")),
    "siglip2_base": Annotator(("siglip2_base.parquet",), "mp4", GPU, ("metadata", "shots")),
    "face": Annotator(("face.parquet",), "mp4", GPU),
    "yolo": Annotator(("yolo.parquet",), "mp4", GPU),
    "tracks": Annotator(
        ("tracks.parquet", "tracks_frames.parquet"),
        "mp4",
        GPU,
        ("shots", "face", "yolo"),
    ),
    "words": Annotator(("words.parquet",), "stems", CPU_AUDIO, ("audioseg",)),
    # The speech stem is the output the cpuaudio side reads, so audioseg is
    # not done until it is there; see output_exists for where it lives.
    "audioseg": Annotator(
        ("audioseg_intervals.parquet", "audioseg_timeline.parquet", "stems/speech"),
        "audioseg",
        GPU_AUDIO,
    ),
    "praat": Annotator(("praat.parquet",), "stems", CPU_AUDIO, ("audioseg",)),
    # Reads words, praat and the audioseg timeline beside the speech stem, so
    # it follows them in a cpuaudio job.
    "laughter": Annotator(
        ("laughter.parquet", "laughter_labels.parquet"),
        "stems",
        CPU_AUDIO,
        ("audioseg", "words", "praat"),
    ),
}


VERSION_COLUMN = "version"


def stamp(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return df.with_columns(pl.lit(ANNOTATORS[name].version, pl.Int32).alias(VERSION_COLUMN))


if next(iter(ANNOTATORS)) != "metadata":
    raise RuntimeError(
        "`metadata` must be the first entry in ANNOTATORS -- every other stage "
        "is allowed to run only once it has established which files the film's "
        "annotations refer to. See the note above this check."
    )


def side_of(name: str) -> str:
    return ANNOTATORS[name].side


def names_for_side(side: str) -> list[str]:
    return [n for n, e in ANNOTATORS.items() if e.side == side]


SOURCES = ("mp4", "m4a", "audioseg", "stems")

# The annotators that read one picture at a time and nothing else, which
# are the ones an image has. The others need the frames around a frame, or
# the sound.
IMAGE_ANNOTATORS = ("dino2", "siglip2", "siglip2_base", "face", "yolo")


def outputs_of(name: str) -> tuple[str, ...]:
    return ANNOTATORS[name].outputs
