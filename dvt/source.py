"""One file to annotate: where it is, what its container says, and where what
is made from it goes."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .tools import filehash, probe
from .tools.values import StrPath

STEM_SUFFIX = ".flac"
SPEECH_STEM = "speech"
STEM = "{stem}"


@dataclass(frozen=True)
class Source:
    video: Path
    # The directory of tables. Its name is the file's name in every log
    # line and table that states one.
    out: Path
    # The m4a taken from the video, or where it would be. A video with no
    # sound has a marker beside that path instead; see dvt.tools.filehash.
    audio: Path
    # The directory of saved frames.
    frames: Path
    # One separated stem, with STEM where the stem's name goes.
    stems: str
    # sar, dar, width, height and audio_lang as dvt.tools.probe.extract_meta
    # gives them, or None when nothing is known of the container.
    container: Mapping[str, object] | None = None

    @property
    def name(self) -> str:
        return self.out.name

    def stem(self, stem: str) -> Path:
        return Path(self.stems.replace(STEM, stem))

    @property
    def speech(self) -> Path:
        return self.stem(SPEECH_STEM)

    @property
    def sound(self) -> Path | None:
        """The m4a, the marker that says there is none, or None when the
        sound has not been taken out yet."""
        return filehash.audio_path_for(self.audio)

    @classmethod
    def under(cls, video: StrPath, root: StrPath) -> "Source":
        """A video laid out under one directory, probed here: the tables
        in annotations/<name>/, the sound in audio/, the saved frames in
        frames/<name>/ and the stems in stems/<stem>/."""
        video, root = Path(video), Path(root)
        name = video.stem
        found = probe.probe_video(video)
        return cls(
            video=video,
            out=root / "annotations" / name,
            audio=root / "audio" / f"{name}.m4a",
            frames=root / "frames" / name,
            stems=str(root / "stems" / STEM / (name + STEM_SUFFIX)),
            container=None if found is None else probe.extract_meta(found),
        )
