"""What language a film is in: the tag on its audio stream, as its container records it."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tools import language as tags

if TYPE_CHECKING:
    from .source import Source

# What metadata.parquet declared before the tag became the source: the
# curated two-letter code, or one of these. A row holding one of them is
# still read, so a film whose words were made from the curated code is not
# redone while the tag agrees with it.
OLD_SENTINELS = frozenset({"unk", "none", "other"})

_ALIGNABLE: set[str] = set()


def alignable() -> set[str] | None:
    """The languages whisperx has an aligner for, or None where whisperx is
    not installed: it comes with the audio extra, and metadata, which asks,
    with the video one."""
    if not _ALIGNABLE:
        try:
            from .align_paths import supported_languages
        except ImportError:
            return None
        _ALIGNABLE.update(supported_languages())
    return _ALIGNABLE


@dataclass(frozen=True)
class Language:
    code: str | None  # what whisperx transcribes and aligns as; None skips
    declared: str | None  # the tag as recorded, and what metadata.parquet keeps
    note: str


def _from_code(code: str | None, declared: str | None, why: str, source: str) -> Language:
    known = alignable()
    if code is not None and known is not None and code not in known:
        return Language(None, declared, f"{why}, which whisperx has no aligner for")
    return Language(code, declared, f"{why}, per {source}")


def from_tag(tag: str | None, source: str) -> Language:
    if tag is None:
        return Language(None, None, f"no audio stream, per {source}")
    tag = tags.normalise(tag)
    return _from_code(tags.code(tag), tag, tags.describe(tag), source)


def resolve(source: "Source") -> Language:
    """The language of a source, from the tag its container gives the first
    audio stream."""
    if source.container is None:
        return Language(None, None, f"nothing is known of the container of {source.name}")
    tag = source.container.get("audio_lang")
    return from_tag(tag if isinstance(tag, str) else None, "its container")


def from_metadata(declared: str | None) -> Language:
    """The language metadata.parquet recorded for this film."""
    source = "metadata.parquet"
    if declared is None:
        return Language(None, None, f"{source} records no audio language for this film")
    if declared in OLD_SENTINELS:
        return Language(None, declared, f"{declared} per the curated table, in {source}")
    if len(declared) == 2:
        return _from_code(declared, declared, f"{declared} per the curated table", source)
    return from_tag(declared, source)
