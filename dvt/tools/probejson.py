"""ffprobe's json, read without trusting its shape.

Each stream ffprobe reports becomes a Stream holding the fields the steps
read. A field that is missing, or not of the kind expected, is None. The
parsing is kept apart from running ffprobe so it can be tested on text.
"""

import json
from dataclasses import dataclass
from typing import cast


def get(obj: object, key: str) -> object:
    """obj[key] when obj is a json object holding key, else None."""
    if not isinstance(obj, dict):
        return None
    return cast("dict[str, object]", obj).get(key)


def items(obj: object) -> list[object]:
    """obj when it is a json array, else an empty list."""
    if not isinstance(obj, list):
        return []
    return cast("list[object]", obj)


def is_object(obj: object) -> bool:
    return isinstance(obj, dict)


def is_array(obj: object) -> bool:
    return isinstance(obj, list)


def as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def as_int(value: object) -> int | None:
    """An int, or a string holding one, as ffprobe writes bit_rate."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def as_float(value: object) -> float | None:
    """A number, or a string holding one, as ffprobe writes duration."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def load(text: str) -> object | None:
    """The parsed json, or None when the text is not a json object."""
    try:
        data = cast("object", json.loads(text))
    except json.JSONDecodeError:
        return None
    return data if is_object(data) else None


def parse_duration_tag(value: str) -> float | None:
    """Seconds in a Matroska DURATION tag, HH:MM:SS.fraction."""
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


@dataclass(frozen=True)
class Stream:
    """One stream as ffprobe reports it. language is None when the stream
    carries no tag or an empty one; duration comes from the stream's
    duration, or failing that from a DURATION tag."""

    index: int
    codec_type: str | None = None
    codec: str | None = None
    channels: int | None = None
    width: int | None = None
    height: int | None = None
    field_order: str | None = None
    bit_rate: int | None = None
    duration: float | None = None
    language: str | None = None


def _duration(raw: object) -> float | None:
    duration = as_float(get(raw, "duration"))
    if duration is not None:
        return duration
    tags = get(raw, "tags")
    if not is_object(tags):
        return None
    for key, value in cast("dict[str, object]", tags).items():
        upper = key.upper()
        text = as_str(value)
        if text is not None and (upper == "DURATION" or upper.startswith("DURATION-")):
            parsed = parse_duration_tag(text)
            if parsed is not None:
                return parsed
    return None


def stream(raw: object, index: int) -> Stream:
    return Stream(
        index=index,
        codec_type=as_str(get(raw, "codec_type")),
        codec=as_str(get(raw, "codec_name")),
        channels=as_int(get(raw, "channels")),
        width=as_int(get(raw, "width")),
        height=as_int(get(raw, "height")),
        field_order=as_str(get(raw, "field_order")),
        bit_rate=as_int(get(raw, "bit_rate")),
        duration=_duration(raw),
        language=as_str(get(get(raw, "tags"), "language")) or None,
    )


def streams(data: object) -> list[Stream]:
    """Every stream in a parsed ffprobe report, in order."""
    return [stream(raw, idx) for idx, raw in enumerate(items(get(data, "streams")))]


def format_field(data: object, key: str) -> object:
    """A field of the report's format section."""
    return get(get(data, "format"), key)
