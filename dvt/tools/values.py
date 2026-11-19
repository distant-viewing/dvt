"""Conversions with exactly the semantics of int() and float(), and the type
names the modules share.

The older modules read ffprobe's json and table rows as plain dicts and
convert a field with int() or float() inside a try. These give the same
answer and None where that try would have caught the error, so the code
reads the same and a type checker can follow it.
"""

import os
from collections.abc import Mapping
from typing import Any, SupportsFloat, SupportsIndex, SupportsInt, cast

type StrPath = str | os.PathLike[str]

# ffprobe's report, or a row of a table, as json.loads or polars gives it:
# a dict whose values have not been checked. Code that indexes it the old
# way takes this; new code parses it with dvt.tools.probejson instead.
type Raw = dict[str, Any]


def int_or_none(value: object) -> int | None:
    """int(value), or None where that raises TypeError or ValueError."""
    if isinstance(value, str | bytes | bytearray | SupportsInt | SupportsIndex):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def float_or_none(value: object) -> float | None:
    """float(value), or None where that raises TypeError or ValueError."""
    if isinstance(value, str | bytes | bytearray | SupportsFloat | SupportsIndex):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def mapping(value: object) -> Mapping[str, object] | None:
    """value when it is a dict, else None."""
    if isinstance(value, dict):
        return cast("Mapping[str, object]", value)
    return None


def mappings(value: object) -> list[Mapping[str, object]]:
    """The dicts in value when it is a list, in order; anything else in the
    list is passed over."""
    if not isinstance(value, list):
        return []
    found: list[Mapping[str, object]] = []
    for item in cast("list[object]", value):
        m = mapping(item)
        if m is not None:
            found.append(m)
    return found
