"""Display aspect ratio: turning measured pixels into what a viewer sees."""

from collections.abc import Mapping
from typing import TypedDict

STANDARD_AR = (1.19, 1.33, 1.37, 1.66, 1.78, 1.85, 2.00, 2.20, 2.35, 2.39, 2.55, 2.76)

AR_MAX = 0.05

AR_TOL = 0.01


def parse_ratio(s: object) -> float | None:
    if s is None:
        return None
    s = str(s)
    if ":" not in s:
        return None
    num, _, den = s.partition(":")
    try:
        num, den = float(num), float(den)
    except ValueError:
        return None
    if not (num > 0 and den > 0):
        return None
    return num / den


def nearest_standard(ar: float) -> tuple[float, float]:
    near = min(STANDARD_AR, key=lambda s: abs(s - ar))
    return near, abs(near - ar)


class Aspect(TypedDict):
    """What metadata.parquet records of the display aspect."""

    sar_ratio: float | None
    ar_known: bool
    display_ar: float
    ar_nearest: float
    ar_dist: float
    ar_break: bool
    ar_note: str


def _whole(value: object) -> int:
    """int(value), for the number or numeric text a container field holds."""
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"not a number: {value!r}")


def aspect_fields(
    content_ar: float, width: int, height: int, row: Mapping[str, object] | None
) -> Aspect:
    """`row` is what the container says of the file, the sar, dar, width and
    height of dvt.tools.probe.extract_meta, or None when nothing is known.
    The notes call it metadata.csv, the corpus inventory those facts are
    kept in, and are stored in the table, so the wording stays."""
    rec: Aspect = {
        "sar_ratio": None,
        "ar_known": False,
        "display_ar": float(content_ar),
        "ar_nearest": 0.0,
        "ar_dist": 0.0,
        "ar_break": False,
        "ar_note": "",
    }

    notes: list[str] = []
    sar = None

    if row is None:
        notes.append("not in metadata.csv")
    else:
        sar = parse_ratio(row.get("sar"))
        dar = parse_ratio(row.get("dar"))

        cw, ch = row.get("width"), row.get("height")
        if cw and ch and (_whole(cw) != int(width) or _whole(ch) != int(height)):
            notes.append(f"dims {width}x{height} != metadata.csv {cw}x{ch}")
            sar = None

        elif sar is None and dar is not None and height:
            sar = dar / (width / height)
            notes.append("sar derived from dar")

        if sar is not None and dar is not None and height:
            expect = (width / height) * sar
            if abs(expect - dar) > max(AR_TOL, AR_TOL * expect):
                notes.append(f"dar {dar:.4f} != (w/h)*sar {expect:.4f}")
                rec["ar_break"] = True
                sar = None

        if sar is None and not notes:
            notes.append("no usable sar")

    if sar is not None:
        rec["sar_ratio"] = float(sar)
        rec["ar_known"] = True
        rec["display_ar"] = float(content_ar) * float(sar)

    near, dist = nearest_standard(rec["display_ar"])
    rec["ar_nearest"] = float(near)
    rec["ar_dist"] = float(dist)
    rec["ar_note"] = "; ".join(notes)
    return rec
