"""Colour vs black-and-white, per film."""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl

from .letterbox import Sample, usable
from .tools.values import StrPath

BW_MAX = 4.0

SCHEMA = {
    "fname": pl.Utf8,
    "n_sampled": pl.Int64,
    "n_usable": pl.Int64,
    "chroma": pl.Float32,
    "chroma_p90": pl.Float32,
    "sat": pl.Float32,
    "bw": pl.Boolean,
}


class Summary(TypedDict):
    """How much colour the usable sampled frames hold."""

    n_sampled: int
    n_usable: int
    chroma: float
    chroma_p90: float
    sat: float


def summarise(rows: Sequence[Sample] | None) -> Summary:
    rec: Summary = {"n_sampled": 0, "n_usable": 0, "chroma": 0.0, "chroma_p90": 0.0, "sat": 0.0}
    if not rows:
        return rec

    rec["n_sampled"] = len(rows)
    use = usable(rows)
    if not use:
        return rec

    vals = np.array([r["f_chroma"] for r in use], dtype=np.float64)
    rec["n_usable"] = len(use)
    rec["chroma"] = float(vals.mean())
    rec["chroma_p90"] = float(np.percentile(vals, 90))
    rec["sat"] = float(np.mean([r["f_sat"] for r in use]))
    return rec


def create_chroma(ip: StrPath, od: Path) -> None:
    out = od / "chroma.parquet"
    if out.exists():
        print(f"[{od.name}:chroma] Already exists")
        return

    t0 = time.monotonic()
    if not Path(ip).exists():
        print(f"[{od.name}:chroma] Missing film at {ip} — skipped")
        return

    from . import letterbox

    rows, _ = letterbox.sampled(ip)
    rec = summarise(rows)
    if not rec["n_usable"]:
        print(f"[{od.name}:chroma] No usable frames — recorded as unmeasured")

    df = pl.DataFrame(
        {"fname": od.name, **rec, "bw": bool(rec["chroma_p90"] < BW_MAX and rec["n_usable"] > 0)},
        schema=SCHEMA,
    )
    od.mkdir(parents=True, exist_ok=True)
    from .annotators import stamp

    stamp(df, "chroma").write_parquet(out)

    verdict = "b&w" if df["bw"][0] else "colour"
    print(
        f"[{od.name}:chroma] {verdict} "
        f"(chroma {rec['chroma']:.2f}, p90 {rec['chroma_p90']:.2f}, "
        f"sat {rec['sat']:.3f}, {rec['n_usable']}/{rec['n_sampled']} usable, "
        f"{time.monotonic() - t0:.0f}s)"
    )
