"""dvt VIDEO --out DIR --models DIR: every table for one video, under one directory."""

import argparse
import sys
from pathlib import Path

from . import config
from .annotators import ANNOTATORS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dvt", description="Run the annotators over one video file.")
    ap.add_argument("video", type=Path, nargs="?", help="the video file")
    ap.add_argument(
        "--out",
        type=Path,
        help="where everything made from the video goes: "
        "annotations/<name>/ holds the tables, audio/ the "
        "sound, frames/ the saved frames, stems/ the "
        "separated stems",
    )
    ap.add_argument(
        "--models", type=Path, help=f"the model directory; ${config.MODELS_ENV} if not given"
    )
    ap.add_argument(
        "--tmp",
        type=Path,
        help="scratch space for the separation, a few GB a film; the system's by default",
    )
    ap.add_argument(
        "--annotators",
        help="comma-separated names to run, all by default: " + ", ".join(ANNOTATORS),
    )
    ap.add_argument(
        "--list", action="store_true", help="list the annotators and what each writes, and stop"
    )
    args = ap.parse_args(argv)

    if args.list:
        for name, est in ANNOTATORS.items():
            needs = f"  after {', '.join(est.requires)}" if est.requires else ""
            print(f"{name:<13} {est.side:<9} {', '.join(est.outputs)}{needs}")
        return 0

    names = list(ANNOTATORS)
    if args.annotators:
        names = [n.strip() for n in args.annotators.split(",") if n.strip()]
        unknown = [n for n in names if n not in ANNOTATORS]
        if unknown:
            ap.error(f"unknown annotator(s): {', '.join(unknown)}")
    if args.video is None or args.out is None:
        ap.error("a video and --out are both needed")
    if not args.video.is_file():
        ap.error(f"{args.video} is not a file")
    try:
        config.configure(args.models if args.models is not None else config.models(), args.tmp)
    except RuntimeError as e:
        ap.error(str(e))

    from . import run
    from .source import Source

    source = Source.under(args.video, args.out)
    run.load(names)
    said = run.extract(source)
    if said is not None:
        print(f"[{source.name}:audio] {said}")
    run.annotate(source, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
